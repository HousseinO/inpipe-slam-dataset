#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

import numpy as np
import gtsam
from gtsam import symbol
from gtsam import imuBias


class IMUGraphSLAM(Node):
    """
    Minimal IMU (+ optional odom) Graph-SLAM with GTSAM 4.2:
      - Classic PreintegratedImuMeasurements + ImuFactor
      - Single constant bias node B(0)
      - Relative ICP odom as BetweenFactorPose3 (accumulated between keyframes)
      - Rotation-only soft prior per keyframe from the IMU quaternion (tight rot / loose pos)
    """

    def __init__(self):
        super().__init__('l515_imu_graph_slam_node')

        # ---------- Parameters ----------
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        self.imu_topic   = self.declare_parameter('imu_topic',  '/inpipe_bot/imu/data').get_parameter_value().string_value
        self.odom_topic  = self.declare_parameter('odom_topic', '/pipe/odom').get_parameter_value().string_value
        self.g           = float(self.declare_parameter('gravity', 9.80665).value)
        self.keyframe_dt = float(self.declare_parameter('keyframe_dt', 0.1).value)  # small to capture elbows

        # IMU noises (match your Gazebo settings)
        sigma_gyro = float(self.declare_parameter('imu_sigma_gyro', 0.001).value)  # rad/s
        sigma_acc  = float(self.declare_parameter('imu_sigma_acc',  0.03).value)   # m/s^2

        # ---------- Graph / ISAM ----------
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        self.isam_params = gtsam.ISAM2Params()
        try:    self.isam_params.setRelinearizeThreshold(0.01)
        except: self.isam_params.relinearizeThreshold = 0.01
        try:    self.isam_params.setRelinearizeSkip(1)
        except: self.isam_params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(self.isam_params)

        # Keys
        self.X = lambda k: symbol('x', k)  # Pose3
        self.V = lambda k: symbol('v', k)  # R^3
        self.B = lambda k: symbol('b', k)  # ConstantBias

        # ---------- IMU Preintegration (classic) ----------
        imu_params = gtsam.PreintegrationParams.MakeSharedU(self.g)
        imu_params.setAccelerometerCovariance(np.eye(3) * (sigma_acc**2))
        imu_params.setGyroscopeCovariance(np.eye(3) * (sigma_gyro**2))
        imu_params.setIntegrationCovariance(np.eye(3) * 1e-6)

        self.bias0 = imuBias.ConstantBias(np.zeros(3), np.zeros(3))  # single constant bias
        self.preint = gtsam.PreintegratedImuMeasurements(imu_params, self.bias0)

        # ---------- Noise models ----------
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.deg2rad([1, 1, 1]), [0.05, 0.05, 0.05]]
        )
        self.prior_vel_noise  = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        bias_covar = np.diag([1e-3,1e-3,1e-3, 1e-4,1e-4,1e-4])
        self.prior_bias_noise = gtsam.noiseModel.Gaussian.Covariance(bias_covar)

        # Rotation-only soft prior (tight rot, very loose pos)
        self.rot_only_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.deg2rad([1.0, 1.0, 2.0]), [999.0, 999.0, 999.0]]
        )

        # ---------- Bookkeeping ----------
        self.k = 0
        self.last_imu_time = None
        self.last_kf_time  = None
        self.vel_k = np.zeros(3)

        self.last_odom = None
        self.odom_accum = gtsam.Pose3()  # identity
        self.last_imu_quat = None  # (w,x,y,z)

        # ---------- ROS I/O ----------
        self.sub_imu  = self.create_subscription(Imu,      self.imu_topic,  self.cb_imu,  200)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 20)
        self.pub_path = self.create_publisher(Path, 'graph_slam/path', 10)

        # heartbeat
        self.imu_count = 0
        self.create_timer(1.0, self.heartbeat)

        # ---------- Initialize prior state ----------
        self.initialize_prior()

        self.get_logger().info(f"IMU Graph-SLAM (GTSAM 4.2) ready. IMU: {self.imu_topic}  ODOM: {self.odom_topic}")

    # ---------------------------------------
    # Init
    # ---------------------------------------
    def initialize_prior(self):
        X0 = gtsam.Pose3()
        V0 = np.zeros(3)
        B0 = self.bias0

        self.values.insert(self.X(0), X0)
        self.values.insert(self.V(0), V0)
        self.values.insert(self.B(0), B0)

        self.graph.add(gtsam.PriorFactorPose3(self.X(0), X0, self.prior_pose_noise))
        self.graph.add(gtsam.PriorFactorVector(self.V(0), V0, self.prior_vel_noise))
        self.graph.add(gtsam.PriorFactorConstantBias(self.B(0), B0, self.prior_bias_noise))

        self.isam.update(self.graph, self.values)
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        self.k = 0
        self.last_kf_time = None

    # ---------------------------------------
    # IMU callback / keyframes
    # ---------------------------------------
    def cb_imu(self, msg: Imu):
        # keep the latest IMU quaternion for rotation-only prior
        q = msg.orientation
        self.last_imu_quat = (q.w, q.x, q.y, q.z)

        self.imu_count += 1
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        if dt <= 0.0:
            # sim time repeated or jumped backward -> skip
            return
        self.last_imu_time = t

        a = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z])
        w = np.array([msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z])
        self.preint.integrateMeasurement(a, w, dt)

        if self.last_kf_time is None:
            self.last_kf_time = t
        if (t - self.last_kf_time) >= self.keyframe_dt:
            self.add_keyframe()
            self.last_kf_time = t

    def add_keyframe(self):
        k1 = self.k + 1

        est = self.isam.calculateEstimate()
        Xk = est.atPose3(self.X(self.k)) if est.exists(self.X(self.k)) else gtsam.Pose3()
        Vk = self.vel_k
        B0 = self.bias0

        # IMU predict pose/vel to seed the next node
        nav_k  = gtsam.NavState(Xk, Vk)
        nav_k1 = self.preint.predict(nav_k, B0)
        Xk1 = nav_k1.pose()
        try:
            Vk1 = nav_k1.v()
        except AttributeError:
            Vk1 = nav_k1.velocity()
        self.vel_k = np.asarray(Vk1, dtype=float)

        # Insert initial guesses
        self.values.insert(self.X(k1), Xk1)
        self.values.insert(self.V(k1), Vk1)

        # (1) Add accumulated relative-odom factor (if any)
        if self.odom_accum is not None:
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.r_[np.deg2rad([2.0, 2.0, 3.0]), [0.03, 0.03, 0.03]]
            )
            self.graph.add(gtsam.BetweenFactorPose3(self.X(self.k), self.X(k1),
                                                    self.odom_accum, odom_noise))
            self.odom_accum = gtsam.Pose3()  # reset

        # (2) Add rotation-only prior from IMU quaternion on X(k1)
        if self.last_imu_quat is not None:
            qw, qx, qy, qz = self.last_imu_quat
            Z_R = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
            Z   = gtsam.Pose3(Z_R, gtsam.Point3(0, 0, 0))
            self.graph.add(gtsam.PriorFactorPose3(self.X(k1), Z, self.rot_only_noise))

        # (3) Add IMU factor that uses the single bias node B(0)
        imu_factor = gtsam.ImuFactor(
            self.X(self.k), self.V(self.k),
            self.X(k1),    self.V(k1),
            self.B(0),
            self.preint
        )
        self.graph.add(imu_factor)

        # Optimize & publish
        self.optimize_and_publish()

        # Prep next
        self.k = k1
        self.preint.resetIntegrationAndSetBias(B0)

    # ---------------------------------------
    # Odom callback: accumulate relative motion & set initial pose
    # ---------------------------------------
    def cb_odom(self, msg: Odometry):
        # helper
        def pose_from_msg(m):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            return gtsam.Pose3(
                gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z),
                gtsam.Point3(p.x, p.y, p.z)
            )

        # On the very first odom, add a prior on X(0) at that pose
        if self.k == 0 and self.last_odom is None:
            X0 = pose_from_msg(msg)
            self.graph.add(gtsam.PriorFactorPose3(self.X(0), X0, self.prior_pose_noise))
            self.optimize_and_publish()
            self.last_odom  = msg
            self.odom_accum = gtsam.Pose3()  # identity
            return

        Z = pose_from_msg(msg)
        if self.last_odom is None:
            self.last_odom = msg
            self.odom_accum = gtsam.Pose3()
            return

        Z_prev = pose_from_msg(self.last_odom)
        # accumulate relative odom since last keyframe
        self.odom_accum = self.odom_accum.compose(Z_prev.between(Z))
        self.last_odom = msg

    # ---------------------------------------
    # Optimize + publish path
    # ---------------------------------------
    def optimize_and_publish(self):
        self.isam.update(self.graph, self.values)
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.publish_path()

    def publish_path(self):
        est = self.isam.calculateEstimate()
        path = Path()
        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        path.header.frame_id = 'odom'  # match your odom frame in RViz/TF

        for i in range(self.k + 1):
            if not est.exists(self.X(i)):
                continue
            Xi = est.atPose3(self.X(i))

            # robust translation extraction
            t_obj = Xi.translation()
            try:
                tx, ty, tz = float(t_obj.x()), float(t_obj.y()), float(t_obj.z())
            except Exception:
                t_arr = np.asarray(t_obj).reshape(-1)
                tx, ty, tz = float(t_arr[0]), float(t_arr[1]), float(t_arr[2])

            # robust quaternion extraction
            try:
                q_obj = Xi.rotation().toQuaternion()
                qw, qx, qy, qz = float(q_obj.w()), float(q_obj.x()), float(q_obj.y()), float(q_obj.z())
            except Exception:
                q_arr = np.asarray(Xi.rotation().quaternion()).reshape(-1)
                qw, qx, qy, qz = float(q_arr[0]), float(q_arr[1]), float(q_arr[2]), float(q_arr[3])

            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = path.header.frame_id
            ps.pose.position.x = tx
            ps.pose.position.y = ty
            ps.pose.position.z = tz
            ps.pose.orientation.w = qw
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            path.poses.append(ps)

        self.pub_path.publish(path)

    # ---------------------------------------
    # Heartbeat
    # ---------------------------------------
    def heartbeat(self):
        self.get_logger().info(f"alive: k={self.k} imu={self.imu_count}")


def main():
    rclpy.init()
    node = IMUGraphSLAM()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

