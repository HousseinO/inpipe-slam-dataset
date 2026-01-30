#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
icp_imu_preint_posegraph.py
---------------------------
ROS2 + GTSAM 4.2 node that fuses:
  - IMU (accelerometer + gyroscope) via *preintegration*
  - ICP odometry (translation-dominant) from /pipe/odom

State per keyframe k:  X(k)=Pose3, V(k)=R^3, B(k)=imuBias (ba,bg)
Factors:
  - ImuFactor(Xk, Vk, Xk1, Vk1, Bk, preint)
  - BetweenFactorConstantBias(Bk, Bk1, 0, Qbias)
  - BetweenFactorPose3(Xk, Xk1, T_icp, Σ_icp)  # translation-only or loose-yaw
  - Priors on X(0), V(0), B(0)

Publishes:
  - nav_msgs/Path on topic "posegraph/path"
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

import gtsam
from gtsam import (symbol, imuBias, PreintegrationParams,
                   PreintegratedImuMeasurements, NavState)


class ICPIMU_PreintPoseGraph(Node):
    def __init__(self):
        super().__init__('icp_imu_preint_posegraph')

        # ---------------- Parameters ----------------
        # Topics
        self.declare_parameter('imu_topic', '/inpipe_bot/imu/data')
        self.declare_parameter('odom_topic', '/pipe/odom')

        # Keyframing
        self.declare_parameter('keyframe_dt', 0.5)  # seconds between keyframes

        # Gravity and IMU noises (tune!)
        self.declare_parameter('gravity', 9.80665)            # m/s^2 (upwards in params)
        self.declare_parameter('imu_accel_noise', 0.03)       # m/s^2 / √Hz
        self.declare_parameter('imu_gyro_noise', 0.001)      # rad/s / √Hz
        self.declare_parameter('imu_accel_bias_rw', 1.0e-4)   # (m/s^2)^2 / Hz
        self.declare_parameter('imu_gyro_bias_rw', 5.0e-6)    # (rad/s)^2 / Hz

        # ICP factor noise (keep similar to your original)
        self.declare_parameter('icp_rot_sigma_deg', [4.0, 4.0, 6.0])
        self.declare_parameter('icp_trans_sigma_m', [0.03, 0.03, 0.03])

        imu_topic = self.get_parameter('imu_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.keyframe_dt = float(self.get_parameter('keyframe_dt').value)

        # ---------------- GTSAM setup ----------------
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.isam = gtsam.ISAM2()

        # Symbol shorthands
        self.X = lambda k: symbol('x', k)  # Pose3
        self.V = lambda k: symbol('v', k)  # R^3
        self.B = lambda k: symbol('b', k)  # imuBias

        # Noise models
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.deg2rad([2, 2, 2]), [0.05, 0.05, 0.05]]
        )
        icp_rot = np.deg2rad(np.array(self.get_parameter('icp_rot_sigma_deg').value, dtype=float))
        icp_trans = np.array(self.get_parameter('icp_trans_sigma_m').value, dtype=float)
        self.icp_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[icp_rot, icp_trans])

        # IMU preintegration params (GTSAM 4.2)
        g = float(self.get_parameter('gravity').value)
        pim_params = PreintegrationParams.MakeSharedU(g)
        accel_noise = float(self.get_parameter('imu_accel_noise').value)
        gyro_noise = float(self.get_parameter('imu_gyro_noise').value)
        # Continuous-time covariances (σ^2)
        pim_params.setAccelerometerCovariance((accel_noise ** 2) * np.eye(3))
        pim_params.setGyroscopeCovariance((gyro_noise ** 2) * np.eye(3))
        # Integration covariance (discretization)
        pim_params.setIntegrationCovariance(1e-8 * np.eye(3))
        # Bias random walk (used by bias evolution factor; params also help marginalization)
        self.accel_bias_rw = float(self.get_parameter('imu_accel_bias_rw').value)
        self.gyro_bias_rw = float(self.get_parameter('imu_gyro_bias_rw').value)
        self.pim_params = pim_params

        # ---------------- Node state ----------------
        self.k = 0
        self.last_kf_time = None

        # IMU data handling
        self.last_imu_time = None
        self.bias = imuBias.ConstantBias()  # start zero
        self.vel = np.zeros(3)
        self.preint = PreintegratedImuMeasurements(self.pim_params, self.bias)

        # ICP accumulation between keyframes
        self.last_odom_msg = None
        self.odom_accum = gtsam.Pose3()

        # ---------------- ROS I/O ----------------
        self.create_subscription(Imu, imu_topic, self.cb_imu, 200)
        self.create_subscription(Odometry, odom_topic, self.cb_odom, 20)
        self.pub_path = self.create_publisher(Path, 'posegraph/path', 10)

        self.timer = self.create_timer(1.0, self.heartbeat)

        # Initialize priors and first state
        self.initialize_graph()

        self.get_logger().info(f"[icp_imu_preint_posegraph] Ready (IMU={imu_topic}, ICP={odom_topic})")

    # ---------------------------------------------------------------
    # Initialization: priors on X0, V0, B0 and push to ISAM2
    def initialize_graph(self):
        X0 = gtsam.Pose3()  # world origin
        V0 = np.zeros(3)
        B0 = imuBias.ConstantBias()

        self.graph.add(gtsam.PriorFactorPose3(self.X(0), X0, self.prior_pose_noise))
        self.graph.add(gtsam.PriorFactorVector(self.V(0), V0, gtsam.noiseModel.Isotropic.Sigma(3, 0.1)))
        self.graph.add(gtsam.PriorFactorConstantBias(self.B(0), B0,
                                                     gtsam.noiseModel.Isotropic.Sigma(6, 1e-2)))

        self.values.insert(self.X(0), X0)
        self.values.insert(self.V(0), V0)
        self.values.insert(self.B(0), B0)

        self.isam.update(self.graph, self.values)
        # Reset accumulators for next update batch
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        self.k = 0
        self.last_kf_time = None
        self.last_imu_time = None
        self.bias = B0
        self.vel = V0
        self.preint = PreintegratedImuMeasurements(self.pim_params, self.bias)
        self.odom_accum = gtsam.Pose3()
        self.last_odom_msg = None

    # ---------------------------------------------------------------
    # IMU callback: feed preintegrator
    def cb_imu(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        gx = float(msg.angular_velocity.x)
        gy = float(msg.angular_velocity.y)
        gz = float(msg.angular_velocity.z)

        if self.last_imu_time is not None:
            dt = t - self.last_imu_time
            # sanity clamp (ignore huge gaps/outliers)
            if 1e-5 < dt < 0.05:
#                self.preint.integrateMeasurement(acc, gyro, float(dt))
                self.preint.integrateMeasurement(np.array([ax, ay, az]),
                                                 np.array([gx, gy, gz]),
                                                 float(dt))
            else:
                # skip bad gaps/outliers
                self.get_logger().warn(f"IMU dt={dt:.3f}s out of range; skipping")
        self.last_imu_time = t

        # Keyframing driven by IMU time (ensures we always have preintegrated data)
        if self.last_kf_time is None:
            self.last_kf_time = t
        elif (t - self.last_kf_time) >= self.keyframe_dt:
            self.add_keyframe()
            self.last_kf_time = t

    # ---------------------------------------------------------------
    # Odom callback: accumulate relative ICP motion between keyframes
    def cb_odom(self, msg: Odometry):
        def pose_from_msg(m):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            return gtsam.Pose3(
                gtsam.Rot3.Quaternion(float(q.w), float(q.x), float(q.y), float(q.z)),
                gtsam.Point3(float(p.x), float(p.y), float(p.z))
            )

        Z = pose_from_msg(msg)
        if self.last_odom_msg is None:
            self.last_odom_msg = msg
            self.odom_accum = gtsam.Pose3()
            return

        Z_prev = pose_from_msg(self.last_odom_msg)
        rel = Z_prev.between(Z)
        self.odom_accum = self.odom_accum.compose(rel)
        self.last_odom_msg = msg

    # ---------------------------------------------------------------
    # Build factors for a new keyframe k -> k+1
    def add_keyframe(self):
        k = self.k
        k1 = k + 1

        # Current estimates (if exist), to seed prediction
        est = self.isam.calculateEstimate()
        Xk = est.atPose3(self.X(k)) if est.exists(self.X(k)) else gtsam.Pose3()
        Vk = est.atVector(self.V(k)) if est.exists(self.V(k)) else np.zeros(3)
        Bk = est.atConstantBias(self.B(k)) if est.exists(self.B(k)) else self.bias

        # ---- IMU factor from preintegrated measurements
        pim = self.preint  # accumulated since previous keyframe
        self.graph.add(gtsam.ImuFactor(self.X(k), self.V(k),
                                       self.X(k1), self.V(k1),
                                       self.B(k), pim))

        # ---- Bias random-walk factor (zero mean)
        # Variances here represent how much we allow bias to change per keyframe.
        # Construct a diagonal covariance with small variances.
        # keyframe interval used for bias RW scaling
        delta_t = self.keyframe_dt  # seconds
        Q_ba = (0.001 ** 2)  # from Gazebo accel bias_stddev
        Q_bg = (0.0001 ** 2) # from Gazebo gyro  bias_stddev
        bias_var = np.array([Q_ba, Q_ba, Q_ba, Q_bg, Q_bg, Q_bg]) * float(delta_t)
        bias_noise = gtsam.noiseModel.Diagonal.Variances(bias_var)
        self.graph.add(
            gtsam.BetweenFactorConstantBias(self.B(k), self.B(k1),
                                            imuBias.ConstantBias(), bias_noise)
        )


        # ---- ICP factor: use translation-only (identity rotation)
        t_rel = self.odom_accum.translation()
        try:
            tx, ty, tz = float(t_rel.x()), float(t_rel.y()), float(t_rel.z())
        except AttributeError:
            t_arr = np.asarray(t_rel).reshape(-1)
            tx, ty, tz = float(t_arr[0]), float(t_arr[1]), float(t_arr[2])
        T_rel = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(tx, ty, tz))
        self.graph.add(gtsam.BetweenFactorPose3(self.X(k), self.X(k1), T_rel, self.icp_between_noise))

        # ---- Initial guesses for new variables (Xk1, Vk1, Bk1)
        nav_k = NavState(Xk, gtsam.Point3(*Vk))
        pred = pim.predict(nav_k, Bk)        # NavState at k1 from IMU only
        Xk1_guess = pred.pose().compose(T_rel)  # let ICP nudge translation
        Vk1_guess = np.array(pred.velocity())
        Bk1_guess = Bk

        self.values.insert(self.X(k1), Xk1_guess)
        self.values.insert(self.V(k1), Vk1_guess)
        self.values.insert(self.B(k1), Bk1_guess)

        # ---- Optimize and publish
        self.optimize_and_publish()

        # ---- Prepare for next keyframe
        self.odom_accum = gtsam.Pose3()
        self.k = k1

        # Reset preintegrator from (possibly updated) bias
        est2 = self.isam.calculateEstimate()
        if est2.exists(self.B(self.k)):
            self.bias = est2.atConstantBias(self.B(self.k))
        if est2.exists(self.V(self.k)):
            self.vel = est2.atVector(self.V(self.k))
        self.preint = PreintegratedImuMeasurements(self.pim_params, self.bias)

    # ---------------------------------------------------------------
    def optimize_and_publish(self):
        self.isam.update(self.graph, self.values)

        # Clear temporary factor graph and initial values for next batch
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        # Publish current path
        self.publish_path()

    # ---------------------------------------------------------------
    def publish_path(self):
        est = self.isam.calculateEstimate()

        path = Path()
        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        path.header.frame_id = 'odom'  # choose your fixed frame

        for i in range(self.k + 1):
            if not est.exists(self.X(i)):
                continue
            Xi = est.atPose3(self.X(i))
            t = Xi.translation()
            try:
                tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
            except AttributeError:
                t_arr = np.asarray(t).reshape(-1)
                tx, ty, tz = float(t_arr[0]), float(t_arr[1]), float(t_arr[2])

            q = Xi.rotation().toQuaternion()
            qw, qx, qy, qz = float(q.w()), float(q.x()), float(q.y()), float(q.z())

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

    # ---------------------------------------------------------------
    def heartbeat(self):
        self.get_logger().info(
            f"k={self.k} nodes, path subscribers={self.pub_path.get_subscription_count()}"
        )


# --------------------------------------------------------------------
def main():
    rclpy.init()
    node = ICPIMU_PreintPoseGraph()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

