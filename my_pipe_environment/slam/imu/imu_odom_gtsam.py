#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

import numpy as np
import gtsam
from gtsam import (
    PreintegrationParams,
    PreintegratedImuMeasurements,
    imuBias,
    Pose3,
    Rot3,
    Point3,
    NonlinearFactorGraph,
    Values,
    PriorFactorPose3,
    noiseModel,
    ISAM2,
    ISAM2Params,
    ImuFactor,
    NavState,
)

# Check optional types
HAS_PriorFactorVector = hasattr(gtsam, "PriorFactorVector")
if HAS_PriorFactorVector:
    from gtsam import PriorFactorVector

HAS_PriorFactorConstantBias = hasattr(gtsam, "PriorFactorConstantBias")
if HAS_PriorFactorConstantBias:
    from gtsam import PriorFactorConstantBias


class ImuIcpFusionNode(Node):
    """
    IMU + ICP odom fusion with GTSAM (compatible with your GTSAM build):

    - Single global bias b0: imuBias.ConstantBias (zero or near-zero)
    - IMU preintegration between keyframes:
        ImuFactor(x_k, v_k, x_{k+1}, v_{k+1}, b0)
    - ICP odometry as soft pose prior on each new pose x_k
    - iSAM2 incremental optimization
    - Publishes:
        /fused_odom  (nav_msgs/Odometry)
        /fused_path  (nav_msgs/Path)
    """

    def __init__(self):
        super().__init__('imu_icp_fusion_node')

        # --- Topics / frames ---
        self.imu_topic = '/inpipe_bot/imu/data'
        self.odom_topic = '/pipe/odom'
        self.frame_id = 'odom'
        self.child_frame_id = 'base_link'

        # --- IMU noise config (from your Gazebo IMU) ---
        imu_rate = 200.0
        dt = 1.0 / imu_rate

        acc_stddev = 0.03
        gyro_stddev = 0.001
        acc_bias_rw_stddev = 0.001
        gyro_bias_rw_stddev = 0.0001

        sigma_acc = acc_stddev / math.sqrt(dt)
        sigma_gyro = gyro_stddev / math.sqrt(dt)
        sigma_acc_bias = acc_bias_rw_stddev / math.sqrt(dt)
        sigma_gyro_bias = gyro_bias_rw_stddev / math.sqrt(dt)

        # Gravity magnitude (world Z-up, IMU aligned)
        g = 9.81
        self.params = PreintegrationParams.MakeSharedU(g)

        self.params.setAccelerometerCovariance((sigma_acc ** 2) * np.eye(3))
        self.params.setGyroscopeCovariance((sigma_gyro ** 2) * np.eye(3))
        self.params.setIntegrationCovariance((1e-4 ** 2) * np.eye(3))

        if hasattr(self.params, "setBiasAccCovariance"):
            self.params.setBiasAccCovariance((sigma_acc_bias ** 2) * np.eye(3))
        else:
            self.get_logger().warn("setBiasAccCovariance not available; skipping.")
        if hasattr(self.params, "setBiasOmegaCovariance"):
            self.params.setBiasOmegaCovariance((sigma_gyro_bias ** 2) * np.eye(3))
        else:
            self.get_logger().warn("setBiasOmegaCovariance not available; skipping.")

        # --- Key generators ---
        self.pose_key = lambda k: gtsam.symbol('x', k)
        self.vel_key = lambda k: gtsam.symbol('v', k)
        self.bias_key = lambda: gtsam.symbol('b', 0)  # single global bias key 'b0'

        # --- Single global bias: start at zero ---
        self.bias0 = imuBias.ConstantBias(
            np.zeros(3),  # accel bias
            np.zeros(3),  # gyro bias
        )

        # Preintegrator uses fixed bias0
        self.preintegrated = PreintegratedImuMeasurements(self.params, self.bias0)

        # Factor graph + initial values
        self.graph = NonlinearFactorGraph()
        self.initial_values = Values()

        # iSAM2 (handle old API)
        isam_params = ISAM2Params()
        if hasattr(isam_params, "setRelinearizeThreshold"):
            isam_params.setRelinearizeThreshold(0.01)
        else:
            isam_params.relinearizeThreshold = 0.01

        if hasattr(isam_params, "setRelinearizeSkip"):
            isam_params.setRelinearizeSkip(1)
        else:
            isam_params.relinearizeSkip = 1

        self.isam = ISAM2(isam_params)

        # State
        self.k = 0
        self.initialized = False
        self.last_imu_time = None

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/fused_odom', 10)
        self.path_pub = self.create_publisher(Path, '/fused_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        # Subscribers
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, 200)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 50)

        self.get_logger().info(
            f"IMU+ICP fusion node (fixed bias) started.\n"
            f"  IMU:  {self.imu_topic}\n"
            f"  Odom: {self.odom_topic}"
        )

    # ---------- Helpers ----------

    def _delta_t_preint(self) -> float:
        """Compatibility wrapper for preintegrated time."""
        if hasattr(self.preintegrated, "deltaT"):
            return self.preintegrated.deltaT()
        if hasattr(self.preintegrated, "deltaTij"):
            return self.preintegrated.deltaTij()
        return 0.0

    @staticmethod
    def _pose3_from_odom(msg: Odometry) -> Pose3:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        R = Rot3.Quaternion(q.w, q.x, q.y, q.z)
        t = Point3(p.x, p.y, p.z)
        return Pose3(R, t)

    @staticmethod
    def _vec3(p):
        """Get (x,y,z) from Point3-like or numpy array-like."""
        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            return float(p.x()), float(p.y()), float(p.z())
        return float(p[0]), float(p[1]), float(p[2])

    @staticmethod
    def _quat(q):
        """Get (x,y,z,w) from quaternion-like or array-like."""
        if hasattr(q, "x") and callable(q.x):
            return float(q.x()), float(q.y()), float(q.z()), float(q.w())
        if hasattr(q, "x"):
            return float(q.x), float(q.y), float(q.z), float(q.w)
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # ---------- Callbacks ----------

    def imu_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        if dt <= 0.0:
            return
        self.last_imu_time = t

        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=float)
        omega = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=float)

        self.preintegrated.integrateMeasurement(acc, omega, dt)

    def odom_callback(self, msg: Odometry):
        """
        Each incoming ICP odom is a keyframe:
        - Uses accumulated IMU (preintegrated) between last keyframe and now
        - Adds ImuFactor(x_k, v_k, x_{k+1}, v_{k+1}, b0)
        - Adds PriorFactorPose3(x_{k+1}, ICP_pose)
        - Runs iSAM2, publishes fused pose/path
        """
        T_odom = self._pose3_from_odom(msg)

        # First ICP: initialize graph
        if not self.initialized:
            self._initialize_graph(T_odom, msg.header.stamp)
            return

        # Need IMU integration between keyframes
        if self._delta_t_preint() <= 0.0:
            return

        k_prev = self.k
        k_new = self.k + 1

        x_prev = self.pose_key(k_prev)
        v_prev = self.vel_key(k_prev)
        x_new = self.pose_key(k_new)
        v_new = self.vel_key(k_new)
        b0 = self.bias_key()

        # 1) IMU factor
        imu_factor = ImuFactor(
            x_prev, v_prev,
            x_new, v_new,
            b0,
            self.preintegrated
        )
        self.graph.add(imu_factor)

        # 2) ICP odom as soft prior on new pose
        icp_pose_noise = noiseModel.Diagonal.Sigmas(
            np.array([
                0.05, 0.05, 0.05,   # roll, pitch, yaw [rad]
                0.02, 0.02, 0.02    # x, y, z [m]
            ])
        )
        self.graph.add(PriorFactorPose3(x_new, T_odom, icp_pose_noise))

        # 3) Initial guess for new pose/vel
        result_prev = self.isam.calculateEstimate()
        prev_pose = result_prev.atPose3(x_prev)
        prev_vel = result_prev.atVector(v_prev)
        prev_state = NavState(prev_pose, prev_vel)

        predicted_state = self.preintegrated.predict(prev_state, self.bias0)

        self.initial_values.insert(x_new, predicted_state.pose())
        self.initial_values.insert(v_new, predicted_state.velocity())

        # 4) Optimize
        self.isam.update(self.graph, self.initial_values)
        result = self.isam.calculateEstimate()

        # Reset for next interval
        self.graph = NonlinearFactorGraph()
        self.initial_values = Values()
        self.preintegrated = PreintegratedImuMeasurements(self.params, self.bias0)

        # 5) Publish fused pose
        fused_pose = result.atPose3(x_new)
        self._publish_fused_states(fused_pose, msg.header.stamp)

        self.k = k_new

    # ---------- Initialization ----------

    def _initialize_graph(self, T0: Pose3, stamp):
        self.get_logger().info("Initializing factor graph with first ICP odom pose.")

        x0 = self.pose_key(0)
        v0 = self.vel_key(0)
        b0 = self.bias_key()

        # Prior on initial pose from ICP
        pose_prior_noise = noiseModel.Diagonal.Sigmas(
            np.array([
                1e-3, 1e-3, 1e-3,   # roll, pitch, yaw
                1e-2, 1e-2, 1e-2    # x, y, z
            ])
        )
        self.graph.add(PriorFactorPose3(x0, T0, pose_prior_noise))

        # Prior on initial velocity (zero)
        if HAS_PriorFactorVector:
            vel_prior_noise = noiseModel.Isotropic.Sigma(3, 1e-3)
            self.graph.add(PriorFactorVector(v0, np.zeros(3), vel_prior_noise))
            self.initial_values.insert(v0, np.zeros(3))
        else:
            self.get_logger().warn("PriorFactorVector not available; skipping vel prior.")
            self.initial_values.insert(v0, np.zeros(3))

        # Prior on global bias b0 (keep near zero) if API exists
        if HAS_PriorFactorConstantBias:
            bias_prior_noise = noiseModel.Isotropic.Sigma(6, 1e-6)
            self.graph.add(PriorFactorConstantBias(b0, self.bias0, bias_prior_noise))
        else:
            # No prior: just insert bias0 as initial value
            pass

        # Insert initial values
        self.initial_values.insert(x0, T0)
        self.initial_values.insert(b0, self.bias0)

        # Optimize once
        self.isam.update(self.graph, self.initial_values)
        self.graph = NonlinearFactorGraph()
        self.initial_values = Values()

        # Publish initial fused pose
        result = self.isam.calculateEstimate()
        fused_pose = result.atPose3(x0)
        self._publish_fused_states(fused_pose, stamp)

        self.initialized = True
        self.k = 0

    # ---------- Publishing ----------

    def _publish_fused_states(self, pose: Pose3, stamp):
        t = pose.translation()
        px, py, pz = self._vec3(t)

        q = pose.rotation().toQuaternion()
        qx, qy, qz, qw = self._quat(q)

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id

        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        self.odom_pub.publish(odom)

        # Path
        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(ps)
        self.path_pub.publish(self.path_msg)


def main():
    rclpy.init()
    node = ImuIcpFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

