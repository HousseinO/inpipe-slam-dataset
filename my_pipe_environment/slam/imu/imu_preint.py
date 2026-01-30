#!/usr/bin/env python3
"""
IMU preintegration dead-reckoning with GTSAM 4.2 (Python) for ROS 2 Humble.

Subscribes:
  /inpipe_bot/imu/data   (sensor_msgs/Imu)

Publishes:
  /imu_preint_odom       (nav_msgs/Odometry)
  /imu_preint_path       (nav_msgs/Path)

Features:
- Uses GTSAM PreintegratedImuMeasurements (Forster-style).
- Auto-calibrates initial orientation & IMU biases from first samples (assumes static).
- IMU-only dead reckoning: expect drift, but no insane nonsense.

Make sure:
- GTSAM Python is installed and importable.
- imu_link publishes orientation + accel + gyro via gazebo_ros_imu_sensor.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion

import gtsam
from gtsam import (
    PreintegrationParams,
    PreintegratedImuMeasurements,
    Rot3,
    Point3,
    Pose3,
    NavState,
    imuBias,
)


def quat_from_rot3(R: Rot3) -> Quaternion:
    """
    Convert gtsam.Rot3 to geometry_msgs/Quaternion
    via rotation matrix -> quaternion (API-agnostic).
    """
    M = np.array(R.matrix())  # 3x3

    tr = M[0, 0] + M[1, 1] + M[2, 2]

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (M[2, 1] - M[1, 2]) / S
        y = (M[0, 2] - M[2, 0]) / S
        z = (M[1, 0] - M[0, 1]) / S
    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
        S = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
        w = (M[2, 1] - M[1, 2]) / S
        x = 0.25 * S
        y = (M[0, 1] + M[1, 0]) / S
        z = (M[0, 2] + M[2, 0]) / S
    elif M[1, 1] > M[2, 2]:
        S = np.sqrt(1.0 - M[0, 0] + M[1, 1] - M[2, 2]) * 2.0
        w = (M[0, 2] - M[2, 0]) / S
        x = (M[0, 1] + M[1, 0]) / S
        y = 0.25 * S
        z = (M[1, 2] + M[2, 1]) / S
    else:
        S = np.sqrt(1.0 - M[0, 0] - M[1, 1] + M[2, 2]) * 2.0
        w = (M[1, 0] - M[0, 1]) / S
        x = (M[0, 2] + M[2, 0]) / S
        y = (M[1, 2] + M[2, 1]) / S
        z = 0.25 * S

    q = Quaternion()
    q.w = float(w)
    q.x = float(x)
    q.y = float(y)
    q.z = float(z)
    return q


class ImuPreintegrationNode(Node):
    def __init__(self):
        super().__init__("imu_preintegration_node")

        # -------- Parameters --------
        self.declare_parameter("imu_topic", "/inpipe_bot/imu/data")
        self.declare_parameter("odom_topic", "/imu_preint_odom")
        self.declare_parameter("path_topic", "/imu_preint_path")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        # From your Gazebo IMU: 0.03 m/s^2 @ 200 Hz -> ~0.424 cont.
        #                        0.001 rad/s  @ 200 Hz -> ~0.0141 cont.
        self.declare_parameter("gravity_mag", 9.81)
        self.declare_parameter("accel_noise_sigma", 0.424)
        self.declare_parameter("gyro_noise_sigma", 0.0141)

        # extrinsic base_link -> imu_link (T_BI); default identity
        self.declare_parameter("T_BI_tx", 0.0)
        self.declare_parameter("T_BI_ty", 0.0)
        self.declare_parameter("T_BI_tz", 0.0)
        self.declare_parameter("T_BI_qx", 0.0)
        self.declare_parameter("T_BI_qy", 0.0)
        self.declare_parameter("T_BI_qz", 0.0)
        self.declare_parameter("T_BI_qw", 1.0)

        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value

        g = self.get_parameter("gravity_mag").get_parameter_value().double_value
        accel_noise_sigma = self.get_parameter("accel_noise_sigma").get_parameter_value().double_value
        gyro_noise_sigma = self.get_parameter("gyro_noise_sigma").get_parameter_value().double_value

        # -------- GTSAM Preintegration Setup --------

        params = PreintegrationParams.MakeSharedU(g)  # gravity along -Z
        I3 = np.identity(3)
        params.setAccelerometerCovariance((accel_noise_sigma ** 2) * I3)
        params.setGyroscopeCovariance((gyro_noise_sigma ** 2) * I3)
        params.setIntegrationCovariance(1e-8 * I3)

        # Bias will be estimated from calibration (start with zero)
        self.bias = imuBias.ConstantBias(
            np.zeros(3),
            np.zeros(3)
        )

        self.preint = PreintegratedImuMeasurements(params, self.bias)

        # NavState will be set after calibration
        self.navstate = None

        # -------- Extrinsic T_BI: base_link -> imu_link --------

        T_BI_tx = self.get_parameter("T_BI_tx").get_parameter_value().double_value
        T_BI_ty = self.get_parameter("T_BI_ty").get_parameter_value().double_value
        T_BI_tz = self.get_parameter("T_BI_tz").get_parameter_value().double_value
        T_BI_qx = self.get_parameter("T_BI_qx").get_parameter_value().double_value
        T_BI_qy = self.get_parameter("T_BI_qy").get_parameter_value().double_value
        T_BI_qz = self.get_parameter("T_BI_qz").get_parameter_value().double_value
        T_BI_qw = self.get_parameter("T_BI_qw").get_parameter_value().double_value

        R_BI = Rot3.Quaternion(T_BI_qw, T_BI_qx, T_BI_qy, T_BI_qz)
        t_BI = Point3(T_BI_tx, T_BI_ty, T_BI_tz)
        self.T_BI = Pose3(R_BI, t_BI)  # base -> imu

        # -------- ROS I/O --------

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=4000,
        )

        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, qos)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.path_pub = self.create_publisher(Path, path_topic, 10)

        self.path = Path()
        self.path.header.frame_id = self.world_frame

        self.last_time = None

        # -------- Calibration state --------
        self.calib_samples = 400          # ~2 seconds at 200 Hz
        self.calib_count = 0
        self.calib_acc = []
        self.calib_gyro = []
        self.R0 = None

        self.calibrated = False

        self.get_logger().info(
            f"IMU preintegration node started:\n"
            f"  imu_topic   = {imu_topic}\n"
            f"  odom_topic  = {odom_topic}\n"
            f"  path_topic  = {path_topic}\n"
            f"  world_frame = {self.world_frame}\n"
            f"  base_frame  = {self.base_frame}\n"
            f"  calibrating for first {self.calib_samples} samples (keep robot still)"
        )

    # ------------ IMU Callback ------------

    def imu_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # --- Calibration phase: estimate R0, biases ---
        if not self.calibrated:
            # On first sample, grab orientation as initial R0
            if self.R0 is None:
                q = msg.orientation
                # If orientation is all zeros, ignore; Gazebo plugin should fill it
                if abs(q.w) < 1e-6 and abs(q.x) < 1e-6 and abs(q.y) < 1e-6 and abs(q.z) < 1e-6:
                    # No valid orientation yet
                    return
                self.R0 = Rot3.Quaternion(q.w, q.x, q.y, q.z)

            # Collect accel & gyro
            a = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ], dtype=float)
            w = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ], dtype=float)

            self.calib_acc.append(a)
            self.calib_gyro.append(w)
            self.calib_count += 1

            if self.calib_count >= self.calib_samples:
                self.finish_calibration()
                self.last_time = t
            return

        # --- From here: normal preintegration ---

        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        #print(dt)

        # handle weird dt
        if dt <= 0.0 or dt > 0.1:
            self.get_logger().warn(f"Skipping IMU sample due to abnormal dt={dt:.6f}")
            self.last_time = t
            self.preint.resetIntegrationAndSetBias(self.bias)
            return

        self.last_time = t

        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=float)
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=float)

        # 1) integrate this interval
        self.preint.integrateMeasurement(acc, gyro, dt)

        # 2) predict new NavState
        new_state = self.preint.predict(self.navstate, self.bias)

        # 3) update & reset
        self.navstate = new_state
        self.preint.resetIntegrationAndSetBias(self.bias)

        # 4) IMU pose in world
        T_WI = self.navstate.pose()

        # 5) base_link pose
        T_WB = T_WI.compose(self.T_BI.inverse())

        # 6) publish odometry
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = float(T_WB.x())
        odom.pose.pose.position.y = float(T_WB.y())
        odom.pose.pose.position.z = float(T_WB.z())
        odom.pose.pose.orientation = quat_from_rot3(T_WB.rotation())

        # velocity
        if hasattr(self.navstate, "v"):
            v_W = np.array(self.navstate.v())
        else:
            v_W = np.array(self.navstate.velocity())

        odom.twist.twist.linear.x = float(v_W[0])
        odom.twist.twist.linear.y = float(v_W[1])
        odom.twist.twist.linear.z = float(v_W[2])

        # angular velocity from IMU (imu/base frame)
        odom.twist.twist.angular.x = gyro[0]
        odom.twist.twist.angular.y = gyro[1]
        odom.twist.twist.angular.z = gyro[2]

        self.odom_pub.publish(odom)

        # 7) path
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.path.header.stamp = odom.header.stamp
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)

    # ------------ Calibration Helper ------------

    def finish_calibration(self):
        g = self.get_parameter("gravity_mag").get_parameter_value().double_value

        acc_mean = np.mean(self.calib_acc, axis=0)
        gyro_mean = np.mean(self.calib_gyro, axis=0)

        # At rest: meas = -R^T g + b_a  =>  b_a = meas + R^T g
        g_world = np.array([0.0, 0.0, -g])
        R0_T = np.array(self.R0.matrix()).T
        ba = acc_mean + R0_T.dot(g_world)
        bg = gyro_mean

        self.bias = imuBias.ConstantBias(bg, ba)

        # Initialize NavState at origin with R0 and zero velocity
        p0 = Point3(0.0, 0.0, 0.0)
        v0 = np.zeros(3)
        self.navstate = NavState(self.R0, p0, v0)

        # Reset preintegrator with calibrated bias
        self.preint.resetIntegrationAndSetBias(self.bias)

        self.calibrated = True
        self.get_logger().info(
            f"Calibration done.\n"
            f"  R0 (from IMU orientation)\n"
            f"  gyro bias = {bg}\n"
            f"  accel bias = {ba}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ImuPreintegrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



##!/usr/bin/env python3
#"""
#IMU preintegration dead-reckoning with GTSAM 4.2 (Python)
#ROS 2 Humble.

#Subscribes:
#  /inpipe_bot/imu/data  (sensor_msgs/Imu)

#Publishes:
#  /imu_preint_odom  (nav_msgs/Odometry)
#  /imu_preint_path  (nav_msgs/Path)

#Notes:
#- IMU-only dead reckoning: will drift (expected).
#- Uses PreintegratedImuMeasurements (Forster-style).
#"""

#import numpy as np

#import rclpy
#from rclpy.node import Node
#from rclpy.qos import (
#    QoSProfile,
#    QoSReliabilityPolicy,
#    QoSHistoryPolicy,
#)

#from sensor_msgs.msg import Imu
#from nav_msgs.msg import Odometry, Path
#from geometry_msgs.msg import PoseStamped, Quaternion

#import gtsam
#from gtsam import (
#    PreintegrationParams,
#    PreintegratedImuMeasurements,
#    Rot3,
#    Point3,
#    Pose3,
#    NavState,
#    imuBias,
#)


#def quat_from_rot3(R: Rot3) -> Quaternion:
#    """
#    Convert gtsam.Rot3 to geometry_msgs/Quaternion.

#    Uses rotation matrix -> quaternion, so it works even if
#    your bindings don't expose R.quaternion().
#    """
#    M = np.array(R.matrix())  # 3x3

#    tr = M[0, 0] + M[1, 1] + M[2, 2]

#    if tr > 0.0:
#        S = np.sqrt(tr + 1.0) * 2.0
#        w = 0.25 * S
#        x = (M[2, 1] - M[1, 2]) / S
#        y = (M[0, 2] - M[2, 0]) / S
#        z = (M[1, 0] - M[0, 1]) / S
#    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
#        S = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
#        w = (M[2, 1] - M[1, 2]) / S
#        x = 0.25 * S
#        y = (M[0, 1] + M[1, 0]) / S
#        z = (M[0, 2] + M[2, 0]) / S
#    elif M[1, 1] > M[2, 2]:
#        S = np.sqrt(1.0 - M[0, 0] + M[1, 1] - M[2, 2]) * 2.0
#        w = (M[0, 2] - M[2, 0]) / S
#        x = (M[0, 1] + M[1, 0]) / S
#        y = 0.25 * S
#        z = (M[1, 2] + M[2, 1]) / S
#    else:
#        S = np.sqrt(1.0 - M[0, 0] - M[1, 1] + M[2, 2]) * 2.0
#        w = (M[1, 0] - M[0, 1]) / S
#        x = (M[0, 2] + M[2, 0]) / S
#        y = (M[1, 2] + M[2, 1]) / S
#        z = 0.25 * S

#    q = Quaternion()
#    q.w = float(w)
#    q.x = float(x)
#    q.y = float(y)
#    q.z = float(z)
#    return q


#class ImuPreintegrationNode(Node):
#    def __init__(self):
#        super().__init__("imu_preintegration_node")

#        # ------------ Parameters ------------
#        self.declare_parameter("imu_topic", "/inpipe_bot/imu/data")
#        self.declare_parameter("odom_topic", "/imu_preint_odom")
#        self.declare_parameter("path_topic", "/imu_preint_path")
#        self.declare_parameter("world_frame", "odom")
#        self.declare_parameter("base_frame", "base_link")

#        self.declare_parameter("gravity_mag", 9.81)
#        self.declare_parameter("accel_noise_sigma", 0.424)   # m/s^2 / sqrt(Hz)
#        self.declare_parameter("gyro_noise_sigma", 0.0141)   # rad/s / sqrt(Hz)

#        # Extrinsic base_link -> imu_link (T_BI); default identity
#        self.declare_parameter("T_BI_tx", 0.0)
#        self.declare_parameter("T_BI_ty", 0.0)
#        self.declare_parameter("T_BI_tz", 0.0)
#        self.declare_parameter("T_BI_qx", 0.0)
#        self.declare_parameter("T_BI_qy", 0.0)
#        self.declare_parameter("T_BI_qz", 0.0)
#        self.declare_parameter("T_BI_qw", 1.0)

#        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
#        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
#        path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
#        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
#        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value

#        g = self.get_parameter("gravity_mag").get_parameter_value().double_value
#        accel_noise_sigma = self.get_parameter("accel_noise_sigma").get_parameter_value().double_value
#        gyro_noise_sigma = self.get_parameter("gyro_noise_sigma").get_parameter_value().double_value

#        # ------------ GTSAM Preintegration Setup ------------

#        params = PreintegrationParams.MakeSharedU(g)
#        I3 = np.identity(3)

#        params.setAccelerometerCovariance((accel_noise_sigma ** 2) * I3)
#        params.setGyroscopeCovariance((gyro_noise_sigma ** 2) * I3)
#        params.setIntegrationCovariance(1e-8 * I3)

#        # Zero bias (sim)
#        self.bias = imuBias.ConstantBias(
#            np.zeros(3),
#            np.zeros(3)
#        )

#        self.preint = PreintegratedImuMeasurements(params, self.bias)

#        # Initial NavState
#        R0 = Rot3()  # identity
#        p0 = Point3(0.0, 0.0, 0.0)
#        v0 = np.zeros(3)
#        self.navstate = NavState(R0, p0, v0)

#        # ------------ Extrinsic T_BI: base_link -> imu_link ------------

#        T_BI_tx = self.get_parameter("T_BI_tx").get_parameter_value().double_value
#        T_BI_ty = self.get_parameter("T_BI_ty").get_parameter_value().double_value
#        T_BI_tz = self.get_parameter("T_BI_tz").get_parameter_value().double_value
#        T_BI_qx = self.get_parameter("T_BI_qx").get_parameter_value().double_value
#        T_BI_qy = self.get_parameter("T_BI_qy").get_parameter_value().double_value
#        T_BI_qz = self.get_parameter("T_BI_qz").get_parameter_value().double_value
#        T_BI_qw = self.get_parameter("T_BI_qw").get_parameter_value().double_value

#        R_BI = Rot3.Quaternion(T_BI_qw, T_BI_qx, T_BI_qy, T_BI_qz)
#        t_BI = Point3(T_BI_tx, T_BI_ty, T_BI_tz)
#        self.T_BI = Pose3(R_BI, t_BI)  # base -> imu

#        # ------------ ROS I/O ------------

#        qos = QoSProfile(
#            reliability=QoSReliabilityPolicy.RELIABLE,
#            history=QoSHistoryPolicy.KEEP_LAST,
#            depth=2000,
#        )

#        self.imu_sub = self.create_subscription(
#            Imu, imu_topic, self.imu_callback, qos
#        )
#        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
#        self.path_pub = self.create_publisher(Path, path_topic, 10)

#        self.path = Path()
#        self.path.header.frame_id = self.world_frame

#        self.last_time = None

#        self.get_logger().info(
#            f"IMU preintegration node started:\n"
#            f"  imu_topic      = {imu_topic}\n"
#            f"  odom_topic     = {odom_topic}\n"
#            f"  path_topic     = {path_topic}\n"
#            f"  world_frame    = {self.world_frame}\n"
#            f"  base_frame     = {self.base_frame}"
#        )

#    # ------------ IMU Callback ------------

#    def imu_callback(self, msg: Imu):
#        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

#        if self.last_time is None:
#            self.last_time = t
#            return

#        dt = t - self.last_time

#        # Skip weird dt (sim reset / long pause)
#        if dt <= 0.0 or dt > 0.1:
#            self.get_logger().warn(f"Skipping IMU sample due to abnormal dt={dt:.6f}")
#            self.last_time = t
#            self.preint.resetIntegrationAndSetBias(self.bias)
#            return

#        self.last_time = t

#        acc = np.array([
#            msg.linear_acceleration.x,
#            msg.linear_acceleration.y,
#            msg.linear_acceleration.z,
#        ], dtype=float)

#        gyro = np.array([
#            msg.angular_velocity.x,
#            msg.angular_velocity.y,
#            msg.angular_velocity.z,
#        ], dtype=float)

#        # 1) Integrate this measurement
#        self.preint.integrateMeasurement(acc, gyro, dt)

#        # 2) Predict new state
#        new_state = self.preint.predict(self.navstate, self.bias)

#        # 3) Update & reset integrator
#        self.navstate = new_state
#        self.preint.resetIntegrationAndSetBias(self.bias)

#        # 4) IMU pose in world
#        T_WI = self.navstate.pose()  # Pose3

#        # 5) base_link pose: T_WB = T_WI * T_BI^{-1}
#        T_WB = T_WI.compose(self.T_BI.inverse())

#        # 6) Odometry message
#        odom = Odometry()
#        odom.header.stamp = msg.header.stamp
#        odom.header.frame_id = self.world_frame
#        odom.child_frame_id = self.base_frame

#        odom.pose.pose.position.x = float(T_WB.x())
#        odom.pose.pose.position.y = float(T_WB.y())
#        odom.pose.pose.position.z = float(T_WB.z())
#        odom.pose.pose.orientation = quat_from_rot3(T_WB.rotation())

#        # Velocity in world
#        if hasattr(self.navstate, "v"):
#            v_W = np.array(self.navstate.v())
#        else:
#            v_W = np.array(self.navstate.velocity())

#        odom.twist.twist.linear.x = float(v_W[0])
#        odom.twist.twist.linear.y = float(v_W[1])
#        odom.twist.twist.linear.z = float(v_W[2])

#        # Angular velocity from IMU (imu/base frame)
#        odom.twist.twist.angular.x = gyro[0]
#        odom.twist.twist.angular.y = gyro[1]
#        odom.twist.twist.angular.z = gyro[2]

#        self.odom_pub.publish(odom)

#        # 7) Path message
#        pose = PoseStamped()
#        pose.header = odom.header
#        pose.pose = odom.pose.pose
#        self.path.header.stamp = odom.header.stamp
#        self.path.poses.append(pose)
#        self.path_pub.publish(self.path)


#def main(args=None):
#    rclpy.init(args=args)
#    node = ImuPreintegrationNode()
#    try:
#        rclpy.spin(node)
#    except KeyboardInterrupt:
#        pass
#    finally:
#        node.destroy_node()
#        rclpy.shutdown()


#if __name__ == "__main__":
#    main()

