#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Preintegration node inspired by LIO-SAM's imuPreintegration.cpp, rewritten in Python for ROS 2 Humble with GTSAM 4.2.

Subscriptions:
  - /inpipe_bot/imu/data : sensor_msgs/msg/Imu
  - /pipe/odom          : nav_msgs/msg/Odometry  (ICP/LiDAR odometry)

Publications:
  - /imu_preintegration/odom : nav_msgs/msg/Odometry
  - /imu_preintegration/path : nav_msgs/msg/Path

Notes:
  * This is a compact, readable Python port that focuses on the core IMU preintegration + factor graph loop.
  * It fuses LiDAR odometry as a Pose3 "prior-like" factor at each keyframe (with configurable noise).
  * Bias is modelled as a random walk with BetweenFactor<imuBias::ConstantBias>.
  * ISAM2 is used for incremental smoothing; we relinearize each update (configurable via params).

You may want to tune all noise parameters for your sensor setup.
"""

import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

import numpy as np

# GTSAM imports (tested with 4.2.x)
import gtsam
from gtsam import (
    Pose3, Rot3, Point3, Values, NonlinearFactorGraph, ISAM2, ISAM2Params,
    PriorFactorPose3, PriorFactorVector, PriorFactorConstantBias, BetweenFactorConstantBias, ImuFactor, BetweenFactorPose3,
    PreintegrationParams, PreintegratedImuMeasurements, NavState,
)
from gtsam.symbol_shorthand import X, V, B
from gtsam.noiseModel import Diagonal


class ImuPreintegrationNode(Node):
    def __init__(self):
        super().__init__('imu_preintegration_node')

        # ---------------------- Parameters ----------------------
        self.declare_parameter('imu_topic', '/inpipe_bot/imu/data')
        self.declare_parameter('lidar_odom_topic', '/pipe/odom')
        self.declare_parameter('pub_odom_topic', '/imu_preintegration/odom')
        self.declare_parameter('pub_path_topic', '/imu_preintegration/path')

        # Gravity (m/s^2) and direction (down along Z in ENU => -9.81 on Z)
        self.declare_parameter('gravity', 9.81)

        # Accel/Gyro continuous-time noise params (per sqrt(Hz))
        self.declare_parameter('accel_noise_sigma', 0.42)   # e.g. 0.2 m/s^2/sqrt(Hz)
        self.declare_parameter('gyro_noise_sigma', 0.014)   # e.g. 0.02 rad/s/sqrt(Hz)

        # Accel/Gyro bias random walk (drift) (per sqrt(Hz))
        self.declare_parameter('accel_bias_rw_sigma', 0.001)
        self.declare_parameter('gyro_bias_rw_sigma', 0.0001)

        # Prior noises
        self.declare_parameter('prior_pose_sigma_xyz', 1.0)
        self.declare_parameter('prior_pose_sigma_rpy_deg', 30.0)
        self.declare_parameter('prior_vel_sigma', 1.0)
        self.declare_parameter('prior_bias_sigma_accel', 0.02)
        self.declare_parameter('prior_bias_sigma_gyro', 0.002)

        # Lidar odom fusion noise (Pose constraint between consecutive LiDAR frames)
        self.declare_parameter('lidar_pose_sigma_xyz', 0.2)
        self.declare_parameter('lidar_pose_sigma_rpy_deg', 6.0)
        # Static extrinsic from IMU/body to LiDAR (Pose3), default identity
        self.declare_parameter('body_T_lidar', [0,0,0,0,0,0])  # xyz (m), rpy (deg)

        # ISAM2 tuning
        self.declare_parameter('relinearize_skip', 1)
        self.declare_parameter('relinearize_threshold', 0.1)

        # Soft velocity prior (used only when no IMU fell between lidar frames)
        self.declare_parameter('soft_vel_prior_sigma_when_no_imu', 100.0)

        # ------------------- Read parameters --------------------
        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.lidar_odom_topic = self.get_parameter('lidar_odom_topic').get_parameter_value().string_value
        self.pub_odom_topic = self.get_parameter('pub_odom_topic').get_parameter_value().string_value
        self.pub_path_topic = self.get_parameter('pub_path_topic').get_parameter_value().string_value

        g = float(self.get_parameter('gravity').value)
        accel_noise = float(self.get_parameter('accel_noise_sigma').value)
        gyro_noise = float(self.get_parameter('gyro_noise_sigma').value)
        accel_rw = float(self.get_parameter('accel_bias_rw_sigma').value)
        gyro_rw = float(self.get_parameter('gyro_bias_rw_sigma').value)

        prior_pose_sigma_xyz = float(self.get_parameter('prior_pose_sigma_xyz').value)
        prior_pose_sigma_rpy_deg = float(self.get_parameter('prior_pose_sigma_rpy_deg').value)
        prior_vel_sigma = float(self.get_parameter('prior_vel_sigma').value)
        prior_bias_sigma_acc = float(self.get_parameter('prior_bias_sigma_accel').value)
        prior_bias_sigma_gyr = float(self.get_parameter('prior_bias_sigma_gyro').value)

        lidar_pose_sigma_xyz = float(self.get_parameter('lidar_pose_sigma_xyz').value)
        lidar_pose_sigma_rpy_deg = float(self.get_parameter('lidar_pose_sigma_rpy_deg').value)
        extr = [float(x) for x in self.get_parameter('body_T_lidar').value]
        self.body_T_lidar = Pose3(Rot3.RzRyRx(math.radians(extr[3]), math.radians(extr[4]), math.radians(extr[5])), Point3(extr[0], extr[1], extr[2]))

        relinearize_skip = int(self.get_parameter('relinearize_skip').value)
        relinearize_threshold = float(self.get_parameter('relinearize_threshold').value)
        soft_vel_prior_sigma = float(self.get_parameter('soft_vel_prior_sigma_when_no_imu').value)

        # ------------------ GTSAM configuration ------------------
        params = PreintegrationParams.MakeSharedU(g)
        I3 = np.eye(3)
        params.setAccelerometerCovariance((accel_noise ** 2) * I3)
        params.setGyroscopeCovariance((gyro_noise ** 2) * I3)
        params.setIntegrationCovariance(1e-8 * I3)

        self.bias_noise_model = Diagonal.Sigmas(np.array([accel_rw, accel_rw, accel_rw, gyro_rw, gyro_rw, gyro_rw]))

        self.prior_pose_noise = Diagonal.Sigmas(np.array([
            prior_pose_sigma_xyz, prior_pose_sigma_xyz, prior_pose_sigma_xyz,
            math.radians(prior_pose_sigma_rpy_deg),
            math.radians(prior_pose_sigma_rpy_deg),
            math.radians(prior_pose_sigma_rpy_deg),
        ]))
        self.prior_vel_noise = Diagonal.Sigmas(np.array([prior_vel_sigma, prior_vel_sigma, prior_vel_sigma]))
        self.soft_vel_prior = Diagonal.Sigmas(np.array([soft_vel_prior_sigma, soft_vel_prior_sigma, soft_vel_prior_sigma]))
        self.prior_bias_noise = Diagonal.Sigmas(np.array([
            prior_bias_sigma_acc, prior_bias_sigma_acc, prior_bias_sigma_acc,
            prior_bias_sigma_gyr, prior_bias_sigma_gyr, prior_bias_sigma_gyr,
        ]))

        self.lidar_pose_noise = Diagonal.Sigmas(np.array([
            lidar_pose_sigma_xyz, lidar_pose_sigma_xyz, lidar_pose_sigma_xyz,
            math.radians(lidar_pose_sigma_rpy_deg),
            math.radians(lidar_pose_sigma_rpy_deg),
            math.radians(lidar_pose_sigma_rpy_deg),
        ]))

        # ISAM2
        isam_params = ISAM2Params()
        # Try multiple bindings styles for compatibility across GTSAM builds
        try:
            isam_params.relinearizeSkip = relinearize_skip
        except Exception:
            try:
                isam_params.setRelinearizeSkip(relinearize_skip)
            except Exception:
                self.get_logger().warn('ISAM2Params: relinearizeSkip unavailable; using default')
        try:
            isam_params.relinearizeThreshold = relinearize_threshold
        except Exception:
            try:
                isam_params.setRelinearizeThreshold(relinearize_threshold)
            except Exception:
                self.get_logger().warn('ISAM2Params: relinearizeThreshold unavailable; using default')
        self.isam = ISAM2(isam_params)

        # Graph & values for current batch update
        self.graph = NonlinearFactorGraph()
        self.initial = Values()

        # State indices
        self.frame_idx = 0
        self.last_lidar_time = None
        self.last_dt = 0.0

        # Initial states
        self.prev_state = NavState(Pose3(), np.zeros(3))
        self.prev_bias = gtsam.imuBias.ConstantBias()

        # Preintegrator
        self.pim_params = params
        self.reset_preintegrator()

        # IMU buffer (time-ordered)
        self.imu_buf = deque(maxlen=4000)
        self.mutex = threading.Lock()

        # ROS interfaces
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 200)
        self.sub_odom = self.create_subscription(Odometry, self.lidar_odom_topic, self.lidar_callback, 50)

        self.pub_odom = self.create_publisher(Odometry, self.pub_odom_topic, 10)
        self.pub_path = self.create_publisher(Path, self.pub_path_topic, 5)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        self.get_logger().info('IMU Preintegration node initialized.')

    # --------------------- Utility methods ---------------------
    def reset_preintegrator(self):
        self.pim = PreintegratedImuMeasurements(self.pim_params, self.prev_bias)

    @staticmethod
    def ros_time_to_sec(msg_time) -> float:
        return float(msg_time.sec) + float(msg_time.nanosec) * 1e-9

    @staticmethod
    def pose_from_odom(odom: Odometry) -> Pose3:
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        rot = Rot3.Quaternion(q.w, q.x, q.y, q.z)
        trans = Point3(p.x, p.y, p.z)
        return Pose3(rot, trans)

    # ----------------------- Callbacks -------------------------
    def imu_callback(self, msg: Imu):
        # Buffer IMU (we will consume during lidar callback)
        with self.mutex:
            t = self.ros_time_to_sec(msg.header.stamp)
            acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            self.imu_buf.append((t, acc, gyr))

    def lidar_callback(self, odom_msg: Odometry):
        # Process a new LiDAR odometry keyframe
        t_k = self.ros_time_to_sec(odom_msg.header.stamp)

        with self.mutex:
            # Integrate IMU measurements between last and current LiDAR time
            if self.last_lidar_time is None:
                self.last_lidar_time = t_k
            else:
                integrated_dt = self.integrate_imu(self.last_lidar_time, t_k)
                self.last_dt = integrated_dt
                self.last_lidar_time = t_k

        # Build/update factor graph
        if self.frame_idx == 0:
            self.add_first_frame_factors(odom_msg)
        else:
            self.add_sequential_factors(odom_msg)

        # Update ISAM2
        self.isam.update(self.graph, self.initial)
        result = self.isam.calculateEstimate()

        # Reset containers for next iteration
        self.graph = NonlinearFactorGraph()
        self.initial = Values()

        # Extract optimized current state
        current_pose = result.atPose3(X(self.frame_idx))
        current_vel = result.atVector(V(self.frame_idx))
        self.prev_bias = result.atConstantBias(B(self.frame_idx))
        self.prev_state = NavState(current_pose, current_vel)

        # Reset preintegrator with new bias
        self.reset_preintegrator()

        # Publish odom & path
        self.publish_outputs(odom_msg, current_pose, current_vel)

        self.frame_idx += 1

    # ------------------- Factor-graph helpers ------------------
    def add_first_frame_factors(self, odom_msg: Odometry):
        # Priors on pose, velocity, bias
        pose0 = self.pose_from_odom(odom_msg)
        vel0 = np.zeros(3)
        bias0 = gtsam.imuBias.ConstantBias()

        self.graph.add(PriorFactorPose3(X(0), pose0, self.prior_pose_noise))
        self.graph.add(PriorFactorVector(V(0), vel0, self.prior_vel_noise))
        self.graph.add(PriorFactorConstantBias(B(0), bias0, self.prior_bias_noise))

        # Initialize values
        self.initial.insert(X(0), pose0)
        self.initial.insert(V(0), vel0)
        self.initial.insert(B(0), bias0)

        # Also softly pin the first pose to LiDAR odom (same as prior here)
        # For clarity, we don't add an extra factor; the prior already reflects LiDAR.

    def add_sequential_factors(self, odom_msg: Odometry):
        k = self.frame_idx

        # 1) IMU factor between k-1 and k (only meaningful if we had IMU)
        if self.last_dt > 1e-6:
            imu_factor = ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), self.pim)
            self.graph.add(imu_factor)
        else:
            # No IMU samples between frames -> add a soft velocity prior to avoid singularity
            self.graph.add(PriorFactorVector(V(k), self.prev_state.velocity(), self.soft_vel_prior))

        # 2) Bias evolution
        self.graph.add(BetweenFactorConstantBias(B(k-1), B(k), gtsam.imuBias.ConstantBias(), self.bias_noise_model))

        # 3) LiDAR odometry as a between constraint between k-1 and k
        pose_k_lidar = self.pose_from_odom(odom_msg)
        # Transform LiDAR pose into body frame if needed: X_body = X_lidar composed with (body_T_lidar)^-1
        pose_k_body = pose_k_lidar.compose(self.body_T_lidar.inverse())
        # Use previous stored lidar/body pose
        if not hasattr(self, 'last_lidar_body_pose'):
            self.last_lidar_body_pose = pose_k_body
        delta = self.last_lidar_body_pose.between(pose_k_body)
        self.graph.add(BetweenFactorPose3(X(k-1), X(k), delta, self.lidar_pose_noise))
        self.last_lidar_body_pose = pose_k_body

        # 4) Initial guesses (predict from previous state)
        pred_state = self.pim.predict(self.prev_state, self.prev_bias)
        self.initial.insert(X(k), pred_state.pose())
        self.initial.insert(V(k), pred_state.velocity())
        self.initial.insert(B(k), self.prev_bias)

    # ---------------------- IMU integration --------------------
    def integrate_imu(self, t0: float, t1: float):
        if t1 <= t0:
            return 0.0
        # Consume IMU samples in (t0, t1]
        # We also handle a simple last-sample extrapolation if needed
        while self.imu_buf and self.imu_buf[0][0] <= t0:
            self.imu_buf.popleft()

        last_t = t0
        acc_last = None
        gyr_last = None
        total_dt = 0.0
        for idx in range(len(self.imu_buf)):
            t, acc, gyr = self.imu_buf[idx]
            if t > t1:
                break
            if t <= t0:
                continue
            if acc_last is None:
                acc_last, gyr_last = acc, gyr

            dt = max(1e-6, t - last_t)
            self.pim.integrateMeasurement(acc, gyr, dt)
            total_dt += dt
            last_t = t
            acc_last, gyr_last = acc, gyr

        # If last imu sample time < t1, integrate the last measurement to t1
        if last_t < t1 and acc_last is not None and gyr_last is not None:
            dt = max(1e-6, t1 - last_t)
            self.pim.integrateMeasurement(acc_last, gyr_last, dt)
            total_dt += dt
        return total_dt

    # ----------------------- Publishing ------------------------
    def publish_outputs(self, odom_src: Odometry, pose: Pose3, vel):
        # Publish odometry in 'map' frame
        odom_out = Odometry()
        odom_out.header.stamp = odom_src.header.stamp
        odom_out.header.frame_id = 'odom'
        odom_out.child_frame_id = odom_src.child_frame_id or 'base_link'

        t = pose.translation()
        # Handle both GTSAM Point3 and numpy array returns
        try:
            tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
        except Exception:
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

        R = pose.rotation()
        # Prefer numpy-returning API if available
        try:
            qw, qx, qy, qz = map(float, R.quaternion())  # [w,x,y,z]
        except Exception:
            q = R.toQuaternion()  # object with w(),x(),y(),z()
            qw, qx, qy, qz = float(q.w()), float(q.x()), float(q.y()), float(q.z())

        odom_out.pose.pose.position.x = tx
        odom_out.pose.pose.position.y = ty
        odom_out.pose.pose.position.z = tz
        odom_out.pose.pose.orientation.w = qw
        odom_out.pose.pose.orientation.x = qx
        odom_out.pose.pose.orientation.y = qy
        odom_out.pose.pose.orientation.z = qz

        odom_out.twist.twist.linear.x = float(vel[0])
        odom_out.twist.twist.linear.y = float(vel[1])
        odom_out.twist.twist.linear.z = float(vel[2])

        self.pub_odom.publish(odom_out)

        ps = PoseStamped()
        ps.header = odom_out.header
        ps.pose = odom_out.pose.pose
        self.path_msg.header.stamp = odom_out.header.stamp
        self.path_msg.poses.append(ps)
        self.pub_path.publish(self.path_msg)


def main():
    rclpy.init()
    node = ImuPreintegrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

