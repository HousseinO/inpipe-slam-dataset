#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Preintegration node inspired by LIO-SAM's imuPreintegration.cpp, rewritten in Python for ROS 2 Humble with GTSAM 4.2.

This version’s preintegration path matches imu_topic_example.py:
 - initial bias calibration from the first N IMU samples
 - integrate using that bias
 - simple stationary detection with slow bias adaptation

LiDAR factor-graph plumbing remains unchanged.
"""

import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist

import gtsam
from gtsam import Pose3, Point3, Rot3, noiseModel
import numpy as np
from gtsam.symbol_shorthand import X


import numpy as np

# GTSAM imports (tested with 4.2.x)
import gtsam
from gtsam import (
    Pose3, Rot3, Point3, Values, NonlinearFactorGraph, ISAM2, ISAM2Params,
    PriorFactorPose3, PriorFactorVector, PriorFactorConstantBias, BetweenFactorConstantBias, ImuFactor, BetweenFactorPose3,
    PreintegrationParams, PreintegratedImuMeasurements, NavState,
)
from gtsam import noiseModel
from gtsam.symbol_shorthand import X, V, B
from gtsam.noiseModel import Diagonal


class RingICPFactor(gtsam.CustomFactor):
    """
    GTSAM factor for ring-to-ring alignment between two poses X(k1), X(k2).

    Error vector e has length N (one scalar residual per correspondence):
        e_i = n_i^T (p2_i - (R * p1_i + t))
    where:
        p1_i : point in frame 1 (laser frame at k-1)
        p2_i : corresponding point in frame 2 (laser frame at k)
        n_i  : normal at p2_i in frame 2
        T_21 = Pose2 * Pose1^-1
    """

    def __init__(self, key1, key2, points1, points2, normals2, noise_sigma):
        """
        points1, points2, normals2 : numpy arrays of shape (N, 3)
        noise_sigma               : scalar std dev for each residual (m)
        """
        assert points1.shape == points2.shape == normals2.shape
        self.N = points1.shape[0]

        # Store data
        self.points1 = points1  # frame k-1
        self.points2 = points2  # frame k
        self.normals2 = normals2

        # Diagonal noise for N residuals
        model = noiseModel.Diagonal.Sigmas(
            np.full(self.N, noise_sigma, dtype=float)
        )

        # dimension = N, keys = [X(k-1), X(k)]
        super().__init__(model, [key1, key2], self._error_function)

    def _error_function(self, this, values, keys):
        """
        GTSAM will call this to get the residual vector.
        keys = [key1, key2]
        """
        k1 = keys[0]
        k2 = keys[1]

        pose1 = values.atPose3(k1)
        pose2 = values.atPose3(k2)

        # T_21 = Pose2 * Pose1^-1
        T21 = pose2.between(pose1)

        R = T21.rotation().matrix()
        t = np.array(T21.translation(), dtype=float).reshape(3)

        # For each correspondence:
        #   r_i = n_i^T (p2_i - (R p1_i + t))
        p1 = self.points1.T   # 3xN
        p2 = self.points2.T   # 3xN
        n2 = self.normals2.T  # 3xN

        Rp1_plus_t = (R @ p1) + t.reshape(3, 1)
        diff = p2 - Rp1_plus_t
        residuals = np.sum(n2 * diff, axis=0)  # shape (N,)

        return residuals




class ImuPreintegrationNode(Node):
    def __init__(self):
        super().__init__('imu_preintegration_node')
        
        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True
            )
        ])

        # ---------------------- Parameters ----------------------
        self.declare_parameter('imu_topic', '/inpipe_bot/imu/data')
        self.declare_parameter('lidar_odom_topic', '/pipe/odom')
        self.declare_parameter('pub_odom_topic', '/imu_preintegration/odom')
        self.declare_parameter('pub_path_topic', '/imu_preintegration/path')

        # Gravity (m/s^2) and direction (down along Z in ENU => -9.81 on Z)
        self.declare_parameter('gravity', 9.81)

        # Accel/Gyro continuous-time noise params (per sqrt(Hz))
        self.declare_parameter('accel_noise_sigma', 0.2)   # e.g. 0.2 m/s^2/sqrt(Hz)
        self.declare_parameter('gyro_noise_sigma', 0.02)   # e.g. 0.02 rad/s/sqrt(Hz)

        # Accel/Gyro bias random walk (drift) (per sqrt(Hz))
        self.declare_parameter('accel_bias_rw_sigma', 0.0002)
        self.declare_parameter('gyro_bias_rw_sigma', 0.00002)

        # Prior noises
        self.declare_parameter('prior_pose_sigma_xyz', 1.0)
        self.declare_parameter('prior_pose_sigma_rpy_deg', 30.0)
        self.declare_parameter('prior_vel_sigma', 1.0)
        self.declare_parameter('prior_bias_sigma_accel', 0.1)
        self.declare_parameter('prior_bias_sigma_gyro', 0.1)

        # Lidar odom fusion noise (Pose constraint between consecutive LiDAR frames)
        self.declare_parameter('lidar_pose_sigma_xyz', 0.1)
        self.declare_parameter('lidar_pose_sigma_rpy_deg', 3.0)
        # Downweight ICP yaw if degeneracy suspected
        self.declare_parameter('icp_rot_sigma_deg_override', -1.0)  # if >0, use this rot sigma instead
        self.declare_parameter('lidar_huber_k', 1.345)  # robust loss on ICP
        # Static extrinsic from IMU/body to LiDAR (Pose3), default identity
        self.declare_parameter('body_T_lidar', [0,0,0,0,0,0])  # xyz (m), rpy (deg)

        # ISAM2 tuning
        self.declare_parameter('relinearize_skip', 1)
        self.declare_parameter('relinearize_threshold', 0.1)

        # Soft velocity prior (used only when no IMU fell between lidar frames)
        self.declare_parameter('soft_vel_prior_sigma_when_no_imu', 100.0)

        # --------------- Extra params to mirror example ---------------
        # For the small stationary/zupt logic (same as example node)
        self.declare_parameter('calib_samples_needed', 1000)   # ~ like the example
        self.declare_parameter('stationary_ang_thresh', 0.005) # rad/s
        self.declare_parameter('stationary_g_thresh', 0.08)    # |‖acc‖-g|
        self.declare_parameter('zupt_bias_blend', 0.001)       # slow bias adaptation
        # (We also subscribe to /cmd_vel to gate ZUPT like the example.)
        
        # ------------------- Read parameters --------------------
        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.lidar_odom_topic = self.get_parameter('lidar_odom_topic').get_parameter_value().string_value
        self.pub_odom_topic = self.get_parameter('pub_odom_topic').get_parameter_value().string_value
        self.pub_path_topic = self.get_parameter('pub_path_topic').get_parameter_value().string_value
        
        self.ring_data = {}

        self.g = float(self.get_parameter('gravity').value)
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
        icp_rot_override = float(self.get_parameter('icp_rot_sigma_deg_override').value)
        huber_k = float(self.get_parameter('lidar_huber_k').value)
        extr = [float(x) for x in self.get_parameter('body_T_lidar').value]
        self.body_T_lidar = Pose3(Rot3.RzRyRx(math.radians(extr[3]), math.radians(extr[4]), math.radians(extr[5])), Point3(extr[0], extr[1], extr[2]))
        # Optionally inflate ICP rot sigma (helps in pipes)
        rot_sigma_deg = lidar_pose_sigma_rpy_deg if icp_rot_override <= 0 else icp_rot_override
        self.lidar_pose_noise = Diagonal.Sigmas(np.array([
            lidar_pose_sigma_xyz, lidar_pose_sigma_xyz, lidar_pose_sigma_xyz,
            math.radians(rot_sigma_deg), math.radians(rot_sigma_deg), math.radians(rot_sigma_deg),
        ]))
        # Wrap with robust loss
        self.lidar_pose_noise_robust = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(huber_k), self.lidar_pose_noise
        )

        relinearize_skip = int(self.get_parameter('relinearize_skip').value)
        relinearize_threshold = float(self.get_parameter('relinearize_threshold').value)
        soft_vel_prior_sigma = float(self.get_parameter('soft_vel_prior_sigma_when_no_imu').value)

        # ------------------ GTSAM configuration ------------------
        params = PreintegrationParams.MakeSharedU(self.g)
        I3 = np.eye(3)
        params.setAccelerometerCovariance((accel_noise ** 2) * I3)
        params.setGyroscopeCovariance((gyro_noise ** 2) * I3)
        params.setIntegrationCovariance(1e-8 * I3)
        # Match imu_topic_example.py: try to set bias covariances if available
        if hasattr(params, "setBiasAccCovariance"):
            params.setBiasAccCovariance((accel_rw ** 2) * I3)
        if hasattr(params, "setBiasOmegaCovariance"):
            params.setBiasOmegaCovariance((gyro_rw ** 2) * I3)

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

        # ISAM2
        isam_params = ISAM2Params()
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

        # ------- Preintegration (now matches example node) -------
        self.pim_params = params
        self.reset_preintegrator()

        # Calibration state (exactly like the example)
        self.calib_samples_needed = int(self.get_parameter('calib_samples_needed').value)
        self.calib_count = 0
        self.acc_sum = np.zeros(3)
        self.omega_sum = np.zeros(3)
        self.calibrated = False

        # Stationary detection thresholds (like the example)
        self.ang_thresh = float(self.get_parameter('stationary_ang_thresh').value)
        self.g_thresh = float(self.get_parameter('stationary_g_thresh').value)
        self.blend = float(self.get_parameter('zupt_bias_blend').value)

        # IMU buffer (time-ordered)
        self.imu_buf = deque(maxlen=4000)
        self.mutex = threading.Lock()

        # ROS interfaces
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 200)
        self.sub_odom = self.create_subscription(Odometry, self.lidar_odom_topic, self.lidar_callback, 50)
        # Like the example, listen to cmd_vel to avoid ZUPT when moving intentionally
        self.last_cmd = Twist()
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

        self.pub_odom = self.create_publisher(Odometry, self.pub_odom_topic, 10)
        self.pub_path = self.create_publisher(Path, self.pub_path_topic, 5)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        self.get_logger().info('IMU Preintegration node initialized (preintegration logic mirrors imu_topic_example.py).')
        
        
    def set_ring_data_for_frame(self, k, points_k, normals_k):
        """
        points_k, normals_k: numpy arrays (N,3) in laser frame at keyframe k.
        Call this from your ring-processing callback.
        """
        self.ring_data[k] = (points_k, normals_k)

    # --------------------- Utility methods ---------------------
    def reset_preintegrator(self):
        self.pim = PreintegratedImuMeasurements(self.pim_params, self.prev_bias)

    def _cmd_cb(self, msg: Twist):
        self.last_cmd = msg

    def _moving_cmd(self) -> bool:
        if self.last_cmd is None:
            return False
        return (abs(self.last_cmd.linear.x) > 0.02 or
                abs(self.last_cmd.linear.y) > 0.02 or
                abs(self.last_cmd.angular.z) > 0.02)

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
        # Calibrate first, exactly like the example: accumulate N samples, estimate biases, then enable integration
        t = self.ros_time_to_sec(msg.header.stamp)
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=float)
        gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float)

        with self.mutex:
            if not self.calibrated:
                self.acc_sum += acc
                self.omega_sum += gyr
                self.calib_count += 1

                if self.calib_count >= self.calib_samples_needed:
                    mean_acc = self.acc_sum / self.calib_count
                    mean_omega = self.omega_sum / self.calib_count
                    # Same convention as example node: expect +Z ~ +g when level
                    expected_acc = np.array([0.0, 0.0, self.g])
                    acc_bias = mean_acc - expected_acc
                    gyro_bias = mean_omega

                    # Set node bias & rebuild preintegrator with this bias (like example)
                    self.prev_bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)
                    self.reset_preintegrator()
                    self.calibrated = True
                    self.get_logger().info(
                        f"IMU bias calibration complete.\n"
                        f"  mean_acc  = {mean_acc}\n"
                        f"  mean_gyro = {mean_omega}\n"
                        f"  acc_bias  = {acc_bias}\n"
                        f"  gyro_bias = {gyro_bias}"
                    )
                # Do not buffer samples for preintegration until calibrated
                return

            # After calibration, buffer for between-frames preintegration
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
        bias0 = self.prev_bias  # use calibrated bias if ready

        self.graph.add(PriorFactorPose3(X(0), pose0, self.prior_pose_noise))
        self.graph.add(PriorFactorVector(V(0), vel0, self.prior_vel_noise))
        self.graph.add(PriorFactorConstantBias(B(0), bias0, self.prior_bias_noise))

        # Initialize values
        self.initial.insert(X(0), pose0)
        self.initial.insert(V(0), vel0)
        self.initial.insert(B(0), bias0)

    def add_sequential_factors(self, odom_msg: Odometry):
        k = self.frame_idx

        # 1) IMU factor
        if self.last_dt > 1e-6:
            imu_factor = ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), self.pim)
            self.graph.add(imu_factor)
        else:
            self.graph.add(PriorFactorVector(V(k), self.prev_state.velocity(), self.soft_vel_prior))

        # 2) Bias evolution
        self.graph.add(BetweenFactorConstantBias(B(k-1), B(k), gtsam.imuBias.ConstantBias(), self.bias_noise_model))

        # 3) LiDAR odometry factor (as before)
        pose_k_lidar = self.pose_from_odom(odom_msg)
        pose_k_body = pose_k_lidar.compose(self.body_T_lidar.inverse())
        if not hasattr(self, 'last_lidar_body_pose'):
            self.last_lidar_body_pose = pose_k_body
        delta = self.last_lidar_body_pose.between(pose_k_body)
        self.graph.add(BetweenFactorPose3(X(k-1), X(k), delta, self.lidar_pose_noise_robust))
        self.last_lidar_body_pose = pose_k_body

        # 4) NEW: ring ICP factor (if ring data available for k-1 and k)
        if (k-1 in self.ring_data) and (k in self.ring_data):
            pts1, n1 = self.ring_data[k-1]   # ring at previous frame
            pts2, n2 = self.ring_data[k]     # ring at current frame

            # For point-to-plane, we need correspondences and normals of ring2.
            # Here, assume pts1 & pts2 already correspond 1:1 (after ICP outside GTSAM),
            # and n2 are normals at pts2.
            # If pts1 and pts2 have N points:
            N1 = pts1.shape[0]
            N2 = pts2.shape[0]
            N = min(N1, N2)
            if N > 0:
                points1 = pts1[:N, :]
                points2 = pts2[:N, :]
                normals2 = n2[:N, :]

                ring_sigma = 0.01  # 1 cm std dev, tune as needed
                ring_factor = RingICPFactor(
                    X(k-1), X(k),
                    points1, points2, normals2,
                    ring_sigma
                )
                self.graph.add(ring_factor)
        else:
            self.get_logger().debug(
                f"No ring data for frame pair ({k-1}, {k}), skipping ring factor."
            )

        # 5) Initial guesses (predict from previous state)
        pred_state = self.pim.predict(self.prev_state, self.prev_bias)
        self.initial.insert(X(k), pred_state.pose())
        self.initial.insert(V(k), pred_state.velocity())
        self.initial.insert(B(k), self.prev_bias)


    # ---------------------- IMU integration --------------------
    def integrate_imu(self, t0: float, t1: float):
        """
        Integrate IMU between (t0, t1] exactly like imu_topic_example.py:
          - skip until calibrated (handled in imu_callback)
          - integrate using current bias reference (self.pim was created with self.prev_bias)
          - perform simple stationary detection and slow bias adaptation
        Note: we do NOT reset the integrator mid-window; adaptation updates self.prev_bias
              and takes full effect after the next reset (which happens each LiDAR keyframe).
        """
        if t1 <= t0 or not self.calibrated:
            return 0.0

        # Drop anything <= t0
        while self.imu_buf and self.imu_buf[0][0] <= t0:
            self.imu_buf.popleft()

        last_t = t0
        last_acc = None
        last_gyr = None
        total_dt = 0.0

        for idx in range(len(self.imu_buf)):
            t, acc, gyr = self.imu_buf[idx]
            if t > t1:
                break
            if t <= t0:
                continue

            dt = max(1e-6, t - last_t)

            # --------- Stationary detection like the example ---------
            # Use |‖acc‖ - g| and angular rate magnitude
            acc_norm = np.linalg.norm(acc)
            is_stationary = (np.linalg.norm(gyr) < self.ang_thresh) and (abs(acc_norm - self.g) < self.g_thresh)
            if is_stationary and not self._moving_cmd():
                # Very slow bias adaptation in-place (like example)
                new_acc_bias  = (1.0 - self.blend) * self.prev_bias.accelerometer() + self.blend * (acc - np.array([0.0, 0.0, self.g]))
                new_gyro_bias = (1.0 - self.blend) * self.prev_bias.gyroscope()     + self.blend * gyr
                self.prev_bias = gtsam.imuBias.ConstantBias(new_acc_bias, new_gyro_bias)
                # We keep integrating; bias change will be fully incorporated after the next reset_preintegrator()

            # Integrate sample
            self.pim.integrateMeasurement(acc, gyr, dt)

            total_dt += dt
            last_t = t
            last_acc, last_gyr = acc, gyr

        # If last sample time < t1, extend the last measurement to t1
        if last_t < t1 and last_acc is not None and last_gyr is not None:
            dt = max(1e-6, t1 - last_t)
            self.pim.integrateMeasurement(last_acc, last_gyr, dt)
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
        try:
            tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
        except Exception:
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

        R = pose.rotation()
        try:
            qw, qx, qy, qz = map(float, R.quaternion())  # [w,x,y,z]
        except Exception:
            q = R.toQuaternion()
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

