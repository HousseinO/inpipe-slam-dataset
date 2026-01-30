#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
from collections import deque
from typing import List

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist

import sensor_msgs_py.point_cloud2 as pc2

import numpy as np

# GTSAM imports
import gtsam
from gtsam import (
    Pose3, Rot3, Point3, Values, NonlinearFactorGraph, ISAM2, ISAM2Params,
    PriorFactorPose3, PriorFactorVector, PriorFactorConstantBias,
    BetweenFactorConstantBias, ImuFactor, BetweenFactorPose3,
    PreintegrationParams, PreintegratedImuMeasurements, NavState,
)
from gtsam import noiseModel
from gtsam.symbol_shorthand import X, V, B
from gtsam.noiseModel import Diagonal


# --------- Ring ICP CustomFactor helper (point-to-plane residual) ---------
def make_ring_icp_factor(
    key1,
    key2,
    points1: np.ndarray,
    points2: np.ndarray,
    normals2: np.ndarray,
    noise_sigma: float,
) -> gtsam.CustomFactor:
    """
    Build a GTSAM CustomFactor implementing the ring ICP residual:
        r_i = n_i^T (p2_i - (R p1_i + t))
    between poses key1 and key2.

    points1, points2, normals2: (N,3) numpy arrays in their laser frames.
    """

    assert points1.shape == points2.shape == normals2.shape
    N = points1.shape[0]

    # Diagonal noise model: one scalar residual per correspondence
    model = gtsam.noiseModel.Diagonal.Sigmas(
        np.full(N, noise_sigma, dtype=float)
    )

    # Capture arrays in closure
    p1 = points1.astype(float)
    p2 = points2.astype(float)
    n2 = normals2.astype(float)

    def error_func(this: gtsam.CustomFactor,
                   values: gtsam.Values,
                   H: List[np.ndarray]) -> np.ndarray:
        """
        CustomFactor callback:
          - this: the factor itself
          - values: current Values
          - H: list of Jacobian matrices (one per key), or None
        """

        k1 = this.keys()[0]
        k2 = this.keys()[1]

        pose1: Pose3 = values.atPose3(k1)
        pose2: Pose3 = values.atPose3(k2)

        # T_21 = pose2 * pose1^-1 (transform from frame1 to frame2)
        T21 = pose2.between(pose1)
        R = T21.rotation().matrix()                         # 3x3
        t = np.array(T21.translation()).reshape(3, 1)       # 3x1

        # 3xN
        P1 = p1.T
        P2 = p2.T
        N2 = n2.T

        # Transform points1 into frame2
        RP1 = R @ P1 + t        # 3xN
        diff = P2 - RP1         # 3xN

        # Residuals r_i = n_i^T (p2_i - (R p1_i + t))
        residuals = np.sum(N2 * diff, axis=0)  # shape (N,)

        # Jacobians: placeholder zeros (you can implement analytic Jacobians later)
        if H is not None:
            H[0] = np.zeros((N, 6))
            H[1] = np.zeros((N, 6))

        return residuals

    cf = gtsam.CustomFactor(model, [key1, key2], error_func)
    return cf


class FullSlamNode(Node):
    def __init__(self):
        super().__init__('full_slam_node')

        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True
            )
        ])

        # ---------------------- Parameters ----------------------
        self.declare_parameter('imu_topic', '/inpipe_bot/imu/data')
        self.declare_parameter('pipe_odom_topic', '/pipe/odom')
        self.declare_parameter('ring_laser_topic', '/inpipe_bot/laser_profiler/points')
        self.declare_parameter('rplidar_odom_topic', '/odom/rplidar')
        self.declare_parameter('rplidar_points_topic', '/inpipe_bot/rplidar/points')

        self.declare_parameter('pub_odom_topic', '/imu_preintegration_full_slam/odom')
        self.declare_parameter('pub_path_topic', '/imu_preintegration_full_slam/path')

        # Gravity (m/s^2)
        self.declare_parameter('gravity', 9.81)

        # IMU noise params
        self.declare_parameter('accel_noise_sigma', 0.2)
        self.declare_parameter('gyro_noise_sigma', 0.02)
        self.declare_parameter('accel_bias_rw_sigma', 0.0002)
        self.declare_parameter('gyro_bias_rw_sigma', 0.00002)

        # Priors
        self.declare_parameter('prior_pose_sigma_xyz', 1.0)
        self.declare_parameter('prior_pose_sigma_rpy_deg', 30.0)
        self.declare_parameter('prior_vel_sigma', 1.0)
        self.declare_parameter('prior_bias_sigma_accel', 0.1)
        self.declare_parameter('prior_bias_sigma_gyro', 0.1)

        # LiDAR odom factor noise (used for both /pipe/odom and /odom/rplidar)
        self.declare_parameter('lidar_pose_sigma_xyz', 0.1)
        self.declare_parameter('lidar_pose_sigma_rpy_deg', 3.0)
        self.declare_parameter('icp_rot_sigma_deg_override', -1.0)
        self.declare_parameter('lidar_huber_k', 1.345)

        # body_T_lidar (IMU/body = base_link, lidar = laser_profiler_link)
        # [x, y, z, roll_deg, pitch_deg, yaw_deg]
        self.declare_parameter('body_T_lidar', [0.3, 0.06, 0.0, 0.0, 90.0, 0.0])

        # ISAM2 tuning
        self.declare_parameter('relinearize_skip', 1)
        self.declare_parameter('relinearize_threshold', 0.1)

        # Soft velocity prior
        self.declare_parameter('soft_vel_prior_sigma_when_no_imu', 100.0)

        # Calibration / stationary detection
        self.declare_parameter('calib_samples_needed', 1000)
        self.declare_parameter('stationary_ang_thresh', 0.005)
        self.declare_parameter('stationary_g_thresh', 0.08)
        self.declare_parameter('zupt_bias_blend', 0.001)

        # Elbow detection
        self.declare_parameter('elbow_distance_threshold', 1.0)  # meters

        # ------------------- Read parameters --------------------
        self.imu_topic = self.get_parameter('imu_topic').value
        self.pipe_odom_topic = self.get_parameter('pipe_odom_topic').value
        self.ring_laser_topic = self.get_parameter('ring_laser_topic').value
        self.rplidar_odom_topic = self.get_parameter('rplidar_odom_topic').value
        self.rplidar_points_topic = self.get_parameter('rplidar_points_topic').value

        self.pub_odom_topic = self.get_parameter('pub_odom_topic').value
        self.pub_path_topic = self.get_parameter('pub_path_topic').value

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
        roll_rad = math.radians(extr[3])
        pitch_rad = math.radians(extr[4])
        yaw_rad = math.radians(extr[5])
        self.body_T_lidar = Pose3(
            Rot3.RzRyRx(yaw_rad, pitch_rad, roll_rad),
            Point3(extr[0], extr[1], extr[2])
        )

        # Rot noise
        rot_sigma_deg = lidar_pose_sigma_rpy_deg if icp_rot_override <= 0 else icp_rot_override
        self.lidar_pose_noise = Diagonal.Sigmas(np.array([
            lidar_pose_sigma_xyz, lidar_pose_sigma_xyz, lidar_pose_sigma_xyz,
            math.radians(rot_sigma_deg), math.radians(rot_sigma_deg), math.radians(rot_sigma_deg),
        ]))
        self.lidar_pose_noise_robust = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(huber_k), self.lidar_pose_noise
        )

        relinearize_skip = int(self.get_parameter('relinearize_skip').value)
        relinearize_threshold = float(self.get_parameter('relinearize_threshold').value)
        soft_vel_prior_sigma = float(self.get_parameter('soft_vel_prior_sigma_when_no_imu').value)

        self.soft_vel_prior = Diagonal.Sigmas(np.array([
            soft_vel_prior_sigma,
            soft_vel_prior_sigma,
            soft_vel_prior_sigma,
        ]))

        # ------------------ GTSAM configuration ------------------
        params = PreintegrationParams.MakeSharedU(self.g)
        I3 = np.eye(3)
        params.setAccelerometerCovariance((accel_noise ** 2) * I3)
        params.setGyroscopeCovariance((gyro_noise ** 2) * I3)
        params.setIntegrationCovariance(1e-8 * I3)
        if hasattr(params, "setBiasAccCovariance"):
            params.setBiasAccCovariance((accel_rw ** 2) * I3)
        if hasattr(params, "setBiasOmegaCovariance"):
            params.setBiasOmegaCovariance((gyro_rw ** 2) * I3)

        self.pim_params = params

        # Bias noise model (for BetweenFactorConstantBias)
        self.bias_noise_model = Diagonal.Sigmas(
            np.array([accel_rw, accel_rw, accel_rw, gyro_rw, gyro_rw, gyro_rw])
        )

        # Prior noise models
        self.prior_pose_noise = Diagonal.Sigmas(np.array([
            prior_pose_sigma_xyz, prior_pose_sigma_xyz, prior_pose_sigma_xyz,
            math.radians(prior_pose_sigma_rpy_deg),
            math.radians(prior_pose_sigma_rpy_deg),
            math.radians(prior_pose_sigma_rpy_deg),
        ]))
        self.prior_vel_noise = Diagonal.Sigmas(
            np.full(3, prior_vel_sigma, dtype=float)
        )
        self.prior_bias_noise = Diagonal.Sigmas(np.array([
            prior_bias_sigma_acc, prior_bias_sigma_acc, prior_bias_sigma_acc,
            prior_bias_sigma_gyr, prior_bias_sigma_gyr, prior_bias_sigma_gyr,
        ]))

        # ISAM2 (shared params, separate instances)
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

        self.isam_params = isam_params
        self.isam_pipe = ISAM2(self.isam_params)
        self.isam_elbow = ISAM2(self.isam_params)

        # Graph & values for current batch update (per mode)
        self.graph_pipe = NonlinearFactorGraph()
        self.initial_pipe = Values()
        self.graph_elbow = NonlinearFactorGraph()
        self.initial_elbow = Values()

        # Frame indices
        self.frame_idx_pipe = 0   # used for pipe graph + ring data indexing
        self.frame_idx_elbow = 0  # used for elbow graph
        self.last_lidar_time = None
        self.last_dt = 0.0

        # Initial states (shared physical state)
        self.prev_state = NavState(Pose3(), np.zeros(3))
        self.prev_bias = gtsam.imuBias.ConstantBias()

        # Preintegration object
        self.reset_preintegrator()

        # Calibration state
        self.calib_samples_needed = int(self.get_parameter('calib_samples_needed').value)
        self.calib_count = 0
        self.acc_sum = np.zeros(3)
        self.omega_sum = np.zeros(3)
        self.calibrated = False

        # Stationary detection
        self.ang_thresh = float(self.get_parameter('stationary_ang_thresh').value)
        self.g_thresh = float(self.get_parameter('stationary_g_thresh').value)
        self.blend = float(self.get_parameter('zupt_bias_blend').value)

        # IMU buffer
        self.imu_buf = deque(maxlen=4000)
        self.mutex = threading.Lock()

        # Ring storage per pipe keyframe index
        self.ring_data = {}
        self.latest_laser_msg: PointCloud2 = None

        # Elbow / near-elbow state from 2D lidar
        self.near_elbow = False
        self.elbow_distance_threshold = float(
            self.get_parameter('elbow_distance_threshold').value
        )
        self.elbow_anchor_pose = None

        # NEW: mode machine (PIPE / ELBOW) and elbow counter
        self.mode = "PIPE"
        self.elbow_count = 0

        # ROS interfaces
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 200)

        # In-pipe odom source
        self.sub_pipe_odom = self.create_subscription(
            Odometry,
            self.pipe_odom_topic,
            self.pipe_odom_callback,
            50
        )

        # 3D ring laser profiler
        self.sub_ring_laser = self.create_subscription(
            PointCloud2,
            self.ring_laser_topic,
            self.laser_callback,
            10
        )

        # 2D lidar (RPLidar) – points + odom
        self.sub_rplidar_points = self.create_subscription(
            PointCloud2,
            self.rplidar_points_topic,
            self.rplidar_points_callback,
            10
        )
        self.sub_rplidar_odom = self.create_subscription(
            Odometry,
            self.rplidar_odom_topic,
            self.rplidar_odom_callback,
            50
        )

        self.last_cmd = Twist()
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

        # Publishers
        self.pub_odom = self.create_publisher(Odometry, self.pub_odom_topic, 10)
        self.pub_path = self.create_publisher(Path, self.pub_path_topic, 5)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        # Publisher for corrected ring cloud in odom frame
        self.corrected_cloud_pub = self.create_publisher(
            PointCloud2,
            '/inpipe_bot/laser_profiler/points_odom_opt',
            10
        )

        self.get_logger().info(
            'Full SLAM node initialized with repeating PIPE/ELBOW mode switching.'
        )

    # ---------------------- Utility ----------------------
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
        t = self.ros_time_to_sec(msg.header.stamp)
        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=float)
        gyr = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=float)

        with self.mutex:
            if not self.calibrated:
                self.acc_sum += acc
                self.omega_sum += gyr
                self.calib_count += 1

                if self.calib_count >= self.calib_samples_needed:
                    mean_acc = self.acc_sum / self.calib_count
                    mean_omega = self.omega_sum / self.calib_count

                    expected_acc = np.array([0.0, 0.0, self.g])
                    acc_bias = mean_acc - expected_acc
                    gyro_bias = mean_omega

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
                return

            self.imu_buf.append((t, acc, gyr))

    def laser_callback(self, msg: PointCloud2):
        """
        Convert incoming PointCloud2 from laser_profiler_link into Nx3 numpy array.
        Compute normals and store as ring_data[current_pipe_frame_idx].
        Also cache the raw cloud to later transform into odom frame after optimization.
        """
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        pts = np.array(pts, dtype=float)

        if pts.shape[0] < 10:
            return

        center = np.mean(pts, axis=0)
        normals = pts - center
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-6)

        k = self.frame_idx_pipe
        self.ring_data[k] = (pts, normals)
        self.latest_laser_msg = msg

        self.get_logger().debug(
            f"Stored laser ring for pipe frame {k}, {pts.shape[0]} points"
        )

    def rplidar_points_callback(self, msg: PointCloud2):
        """
        Use the 2D lidar scan to detect if something is closer than threshold
        in front or behind the lidar. This drives the PIPE <-> ELBOW mode switches.
        """
        min_r = None

        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y = float(p[0]), float(p[1])
            r = math.sqrt(x * x + y * y)
            if r <= 0.01:
                continue

            angle = math.atan2(y, x)  # -pi..pi

            # front: ±6° around +X
            front = abs(angle) < math.radians(6.0)
            # back: ±6° around -X
            back = abs(abs(angle) - math.pi) < math.radians(6.0)

            if not (front or back):
                continue

            if min_r is None or r < min_r:
                min_r = r

        if min_r is None:
            return

        prev_near = self.near_elbow
        self.near_elbow = (min_r <= self.elbow_distance_threshold)

        # ------------ PIPE -> ELBOW (entering elbow) ------------
        if self.near_elbow and not prev_near and self.mode == "PIPE":
            self.elbow_count += 1
            self.get_logger().info(
                f"[rplidar_points_callback] Entering elbow #{self.elbow_count} "
                f"(min_r={min_r:.2f} m). Anchoring elbow graph at current SLAM pose."
            )

            # Anchor elbow pose at current optimized pose
            try:
                self.elbow_anchor_pose = self.prev_state.pose()
            except Exception:
                self.elbow_anchor_pose = Pose3()

            # Reset elbow graph & iSAM for this elbow
            self.frame_idx_elbow = 0
            self.graph_elbow = NonlinearFactorGraph()
            self.initial_elbow = Values()
            self.isam_elbow = ISAM2(self.isam_params)

            if hasattr(self, 'last_lidar_body_pose_elbow'):
                del self.last_lidar_body_pose_elbow

            # Switch mode
            self.mode = "ELBOW"

        # ------------ ELBOW -> PIPE (exiting elbow) -------------
        if (not self.near_elbow) and prev_near and self.mode == "ELBOW":
            self.get_logger().info(
                f"[rplidar_points_callback] Exiting elbow #{self.elbow_count}, "
                f"switching back to PIPE mode."
            )
            # Simply switch back; pipe graph continues globally
            self.mode = "PIPE"

    def pipe_odom_callback(self, odom_msg: Odometry):
        """
        PIPE SLAM: IMU + /pipe/odom + ring ICP.

        Used whenever mode == "PIPE".
        """
        if self.mode != "PIPE":
            self.get_logger().debug(
                "[pipe_odom_callback] mode != PIPE -> ignoring /pipe/odom"
            )
            return

        t_k = self.ros_time_to_sec(odom_msg.header.stamp)

        with self.mutex:
            if self.last_lidar_time is None:
                self.last_lidar_time = t_k
            else:
                integrated_dt = self.integrate_imu(self.last_lidar_time, t_k)
                self.last_dt = integrated_dt
                self.last_lidar_time = t_k

        if self.frame_idx_pipe == 0:
            self.add_first_frame_factors_pipe(odom_msg)
        else:
            self.add_sequential_factors_pipe(odom_msg)

        # Optimize pipe graph
        self.isam_pipe.update(self.graph_pipe, self.initial_pipe)
        result = self.isam_pipe.calculateEstimate()

        # Reset graph & initial for next iteration
        self.graph_pipe = NonlinearFactorGraph()
        self.initial_pipe = Values()

        # Extract optimized current state
        current_pose = result.atPose3(X(self.frame_idx_pipe))
        current_vel = result.atVector(V(self.frame_idx_pipe))
        self.prev_bias = result.atConstantBias(B(self.frame_idx_pipe))
        self.prev_state = NavState(current_pose, current_vel)

        # Reset preintegrator with new bias
        self.reset_preintegrator()

        # Publish odom & path
        self.publish_outputs(odom_msg, current_pose, current_vel)

        # Publish corrected laser profiler cloud in odom frame
        self.publish_corrected_cloud(current_pose, odom_msg.header.stamp)

        self.frame_idx_pipe += 1

    def rplidar_odom_callback(self, odom_msg: Odometry):
        """
        ELBOW SLAM: IMU + /odom/rplidar (2D lidar odom).

        Only used when mode == "ELBOW" and elbow_anchor_pose is set.
        """
        if self.mode != "ELBOW" or self.elbow_anchor_pose is None:
            self.get_logger().debug(
                "[rplidar_odom_callback] mode != ELBOW or anchor unset "
                "-> ignoring /odom/rplidar"
            )
            return

        t_k = self.ros_time_to_sec(odom_msg.header.stamp)

        with self.mutex:
            if self.last_lidar_time is None:
                self.last_lidar_time = t_k
            else:
                integrated_dt = self.integrate_imu(self.last_lidar_time, t_k)
                self.last_dt = integrated_dt
                self.last_lidar_time = t_k

        if self.frame_idx_elbow == 0:
            self.add_first_frame_factors_elbow(odom_msg)
        else:
            self.add_sequential_factors_elbow(odom_msg)

        # Optimize elbow graph
        self.isam_elbow.update(self.graph_elbow, self.initial_elbow)
        result = self.isam_elbow.calculateEstimate()

        # Reset graph & initial for next iteration
        self.graph_elbow = NonlinearFactorGraph()
        self.initial_elbow = Values()

        current_pose = result.atPose3(X(self.frame_idx_elbow))
        current_vel = result.atVector(V(self.frame_idx_elbow))
        self.prev_bias = result.atConstantBias(B(self.frame_idx_elbow))
        self.prev_state = NavState(current_pose, current_vel)

        self.reset_preintegrator()

        # Publish odom & path
        self.publish_outputs(odom_msg, current_pose, current_vel)

        # Optionally still publish ring cloud in odom if available
        self.publish_corrected_cloud(current_pose, odom_msg.header.stamp)

        self.frame_idx_elbow += 1

    # ------------------- Factor-graph helpers ------------------
    def add_first_frame_factors_pipe(self, odom_msg: Odometry):
        pose0 = self.pose_from_odom(odom_msg)
        vel0 = np.zeros(3)
        bias0 = self.prev_bias

        self.graph_pipe.add(PriorFactorPose3(X(0), pose0, self.prior_pose_noise))
        self.graph_pipe.add(PriorFactorVector(V(0), vel0, self.prior_vel_noise))
        self.graph_pipe.add(PriorFactorConstantBias(B(0), bias0, self.prior_bias_noise))

        self.initial_pipe.insert(X(0), pose0)
        self.initial_pipe.insert(V(0), vel0)
        self.initial_pipe.insert(B(0), bias0)

    def add_sequential_factors_pipe(self, odom_msg: Odometry):
        k = self.frame_idx_pipe

        # 1) IMU factor
        if self.last_dt > 1e-6:
            imu_factor = ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), self.pim)
            self.graph_pipe.add(imu_factor)
        else:
            self.graph_pipe.add(
                PriorFactorVector(V(k), self.prev_state.velocity(), self.soft_vel_prior)
            )

        # 2) Bias evolution
        self.graph_pipe.add(BetweenFactorConstantBias(
            B(k-1), B(k), gtsam.imuBias.ConstantBias(), self.bias_noise_model
        ))

        # 3) Pipe LiDAR odometry factor (from /pipe/odom) as relative constraint
        pose_k_body = self.pose_from_odom(odom_msg)
        if not hasattr(self, 'last_lidar_body_pose_pipe'):
            self.last_lidar_body_pose_pipe = pose_k_body
        delta = self.last_lidar_body_pose_pipe.between(pose_k_body)
        self.graph_pipe.add(BetweenFactorPose3(
            X(k-1), X(k), delta, self.lidar_pose_noise_robust
        ))
        self.last_lidar_body_pose_pipe = pose_k_body

        # 4) Ring ICP factor (if ring data available for k-1 and k)
        if (k-1 in self.ring_data) and (k in self.ring_data):
            pts1, n1 = self.ring_data[k-1]
            pts2, n2 = self.ring_data[k]

            N1 = pts1.shape[0]
            N2 = pts2.shape[0]
            N = min(N1, N2)
            if N > 0:
                points1 = pts1[:N, :]
                points2 = pts2[:N, :]
                normals2 = n2[:N, :]

                ring_sigma = 0.01  # 1 cm
                ring_factor = make_ring_icp_factor(
                    X(k-1), X(k),
                    points1, points2, normals2,
                    ring_sigma
                )
                self.graph_pipe.add(ring_factor)
                self.get_logger().info(
                    f"Added ring ICP factor between pipe frames {k-1} and {k} with {N} points"
                )
        else:
            self.get_logger().debug(
                f"No ring data for pipe frame pair ({k-1}, {k}), skipping ring factor."
            )

        # 5) Initial guess from IMU prediction
        pred_state = self.pim.predict(self.prev_state, self.prev_bias)
        self.initial_pipe.insert(X(k), pred_state.pose())
        self.initial_pipe.insert(V(k), pred_state.velocity())
        self.initial_pipe.insert(B(k), self.prev_bias)

    def add_first_frame_factors_elbow(self, odom_msg: Odometry):
        """
        First node of elbow graph.

        Prior pose is elbow_anchor_pose (optimized pose when entering elbow).
        """
        pose0 = self.elbow_anchor_pose if self.elbow_anchor_pose is not None \
            else self.pose_from_odom(odom_msg)

        vel0 = np.zeros(3)
        bias0 = self.prev_bias

        self.graph_elbow.add(PriorFactorPose3(X(0), pose0, self.prior_pose_noise))
        self.graph_elbow.add(PriorFactorVector(V(0), vel0, self.prior_vel_noise))
        self.graph_elbow.add(PriorFactorConstantBias(B(0), bias0, self.prior_bias_noise))

        self.initial_elbow.insert(X(0), pose0)
        self.initial_elbow.insert(V(0), vel0)
        self.initial_elbow.insert(B(0), bias0)

        # Initialize last lidar pose for elbow deltas
        pose_k_body = self.pose_from_odom(odom_msg)
        self.last_lidar_body_pose_elbow = pose_k_body

    def add_sequential_factors_elbow(self, odom_msg: Odometry):
        k = self.frame_idx_elbow

        # 1) IMU factor
        if self.last_dt > 1e-6:
            imu_factor = ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), self.pim)
            self.graph_elbow.add(imu_factor)
        else:
            self.graph_elbow.add(
                PriorFactorVector(V(k), self.prev_state.velocity(), self.soft_vel_prior)
            )

        # 2) Bias evolution
        self.graph_elbow.add(BetweenFactorConstantBias(
            B(k-1), B(k), gtsam.imuBias.ConstantBias(), self.bias_noise_model
        ))

        # 3) 2D lidar odometry factor (from /odom/rplidar) as relative constraint
        pose_k_body = self.pose_from_odom(odom_msg)
        if not hasattr(self, 'last_lidar_body_pose_elbow'):
            self.last_lidar_body_pose_elbow = pose_k_body
        delta = self.last_lidar_body_pose_elbow.between(pose_k_body)
        self.graph_elbow.add(BetweenFactorPose3(
            X(k-1), X(k), delta, self.lidar_pose_noise_robust
        ))
        self.last_lidar_body_pose_elbow = pose_k_body

        # 4) Initial guess from IMU prediction
        pred_state = self.pim.predict(self.prev_state, self.prev_bias)
        self.initial_elbow.insert(X(k), pred_state.pose())
        self.initial_elbow.insert(V(k), pred_state.velocity())
        self.initial_elbow.insert(B(k), self.prev_bias)

    # ---------------------- IMU integration --------------------
    def integrate_imu(self, t0: float, t1: float):
        if t1 <= t0 or not self.calibrated:
            return 0.0

        # Drop measurements older than t0
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

            acc_norm = np.linalg.norm(acc)
            is_stationary = (
                np.linalg.norm(gyr) < self.ang_thresh
                and abs(acc_norm - self.g) < self.g_thresh
            )
            if is_stationary and not self._moving_cmd():
                new_acc_bias = (1.0 - self.blend) * self.prev_bias.accelerometer() + \
                               self.blend * (acc - np.array([0.0, 0.0, self.g]))
                new_gyro_bias = (1.0 - self.blend) * self.prev_bias.gyroscope() + \
                                self.blend * gyr
                self.prev_bias = gtsam.imuBias.ConstantBias(new_acc_bias, new_gyro_bias)

            self.pim.integrateMeasurement(acc, gyr, dt)

            total_dt += dt
            last_t = t
            last_acc, last_gyr = acc, gyr

        if last_t < t1 and last_acc is not None and last_gyr is not None:
            dt = max(1e-6, t1 - last_t)
            self.pim.integrateMeasurement(last_acc, last_gyr, dt)
            total_dt += dt

        return total_dt

    # ----------------------- Publishing ------------------------
    def publish_outputs(self, odom_src: Odometry, pose: Pose3, vel):
        # Publish odometry in 'odom' frame
        odom_out = Odometry()
        odom_out.header.stamp = odom_src.header.stamp
        odom_out.header.frame_id = 'odom'
        odom_out.child_frame_id = odom_src.child_frame_id or 'base_link'

        # --- robust translation extraction ---
        t = pose.translation()
        try:
            tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
        except Exception:
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

        # --- robust rotation extraction ---
        R = pose.rotation()
        try:
            qw, qx, qy, qz = map(float, R.quaternion())
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

    def publish_corrected_cloud(self, pose: Pose3, stamp):
        """
        Transform the latest laser profiler cloud from laser_profiler_link into odom frame
        using the optimized pose and the known base_link -> laser_profiler_link extrinsic.
        """
        if self.latest_laser_msg is None:
            return

        msg = self.latest_laser_msg

        # Convert cloud -> Nx3
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if not pts:
            return

        P_laser = np.array(pts, dtype=float).T  # 3 x N

        # T_odom_laser = T_odom_base * T_base_laser
        T_odom_laser = pose.compose(self.body_T_lidar)
        Rm = T_odom_laser.rotation().matrix()
        tt = T_odom_laser.translation()

        # --- robust handling whether tt is Point3 or numpy array ---
        try:
            tx, ty, tz = float(tt.x()), float(tt.y()), float(tt.z())
        except Exception:
            tx, ty, tz = float(tt[0]), float(tt[1]), float(tt[2])

        t_vec = np.array([[tx], [ty], [tz]])  # 3 x 1

        # Transform points: p_odom = R * p_laser + t
        P_odom = Rm @ P_laser + t_vec  # 3 x N
        points_odom_list = P_odom.T.tolist()  # N x 3

        # Build output cloud in odom frame
        header = msg.header
        header.stamp = stamp
        header.frame_id = 'odom'

        cloud_odom = pc2.create_cloud_xyz32(header, points_odom_list)
        self.corrected_cloud_pub.publish(cloud_odom)


def main():
    rclpy.init()
    node = FullSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

