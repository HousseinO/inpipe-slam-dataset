#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2  # apt/rosdep: python3-sensor-msgs or equivalent

import gtsam
from gtsam import (
    NonlinearFactorGraph, Values,
    Pose2,
    BetweenFactorPose2, PriorFactorPose2,
    noiseModel, ISAM2, ISAM2Params
)

def yaw_from_quat(q):
    # geometry_msgs/Quaternion -> yaw (ROS standard)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class IcpSonarDeltaFusionNode(Node):
    def __init__(self):
        super().__init__("icp_sonar_delta_fusion_node")

        # ----------------- Parameters -----------------
        self.declare_parameter("odom_topic", "/pipe/odom")
        # axial sonar as PointCloud2, like your /inpipe_bot/sonar/left/range
        self.declare_parameter("sonar_cloud_topic", "/inpipe_bot/sonar/left/range")
        self.declare_parameter("fixed_frame", "odom")
        self.declare_parameter("robot_frame", "base_link")

        # distance [m] at which we say "we are close to elbow → trust sonar only for x"
        self.declare_parameter("sonar_elbow_threshold", 1.0)

        odom_topic = self.get_parameter("odom_topic").value
        sonar_cloud_topic = self.get_parameter("sonar_cloud_topic").value
        self.elbow_threshold = float(self.get_parameter("sonar_elbow_threshold").value)

        # ----------------- Subscriptions -----------------
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 50)
        self.create_subscription(PointCloud2, sonar_cloud_topic, self.sonar_cloud_callback, 10)

        # ----------------- Publishers -----------------
        self.odom_pub = self.create_publisher(Odometry, "fused_odom", 10)
        self.path_pub = self.create_publisher(Path, "fused_path", 10)
        self.fused_path = Path()
        self.fused_path.header.frame_id = self.get_parameter("fixed_frame").value

        # ----------------- iSAM2 -----------------
        # iSAM2 config (API-safe)
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

        self.graph = NonlinearFactorGraph()
        self.initial = Values()

        self.is_initialized = False
        self.last_key = -1
        self.last_odom_pose = None  # Pose2

        # sonar readings (scalar d = min distance in front)
        self.latest_sonar_range = None
        self.prev_sonar_range_for_key = None

        # ----------------- Noise models -----------------
        # Prior: fairly confident on first pose
        self.prior_pose_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.02], dtype=float)  # x, y, theta
        )

        # Normal ICP factor: used when far from elbow
        self.odom_noise_normal = noiseModel.Diagonal.Sigmas(
            np.array([0.02, 0.02, 0.02], dtype=float)
        )

        # ICP factor near elbow: trust only yaw (roll/pitch/yaw in 3D -> here only theta)
        # Big noise on x,y => effectively ignore translation.
        self.odom_noise_orientation_only = noiseModel.Diagonal.Sigmas(
            np.array([50.0, 10.0, 0.02], dtype=float)  # huge x,y; good theta
        )

        # Sonar-delta factor (normal zone): soft x correction
        self.sonar_delta_noise_normal = noiseModel.Diagonal.Sigmas(
            np.array([
                0.05,  # x (5 cm)
                1.0,   # y large
                1.0    # theta large
            ], dtype=float)
        )

        # Sonar-delta factor (elbow zone): strong x constraint
        self.sonar_delta_noise_strong = noiseModel.Diagonal.Sigmas(
            np.array([
                0.001,  # x (1 cm) -> "take only x from sonar"
                1.0,   # y large
                1.0    # theta large
            ], dtype=float)
        )

        # Prior on y=0 in elbow zone: keep robot centered in pipe
        # We'll create a temporary PriorFactorPose2 with this when needed.
        self.y_zero_noise_strong = noiseModel.Diagonal.Sigmas(
            np.array([
                1.0,    # x free-ish
                0.01,   # y strongly -> 0
                1.0     # theta free-ish
            ], dtype=float)
        )

        self.get_logger().info(
            "ICP + sonar(delta-from-cloud) fusion node started.\n"
            f"  Odom topic          : {odom_topic}\n"
            f"  Sonar cloud topic   : {sonar_cloud_topic}\n"
            f"  Elbow threshold [m] : {self.elbow_threshold}\n"
            "Behavior:\n"
            "  - d > threshold: normal ICP (+ optional soft sonar).\n"
            "  - d <= threshold: x from sonar, y->0, yaw from ICP."
        )

    # ----------------- Helpers -----------------

    def pose2_from_odom(self, msg: Odometry) -> Pose2:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        return Pose2(x, y, yaw)

    def pose_key(self, i: int):
        return gtsam.symbol('x', i)

    # ----------------- Sonar from PointCloud2 -----------------

    def sonar_cloud_callback(self, msg: PointCloud2):
        """
        Compute sonar-like distance:
        d = min distance to any point with x > 0 (in sensor frame).
        """
        min_r = None

        for x, y, z in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            if x <= 0.0:
                continue  # behind sensor
            r = math.sqrt(x * x + y * y + z * z)
            if min_r is None or r < min_r:
                min_r = r

        if min_r is not None:
            self.latest_sonar_range = float(min_r)

    # ----------------- Odom callback (core logic) -----------------

    def odom_callback(self, msg: Odometry):
        current_pose_odom = self.pose2_from_odom(msg)
        stamp = msg.header.stamp

        # -------- initialization --------
        if not self.is_initialized:
            k0 = 0
            x0 = self.pose_key(k0)

            # Prior from first ICP
            self.graph.add(PriorFactorPose2(x0, current_pose_odom, self.prior_pose_noise))
            self.initial.insert(x0, current_pose_odom)

            self.isam.update(self.graph, self.initial)
            self.graph = NonlinearFactorGraph()
            self.initial = Values()

            self.last_key = k0
            self.last_odom_pose = current_pose_odom
            self.is_initialized = True

            if self.latest_sonar_range is not None:
                self.prev_sonar_range_for_key = self.latest_sonar_range

            self.get_logger().info("Initialized factor graph with first ICP pose.")
            self.publish_fused_pose(current_pose_odom, stamp)
            return

        # -------- new keyframe --------
        k_prev = self.last_key
        k_new = k_prev + 1
        x_prev = self.pose_key(k_prev)
        x_new = self.pose_key(k_new)

        delta_icp = self.last_odom_pose.between(current_pose_odom)
        sonar_now = self.latest_sonar_range
        in_elbow_zone = (
            sonar_now is not None and sonar_now <= self.elbow_threshold
        )

        # 1) Add ICP factor:
        if in_elbow_zone:
            # Near elbow: use ICP only for orientation
            odom_noise = self.odom_noise_orientation_only
        else:
            # Normal: use full ICP
            odom_noise = self.odom_noise_normal

        self.graph.add(BetweenFactorPose2(x_prev, x_new, delta_icp, odom_noise))

        # 2) Add sonar-delta factor, if we have consecutive sonar readings
        if sonar_now is not None and self.prev_sonar_range_for_key is not None:
            dx_sonar = self.prev_sonar_range_for_key - sonar_now
            delta_sonar = Pose2(dx_sonar, 0.0, 0.0)

            if in_elbow_zone:
                # Strong: "take x from sonar"
                sonar_noise = self.sonar_delta_noise_strong
            else:
                # Soft: small assist
                sonar_noise = self.sonar_delta_noise_normal

            self.graph.add(
                BetweenFactorPose2(x_prev, x_new, delta_sonar, sonar_noise)
            )

        # 3) If in elbow zone, clamp y ≈ 0 for this pose
        if in_elbow_zone:
            # We don't know exact x/theta yet, so:
            # - get a crude guess from ICP,
            # - only strongly constrain y to 0 via noise.
            guess_pose = Pose2(current_pose_odom.x(), 0.0, current_pose_odom.theta())
            self.graph.add(
                PriorFactorPose2(x_new, guess_pose, self.y_zero_noise_strong)
            )

        # 4) Initial guess for x_new
        try:
            est = self.isam.calculateEstimate()
            last_est = est.atPose2(x_prev)
            init_new = last_est.compose(delta_icp)
        except Exception:
            init_new = current_pose_odom

        if in_elbow_zone:
            # apply our rule for initial guess:
            # use sonar dx for x, keep y=0, orientation from ICP
            if sonar_now is not None and self.prev_sonar_range_for_key is not None:
                dx_sonar = self.prev_sonar_range_for_key - sonar_now
                init_new = Pose2(
                    last_est.x() + dx_sonar if 'last_est' in locals() else init_new.x(),
                    0.0,
                    init_new.theta()
                )

        self.initial.insert(x_new, init_new)

        # 5) iSAM update
        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial = Values()

        # 6) Extract fused pose
        result = self.isam.calculateEstimate()
        fused_pose = result.atPose2(x_new)

        # Apply final enforcement of your rule for publishing:
        # if close to elbow -> x from sonar-delta, y=0, yaw from optimized (ICP-dominated).
        if in_elbow_zone and sonar_now is not None and self.prev_sonar_range_for_key is not None:
            # compute sonar-based dx again relative to previous fused pose
            try:
                prev_fused = result.atPose2(x_prev)
            except Exception:
                prev_fused = fused_pose  # fallback

            dx_sonar = self.prev_sonar_range_for_key - sonar_now
            fused_pose = Pose2(
                prev_fused.x() + dx_sonar,
                0.0,                    # y = 0 in elbow zone
                fused_pose.theta()      # orientation from graph (essentially ICP)
            )

        # 7) Update stored values
        self.last_key = k_new
        self.last_odom_pose = current_pose_odom
        if sonar_now is not None:
            self.prev_sonar_range_for_key = sonar_now

        # 8) Publish
        self.publish_fused_pose(fused_pose, stamp)

    # ----------------- Publishing -----------------

    def publish_fused_pose(self, pose: Pose2, stamp):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.get_parameter("fixed_frame").value
        odom_msg.child_frame_id = self.get_parameter("robot_frame").value

        odom_msg.pose.pose.position.x = pose.x()
        odom_msg.pose.pose.position.y = pose.y()
        odom_msg.pose.pose.position.z = 0.0  # 2D assumption

        cy = math.cos(pose.theta() * 0.5)
        sy = math.sin(pose.theta() * 0.5)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = sy
        odom_msg.pose.pose.orientation.w = cy

        self.odom_pub.publish(odom_msg)

        ps = PoseStamped()
        ps.header = odom_msg.header
        ps.pose = odom_msg.pose.pose

        self.fused_path.header.stamp = stamp
        self.fused_path.poses.append(ps)
        self.path_pub.publish(self.fused_path)

def main(args=None):
    rclpy.init(args=args)
    node = IcpSonarDeltaFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

