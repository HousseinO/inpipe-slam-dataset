#!/usr/bin/env python3
"""
imu_icp_posegraph.py
-----------------------------------
GTSAM 4.2 + ROS2 node that fuses:
  - IMU orientation (roll, pitch, yaw)
  - ICP translation (from /pipe/odom or similar)

Each keyframe node X(k) stores a Pose3.
  * Rotation comes directly from IMU
  * Translation increment comes from ICP
Graph factors:
  1. BetweenFactorPose3 (translation only, loose yaw)
  2. Rotation prior from IMU quaternion (tight roll/pitch/yaw)
  3. Optional small velocity regularizer (keeps graph stable)

This avoids ICP rotation drift in elbows and leverages the IMU for full 3D orientation.
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

import numpy as np
import gtsam
from gtsam import symbol


class ICPIMU_PoseGraph(Node):
    def __init__(self):
        super().__init__('icp_imu_posegraph')

        # Parameters
        self.declare_parameter('imu_topic', '/inpipe_bot/imu/data')
        self.declare_parameter('odom_topic', '/pipe/odom')
        self.declare_parameter('keyframe_dt', 0.05)
        self.declare_parameter('gravity', 9.80665)

        imu_topic = self.get_parameter('imu_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.keyframe_dt = float(self.get_parameter('keyframe_dt').value)

        # GTSAM setup
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.isam = gtsam.ISAM2()

        self.X = lambda k: symbol('x', k)

        # Noise models
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.deg2rad([2, 2, 2]), [0.05, 0.05, 0.05]]
        )
        self.rot_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.deg2rad([1, 1, 1]), [999.0, 999.0, 999.0]]
        )
        self.trans_factor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.deg2rad([4, 4, 6]), [0.03, 0.03, 0.03]]
        )

        # State
        self.k = 0
        self.last_time = None
        self.last_imu_quat = None
        self.last_odom = None
        self.odom_accum = gtsam.Pose3()
        self.last_kf_time = None

        # Publishers
        self.pub_path = self.create_publisher(Path, 'posegraph/path', 10)

        # Subscribers
        self.create_subscription(Imu, imu_topic, self.cb_imu, 200)
        self.create_subscription(Odometry, odom_topic, self.cb_odom, 20)

        self.timer = self.create_timer(1.0, self.heartbeat)
        self.initialize_prior()

        self.get_logger().info(f"PoseGraph ready (IMU={imu_topic}, ICP={odom_topic})")

    # ----------------------------------------------------------------
    def initialize_prior(self):
        X0 = gtsam.Pose3()
        self.graph.add(gtsam.PriorFactorPose3(self.X(0), X0, self.prior_pose_noise))
        self.values.insert(self.X(0), X0)
        self.isam.update(self.graph, self.values)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.k = 0

    # ----------------------------------------------------------------
    def cb_imu(self, msg: Imu):
        # store quaternion for next keyframe
        q = msg.orientation
        self.last_imu_quat = (q.w, q.x, q.y, q.z)

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_kf_time is None:
            self.last_kf_time = t
        elif (t - self.last_kf_time) >= self.keyframe_dt:
            self.add_keyframe()
            self.last_kf_time = t

    # ----------------------------------------------------------------
    def cb_odom(self, msg: Odometry):
        """Accumulate ICP relative motion (translation + small yaw) between keyframes."""
        def pose_from_msg(m):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            return gtsam.Pose3(
                gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z),
                gtsam.Point3(p.x, p.y, p.z)
            )

        Z = pose_from_msg(msg)
        if self.last_odom is None:
            self.last_odom = msg
            self.odom_accum = gtsam.Pose3()
            return

        Z_prev = pose_from_msg(self.last_odom)
        rel = Z_prev.between(Z)
        self.odom_accum = self.odom_accum.compose(rel)
        self.last_odom = msg

    # ----------------------------------------------------------------
    def add_keyframe(self):
        """Fuse accumulated ICP translation + IMU rotation into the graph."""
        if self.last_imu_quat is None:
            return
        k1 = self.k + 1

        est = self.isam.calculateEstimate()
        Xk = est.atPose3(self.X(self.k)) if est.exists(self.X(self.k)) else gtsam.Pose3()

        # --- (1) Build new pose using IMU orientation + ICP translation ---
        qw, qx, qy, qz = self.last_imu_quat
        R_imu = gtsam.Rot3.Quaternion(qw, qx, qy, qz)

        # Extract translation only from ICP accum
        t_rel = self.odom_accum.translation()
        # Robust extraction (handles both Point3 and numpy.ndarray)
        try:
            tx, ty, tz = float(t_rel.x()), float(t_rel.y()), float(t_rel.z())
        except AttributeError:
            t_arr = np.asarray(t_rel).reshape(-1)
            tx, ty, tz = float(t_arr[0]), float(t_arr[1]), float(t_arr[2])

        T_rel = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(tx, ty, tz))


        # Compose current estimate with new relative motion
        Xk1 = Xk.compose(gtsam.Pose3(R_imu, gtsam.Point3(tx, ty, tz)))

        # Insert initial guess
        self.values.insert(self.X(k1), Xk1)

        # --- (2) BetweenFactor using ICP translation (loose yaw) ---
        self.graph.add(gtsam.BetweenFactorPose3(
            self.X(self.k), self.X(k1), T_rel, self.trans_factor_noise
        ))

        # --- (3) Rotation prior from IMU (tight roll, pitch, yaw) ---
        Z_R = gtsam.Pose3(R_imu, gtsam.Point3(0, 0, 0))
        self.graph.add(gtsam.PriorFactorPose3(self.X(k1), Z_R, self.rot_prior_noise))

        # Optimize and publish
        self.optimize_and_publish()

        # Reset accumulators
        self.odom_accum = gtsam.Pose3()
        self.k = k1

    # ----------------------------------------------------------------
    def optimize_and_publish(self):
        self.isam.update(self.graph, self.values)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.publish_path()

    # ----------------------------------------------------------------
    def publish_path(self):
        est = self.isam.calculateEstimate()
        path = Path()
        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        path.header.frame_id = 'odom'

        for i in range(self.k + 1):
            if not est.exists(self.X(i)):
                continue
            Xi = est.atPose3(self.X(i))
            t = Xi.translation()
            # Robust handling for Point3 or ndarray
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

    # ----------------------------------------------------------------
    def heartbeat(self):
        self.get_logger().info(
    f"k={self.k} nodes, path subscribers={self.pub_path.get_subscription_count()}"
)
# --------------------------------------------------------------------

def main():
    rclpy.init()
    node = ICPIMU_PoseGraph()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

