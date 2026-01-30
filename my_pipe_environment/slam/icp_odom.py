#!/usr/bin/env python3
# pipe_odom_icp_tf_elbow.py
# Improved ICP odometry for in-pipe navigation with elbows.
# Features: motion prior, rolling submap, elbow detection, multiscale ICP,
#           radial normal filtering, quality gating.

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import open3d as o3d


# ---------- SE3 / TF helpers ----------
def tf_to_matrix(tf):
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s
    return float(x), float(y), float(z), float(w)


def odom_msg_from_T(T, frame_id, child_frame_id, stamp):
    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = frame_id
    odom.child_frame_id = child_frame_id
    R = T[:3, :3]
    t = T[:3, 3]
    qx, qy, qz, qw = quat_from_R(R)
    odom.pose.pose.position.x = float(t[0])
    odom.pose.pose.position.y = float(t[1])
    odom.pose.pose.position.z = float(t[2])
    odom.pose.pose.orientation.x = qx
    odom.pose.pose.orientation.y = qy
    odom.pose.pose.orientation.z = qz
    odom.pose.pose.orientation.w = qw
    return odom


def tf_from_T(T, frame_id, child_frame_id, stamp):
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id
    R = T[:3, :3]
    t = T[:3, 3]
    qx, qy, qz, qw = quat_from_R(R)
    tf.transform.translation.x = float(t[0])
    tf.transform.translation.y = float(t[1])
    tf.transform.translation.z = float(t[2])
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


# ---------- ICP Odometry Node ----------
class PipeOdomICPElbow(Node):
    def __init__(self):
        super().__init__('pipe_odom_icp_elbow')

        # Parameters
        self.declare_parameter('cloud_topic', '/inpipe_bot/l515/l515_depth/points')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('voxel', 0.02)
        self.declare_parameter('icp_max_corr_dist', 0.06)
        self.declare_parameter('icp_max_iter', 40)
        self.declare_parameter('min_fitness', 0.25)

        topic = self.get_parameter('cloud_topic').value
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, 1)

        # Publishers
        self.pub_odom = self.create_publisher(Odometry, 'pipe/odom', 10)
        self.pub_path = self.create_publisher(Path, 'pipe/path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.get_parameter('odom_frame').value

        # TF
        self.tfbuf = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tflist = tf2_ros.TransformListener(self.tfbuf, self)
        self.tfb = tf2_ros.TransformBroadcaster(self)

        # State
        self.prev_pcd = None
        self.T_odom_base = np.eye(4)
        self.T_last_inc = np.eye(4)
        self.prev_axis = None

        # Submap
        self.submap = None
        self.submap_frames = 0
        self.max_submap_frames = 10

        self.get_logger().info(f"ICP odom (elbow-aware): subscribing {topic}")

    # ---------- Helpers ----------
    def to_o3d(self, pts, voxel):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if voxel > 0:
            pcd = pcd.voxel_down_sample(voxel)
        if np.asarray(pcd.points).shape[0] >= 50:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=50))
        return pcd

    def _principal_axis(self, pts):
        ctr = pts.mean(axis=0)
        U, S, Vt = np.linalg.svd(pts - ctr, full_matrices=False)
        return Vt[0] / (np.linalg.norm(Vt[0]) + 1e-9)

    def _angle_deg(self, a, b):
        c = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    def _merge_into_submap(self, pcd_world, voxel):
        if self.submap is None:
            self.submap = pcd_world
            self.submap_frames = 1
        else:
            self.submap += pcd_world
            self.submap = self.submap.voxel_down_sample(voxel)
            self.submap_frames = min(self.submap_frames + 1, self.max_submap_frames)

    def icp_multiscale(self, src, dst, T_init, max_corr, max_iter):
        stages = [
            (max_corr * 1.5, 20),
            (max_corr, max_iter)
        ]
        T = T_init.copy()
        last = None
        for mc, it in stages:
            reg = o3d.pipelines.registration.registration_icp(
                src, dst, mc, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it)
            )
            T = np.asarray(reg.transformation)
            last = reg
        return last

    # ---------- Main callback ----------
    def cb(self, msg: PointCloud2):
        try:
            pts_src = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception as e:
            self.get_logger().warn(f"read_points_numpy failed: {e}")
            return
        if pts_src.size == 0:
            return
        pts_src = pts_src[np.isfinite(pts_src).all(axis=1)]
        if pts_src.shape[0] < 500:
            return

        # Transform to target_frame
        source = msg.header.frame_id
        target = self.get_parameter('target_frame').value or source
        if target != source:
            try:
                tf = self.tfbuf.lookup_transform(target, source, msg.header.stamp, timeout=Duration(seconds=0.2))
                T = tf_to_matrix(tf)
                pts = (T @ np.c_[pts_src.astype(np.float64), np.ones((pts_src.shape[0], 1))].T).T[:, :3]
            except Exception as e:
                self.get_logger().warn(f"TF {target} <- {source} failed: {e}")
                return
        else:
            pts = pts_src.astype(np.float64)

        voxel = float(self.get_parameter('voxel').value)
        pcd = self.to_o3d(pts, voxel)
        if np.asarray(pcd.points).shape[0] < 300:
            return

        # Detect elbow (principal direction change)
        axis_now = self._principal_axis(np.asarray(pcd.points))
        elbow = False
        if self.prev_axis is not None:
            ang = self._angle_deg(axis_now, self.prev_axis)
            elbow = ang > 15.0
        self.prev_axis = axis_now

        if elbow:
            self.get_logger().warn("Elbow detected -> reset submap and prior")
            self.submap = None
            self.submap_frames = 0
            self.T_last_inc = np.eye(4)

        # Initialize
        if self.prev_pcd is None:
            self.prev_pcd = pcd
            self.publish_pose(msg.header.stamp)
            return

        # Use prior
        T_init = self.T_last_inc

        # Choose target (submap or previous)
        target_pcd = self.submap if (self.submap is not None and len(self.submap.points) > 0) else self.prev_pcd

        # Filter normals (keep radial)
        if len(pcd.normals):
            normals = np.asarray(pcd.normals)
            if self.prev_axis is not None:
                axis = self.prev_axis
                cosang = np.abs(normals @ axis)
                keep = cosang < 0.9
                idx = np.where(keep)[0]
                if idx.size > 200:
                    pcd = pcd.select_by_index(idx.tolist())

        # ICP
        max_corr = float(self.get_parameter('icp_max_corr_dist').value)
        max_iter = int(self.get_parameter('icp_max_iter').value)
        reg = self.icp_multiscale(pcd, target_pcd, T_init, max_corr, max_iter)

        # Quality gate
        MIN_FIT = float(self.get_parameter('min_fitness').value)
        MAX_RMSE = 0.04
        if (float(reg.fitness) < MIN_FIT) or (float(reg.inlier_rmse) > MAX_RMSE):
            self.get_logger().warn(f"ICP rejected (fit={reg.fitness:.2f}, rmse={reg.inlier_rmse:.3f})")
            self.prev_pcd = pcd
            self.T_last_inc = np.eye(4)
            self.publish_pose(msg.header.stamp)
            return

        # Accumulate
        T_prev_to_curr = np.asarray(reg.transformation)
        self.T_odom_base = self.T_odom_base @ T_prev_to_curr
        self.T_last_inc = T_prev_to_curr.copy()

        # Update submap
        pcd_world = o3d.geometry.PointCloud(pcd)
        pcd_world.transform(self.T_odom_base)
        self._merge_into_submap(pcd_world, voxel)

        self.prev_pcd = pcd
        self.publish_pose(msg.header.stamp)
        self.get_logger().info(f"ICP OK: fit={reg.fitness:.2f} rmse={reg.inlier_rmse:.3f}")

    # ---------- Publish pose / path ----------
    def publish_pose(self, stamp):
        odom_frame = self.get_parameter('odom_frame').value
        base_frame = self.get_parameter('base_frame').value
        odom = odom_msg_from_T(self.T_odom_base, odom_frame, base_frame, stamp)
        self.pub_odom.publish(odom)
        self.tfb.sendTransform(tf_from_T(self.T_odom_base, odom_frame, base_frame, stamp))

        ps = PoseStamped()
        ps.header.frame_id = odom_frame
        ps.header.stamp = stamp
        ps.pose = odom.pose.pose
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(ps)
        if len(self.path_msg.poses) > 2000:
            self.path_msg.poses = self.path_msg.poses[-2000:]
        self.pub_path.publish(self.path_msg)


def main():
    rclpy.init()
    node = PipeOdomICPElbow()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

