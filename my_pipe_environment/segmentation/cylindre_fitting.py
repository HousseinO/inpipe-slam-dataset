#!/usr/bin/env python3
# pipe_axis_from_cloud.py — ROS 2 Humble
# Subscribe to /inpipe_bot/l515/l515_depth/points and detect cylinder axis
# Method: voxel downsample -> PCA on points in target frame (same as your files)

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

import tf2_ros
import open3d as o3d

# ---------- helpers (same approach as your estimator) ----------

def tf_to_matrix(tf):
    t = tf.transform.translation; q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64); T[:3, :3] = R; T[:3, 3] = [t.x, t.y, t.z]
    return T

def voxel_downsample_xyz(xyz: np.ndarray, voxel=0.015) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points)

def pca_axis(xyz: np.ndarray):
    ctr = xyz.mean(axis=0)
    X = xyz - ctr
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = Vt[0, :] / np.linalg.norm(Vt[0, :])
    return axis, ctr

def quat_from_two_vectors(a: np.ndarray, b: np.ndarray):
    """Quaternion rotating vector a -> b (x,y,z,w). Robust for 180° case."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # 180 deg
        # find any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return (v[0], v[1], v[2], 0.0)
    q = np.array([v[0], v[1], v[2], 1.0 + c], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return tuple(q.tolist())

# ---------- node ----------

class PipeAxisFromCloud(Node):
    def __init__(self):
        super().__init__('pipe_axis_from_cloud')

        # Parameters
        self.declare_parameter('cloud_topic', '/inpipe_bot/l515/l515_depth/points')
        self.declare_parameter('target_frame', 'base_link')   # '' to keep source frame
        self.declare_parameter('voxel', 0.015)                # 1.5 cm
        self.declare_parameter('min_points_total', 400)
        self.declare_parameter('marker_length_m', 2.0)        # fallback if cloud span is small
        self.declare_parameter('marker_ns', 'pipe_axis')
        self.declare_parameter('marker_scale', 0.02)          # line width (m)

        topic = self.get_parameter('cloud_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        # I/O
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, 1)
        self.pub_axis_pose = self.create_publisher(PoseStamped, 'pipe/axis_pose', 1)
        self.pub_axis_marker = self.create_publisher(Marker, 'pipe/axis_marker', 1)

        # TF
        self.tfbuf = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tflist = tf2_ros.TransformListener(self.tfbuf, self)

        self.get_logger().info(f"Listening on {topic}, target_frame={self.target_frame or '[source frame]'}")

    def cb(self, msg: PointCloud2):
        # 1) Read cloud
        try:
            pts_src = pc2.read_points_numpy(msg, field_names=('x','y','z'), skip_nans=True)
        except Exception as e:
            self.get_logger().warn(f"read_points_numpy failed: {e}")
            return
        if pts_src.size == 0:
            return
        pts_src = pts_src[np.isfinite(pts_src).all(axis=1)]
        if pts_src.shape[0] < int(self.get_parameter('min_points_total').value):
            return

        # 2) Transform to target frame (same as your code)
        source = msg.header.frame_id
        target = self.target_frame or source
        if target != source:
            try:
                tf = self.tfbuf.lookup_transform(target, source, rclpy.time.Time(), timeout=Duration(seconds=0.2))
            except Exception as e:
                self.get_logger().warn(f"TF lookup {target} <- {source} failed: {e}")
                return
            T = tf_to_matrix(tf)
            pts = (T @ np.c_[pts_src.astype(np.float64), np.ones((pts_src.shape[0], 1))].T).T[:, :3]
            out_frame = target
        else:
            pts = pts_src.astype(np.float64); out_frame = source

        # 3) Voxel downsample (identical approach)
        voxel = float(self.get_parameter('voxel').value)
        pts_ds = voxel_downsample_xyz(pts, voxel)
        if pts_ds.shape[0] < 100:
            return

        # 4) PCA axis (identical approach)
        axis, center = pca_axis(pts_ds)

        # 5) Choose a marker length from the cloud span
        s = (pts_ds - center) @ axis
        if s.size:
            L = float(max(s.max() - s.min(), self.get_parameter('marker_length_m').value))
        else:
            L = float(self.get_parameter('marker_length_m').value)
        p0 = center - 0.5 * L * axis
        p1 = center + 0.5 * L * axis

        # 6) Publish PoseStamped with X-axis aligned to pipe axis
        qx, qy, qz, qw = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), axis)
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = out_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = center.tolist()
        pose.pose.orientation.x = qx; pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz; pose.pose.orientation.w = qw
        self.pub_axis_pose.publish(pose)

        # 7) Publish Marker (LINE_STRIP with two points)
        m = Marker()
        m.header.frame_id = out_frame
        m.header.stamp = msg.header.stamp
        m.ns = self.get_parameter('marker_ns').value
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = float(self.get_parameter('marker_scale').value)  # line width
        m.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0)
        m.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                    Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
        m.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()  # persistent
        self.pub_axis_marker.publish(m)

        self.get_logger().info(
            f"Axis {np.round(axis,3).tolist()} | center {np.round(center,3).tolist()} | span {L:.2f} m"
        )

def main():
    rclpy.init()
    node = PipeAxisFromCloud()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

