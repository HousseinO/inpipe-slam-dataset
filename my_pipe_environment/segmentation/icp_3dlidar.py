#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion


class PipeCylinderPCAImproved(Node):
    """
    Same pipeline as original node, but PCA is made more robust:

      - Do a first PCA to get a rough axis.
      - Project points onto this axis and split into bins along the axis.
      - Compute one representative point per bin (mean).
      - Run a second PCA on those bin centroids to get final axis.

    This reduces the bias from regions with many more points.
    """

    def __init__(self):
        super().__init__('pipe_cylinder_pca_improved')

        # Parameters (same names as your original node so you can reuse launch files)
        self.declare_parameter('cloud_topic', '/inpipe_bot/lidar/points')
        self.declare_parameter('cloud_out_topic', '/inpipe_bot/lidar/points_baselink')
        self.declare_parameter('marker_topic', '/pipe_cylinder_markers_pca')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('pipe_diameter', 0.5)
        self.declare_parameter('max_points', 8000)
        self.declare_parameter('min_points', 500)
        self.declare_parameter('line_width', 0.02)
        self.declare_parameter('min_length', 1.5)

        self.declare_parameter('min_x', 0.7)
        self.declare_parameter('max_x', 3.0)

        # extra params for improved PCA
        self.declare_parameter('num_axis_bins', 30)  # how many slices along pipe axis

        cloud_topic = self.get_parameter('cloud_topic').value
        cloud_out_topic = self.get_parameter('cloud_out_topic').value
        marker_topic = self.get_parameter('marker_topic').value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.sub = self.create_subscription(
            PointCloud2,
            cloud_topic,
            self.cloud_callback,
            qos
        )

        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, cloud_out_topic, 10)

        # Precompute static transform lidar_link -> base_link (same as your original node)
        tx, ty, tz = 0.2, 0.0, 0.08
        roll, pitch, yaw = 0.0, 0.523599, 0.0

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        self.R_bl = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=np.float32)

        self.t_bl = np.array([tx, ty, tz], dtype=np.float32)

        self.get_logger().info(
            f'[PCA-Improved] listening on {cloud_topic}, '
            f'publishing transformed cloud on {cloud_out_topic}, '
            f'markers on {marker_topic}'
        )

    # -------------------- Point cloud callback -------------------- #
    def cloud_callback(self, msg: PointCloud2):
        max_points = self.get_parameter('max_points').value
        min_points = self.get_parameter('min_points').value
        min_x = float(self.get_parameter('min_x').value)
        max_x = float(self.get_parameter('max_x').value)
        base_frame = self.get_parameter('base_frame').value

        points_iter = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts_list = [ [float(p[0]), float(p[1]), float(p[2])] for p in points_iter ]

        if not pts_list:
            return

        pts_lidar = np.asarray(pts_list, dtype=np.float32)

        # transform to base_link
        pts_base = pts_lidar @ self.R_bl.T + self.t_bl

        # filter forward region
        mask = (pts_base[:, 0] > min_x) & (pts_base[:, 0] < max_x)
        pts_base_filtered = pts_base[mask]

        if pts_base_filtered.shape[0] < min_points:
            return

        # publish filtered cloud
        header = msg.header
        header.frame_id = base_frame
        cloud_out = pc2.create_cloud_xyz32(header, pts_base_filtered.tolist())
        self.cloud_pub.publish(cloud_out)

        # downsample for fitting
        if pts_base_filtered.shape[0] > max_points:
            idx = np.random.choice(pts_base_filtered.shape[0], size=max_points, replace=False)
            pts_fit = pts_base_filtered[idx]
        else:
            pts_fit = pts_base_filtered

        center, axis_dir, length = self.fit_cylinder_pca_improved(pts_fit)
        if center is None:
            return

        marker_array = self.make_markers(
            base_frame,
            msg.header.stamp,
            center,
            axis_dir,
            length
        )
        self.marker_pub.publish(marker_array)

    # -------------------- Improved PCA -------------------- #
    def fit_cylinder_pca_improved(self, pts):
        """
        Two-stage PCA:
          1) PCA on all points -> rough axis
          2) Project points onto that axis, divide into bins, compute one
             representative point per bin, PCA again on those bin-centers.

        This reduces the effect of dense local clusters.
        """
        if pts.shape[0] < 3:
            return None, None, None

        # first PCA
        center0 = pts.mean(axis=0)
        centered0 = pts - center0
        cov0 = np.cov(centered0.T)
        eigvals, eigvecs = np.linalg.eigh(cov0)
        axis0 = eigvecs[:, np.argmax(eigvals)]
        axis0 = axis0 / np.linalg.norm(axis0)

        # project onto rough axis
        proj = centered0 @ axis0

        num_bins = int(self.get_parameter('num_axis_bins').value)
        if num_bins < 5:
            num_bins = 5

        t_min, t_max = proj.min(), proj.max()
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            self.get_logger().warn('Improved PCA: invalid projection range.')
            return None, None, None

        edges = np.linspace(t_min, t_max, num_bins + 1)

        bin_centers = []
        for i in range(num_bins):
            mask = (proj >= edges[i]) & (proj < edges[i + 1])
            if not np.any(mask):
                continue
            bin_pts = pts[mask]
            bin_centers.append(bin_pts.mean(axis=0))

        if len(bin_centers) < 3:
            # fallback to original PCA
            self.get_logger().warn('Improved PCA: too few bins with points, falling back to simple PCA.')
            return self.fit_cylinder_simple_pca(pts)

        bin_centers = np.vstack(bin_centers)

        # second PCA on bin centroids
        center = bin_centers.mean(axis=0)
        centered = bin_centers - center
        cov = np.cov(centered.T)
        eigvals2, eigvecs2 = np.linalg.eigh(cov)
        axis_dir = eigvecs2[:, np.argmax(eigvals2)]
        axis_dir = axis_dir / np.linalg.norm(axis_dir)

        # compute length from original points projected on final axis
        proj_all = (pts - center) @ axis_dir
        min_t = proj_all.min()
        max_t = proj_all.max()
        length = float(max_t - min_t)

        min_length = float(self.get_parameter('min_length').value)
        if length < min_length:
            length = min_length

        if not np.isfinite(length) or length <= 0.0:
            self.get_logger().warn('Improved PCA: invalid length.')
            return None, None, None

        return center, axis_dir, length

    def fit_cylinder_simple_pca(self, pts):
        center = pts.mean(axis=0)
        centered = pts - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis_dir = eigvecs[:, np.argmax(eigvals)]
        axis_dir = axis_dir / np.linalg.norm(axis_dir)

        proj = centered @ axis_dir
        length = float(proj.max() - proj.min())
        min_length = float(self.get_parameter('min_length').value)
        if length < min_length:
            length = min_length

        if not np.isfinite(length) or length <= 0.0:
            return None, None, None

        return center, axis_dir, length

    # -------------------- Quaternion helper -------------------- #
    def quat_from_z_to_vec(self, v):
        v = np.asarray(v, dtype=float)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        v = v / v_norm

        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.dot(z_axis, v))

        if dot > 0.999999:
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        if dot < -0.999999:
            return Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)

        cross = np.cross(z_axis, v)
        s = math.sqrt((1.0 + dot) * 2.0)
        inv_s = 1.0 / s
        qx, qy, qz = cross * inv_s
        qw = 0.5 * s

        q = Quaternion()
        q.x = float(qx)
        q.y = float(qy)
        q.z = float(qz)
        q.w = float(qw)
        return q

    # -------------------- Marker construction -------------------- #
    def make_markers(self, frame_id, stamp, center, axis_dir, length):
        line_width = float(self.get_parameter('line_width').value)
        pipe_diameter = float(self.get_parameter('pipe_diameter').value)
        radius = pipe_diameter / 2.0

        marker_array = MarkerArray()

        def init_marker(marker_id, marker_type):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = 'pipe_cylinder_pca'
            m.id = marker_id
            m.type = marker_type
            m.action = Marker.ADD
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            m.color.a = 1.0
            return m

        # cylinder
        cyl = init_marker(0, Marker.CYLINDER)
        cyl.color.r = 0.0
        cyl.color.g = 0.6
        cyl.color.b = 1.0
        cyl.pose.position.x = float(center[0])
        cyl.pose.position.y = float(center[1])
        cyl.pose.position.z = float(center[2])
        cyl.pose.orientation = self.quat_from_z_to_vec(axis_dir)
        cyl.scale.x = 2.0 * radius
        cyl.scale.y = 2.0 * radius
        cyl.scale.z = length
        marker_array.markers.append(cyl)

        # axis line
        axis_marker = init_marker(1, Marker.LINE_LIST)
        axis_marker.color.r = 1.0
        axis_marker.color.g = 0.0
        axis_marker.color.b = 0.0
        axis_marker.scale.x = line_width

        half_len = length / 2.0
        start = center - axis_dir * half_len
        end = center + axis_dir * half_len
        p_start = Point(x=float(start[0]), y=float(start[1]), z=float(start[2]))
        p_end = Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))
        axis_marker.points.append(p_start)
        axis_marker.points.append(p_end)
        marker_array.markers.append(axis_marker)

        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = PipeCylinderPCAImproved()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

