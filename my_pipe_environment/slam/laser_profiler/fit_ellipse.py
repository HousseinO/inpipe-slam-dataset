#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import cv2


class EllipseFitter(Node):
    def __init__(self):
        super().__init__('ellipse_fitter')

        # Parameters
        self.declare_parameter('cloud_topic', '/inpipe_bot/laser_profiler/points')
        self.declare_parameter('marker_topic', '/laser_profiler/ellipse_marker')

        cloud_topic = self.get_parameter(
            'cloud_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter(
            'marker_topic').get_parameter_value().string_value

        # Subscribers / publishers
        self.cloud_sub = self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_callback,
            qos_profile_sensor_data)

        self.marker_pub = self.create_publisher(Marker, marker_topic, 10)

        self.get_logger().info('ellipse_fitter node started.')

    # ----------------- main callback -----------------
    def cloud_callback(self, cloud: PointCloud2):
        # Extract xyz from cloud
        pts = []
        for p in pc2.read_points(cloud, field_names=('x', 'y', 'z'),
                                 skip_nans=True):
            pts.append([p[0], p[1], p[2]])

        if len(pts) < 5:
            self.get_logger().warn('Not enough points to fit ellipse.')
            return

        pts = np.asarray(pts, dtype=np.float64)

        # 1) PCA to find best-fit plane for the ring
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # normal is eigenvector with smallest eigenvalue
        idx = np.argsort(eigvals)
        normal = eigvecs[:, idx[0]]

        # two in-plane axes
        u = eigvecs[:, idx[1]]   # first in-plane axis
        v = eigvecs[:, idx[2]]   # second in-plane axis

        # 2) Project points into this plane (2D coordinates)
        xy = np.stack([
            pts_centered.dot(u),
            pts_centered.dot(v)
        ], axis=1)  # shape (N,2)

        # OpenCV fitEllipse expects N x 1 x 2 float32
        pts_cv = xy.astype(np.float32).reshape((-1, 1, 2))

        if pts_cv.shape[0] < 5:
            self.get_logger().warn('Not enough points for cv2.fitEllipse.')
            return

        try:
            ellipse = cv2.fitEllipse(pts_cv)
        except cv2.error as e:
            self.get_logger().warn(f'cv2.fitEllipse failed: {e}')
            return

        # ellipse: ((cx, cy), (major_len, minor_len), angle_deg)
        (cx, cy), (major_len, minor_len), angle_deg = ellipse

        # Radii (semi-axes)
        a = major_len / 2.0   # major radius
        b = minor_len / 2.0   # minor radius

        # 3) Compute 3D center
        center_2d = np.array([cx, cy])
        center_3d = centroid + center_2d[0] * u + center_2d[1] * v

        # 4) Sample points on fitted ellipse in 2D, then back to 3D
        angle_rad = np.deg2rad(angle_deg)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        num_samples = 100
        ts = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=True)

        ellipse_points_3d = []
        for t in ts:
            xe = a * np.cos(t)
            ye = b * np.sin(t)

            # rotate in 2D
            xr = xe * cos_t - ye * sin_t
            yr = xe * sin_t + ye * cos_t

            # translate to 2D center
            x2d = xr + cx
            y2d = yr + cy

            # back to 3D: centroid + x*u + y*v
            p3 = centroid + x2d * u + y2d * v
            ellipse_points_3d.append(p3)

        # 5) Publish markers in RViz

        # main ellipse marker (LINE_STRIP)
        ellipse_marker = Marker()
        ellipse_marker.header.frame_id = cloud.header.frame_id
        ellipse_marker.header.stamp = self.get_clock().now().to_msg()
        ellipse_marker.ns = 'fitted_ellipse'
        ellipse_marker.id = 0
        ellipse_marker.type = Marker.LINE_STRIP
        ellipse_marker.action = Marker.ADD

        ellipse_marker.scale.x = 0.01  # line width (meters)

        ellipse_marker.color.r = 0.0
        ellipse_marker.color.g = 1.0
        ellipse_marker.color.b = 0.0
        ellipse_marker.color.a = 1.0

        ellipse_marker.points = []
        for p in ellipse_points_3d:
            pt = Point()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
            ellipse_marker.points.append(pt)
        # Close the loop
        ellipse_marker.points.append(ellipse_marker.points[0])

        # center marker (small sphere)
        center_marker = Marker()
        center_marker.header.frame_id = cloud.header.frame_id
        center_marker.header.stamp = ellipse_marker.header.stamp
        center_marker.ns = 'fitted_ellipse'
        center_marker.id = 1
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD

        center_marker.pose.position.x = float(center_3d[0])
        center_marker.pose.position.y = float(center_3d[1])
        center_marker.pose.position.z = float(center_3d[2])
        center_marker.pose.orientation.w = 1.0  # identity orientation

        center_marker.scale.x = 0.03
        center_marker.scale.y = 0.03
        center_marker.scale.z = 0.03

        center_marker.color.r = 0.0
        center_marker.color.g = 1.0
        center_marker.color.b = 0.0
        center_marker.color.a = 1.0

        self.marker_pub.publish(ellipse_marker)
        self.marker_pub.publish(center_marker)

        # Log useful numbers
        self.get_logger().info(
            f'Ellipse center (3D): {center_3d}, '
            f'axes (a,b) = ({a:.4f}, {b:.4f}) [m], '
            f'angle = {angle_deg:.2f} deg in plane'
        )


def main(args=None):
    rclpy.init(args=args)
    node = EllipseFitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

