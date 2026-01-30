#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import cv2

from sensor_msgs.msg import PointCloud2, Imu
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class EllipseFitter(Node):
    def __init__(self):
        super().__init__('ellipse_fitter')

        # Parameters
        self.declare_parameter('cloud_topic', '/inpipe_bot/laser_profiler/points')
        self.declare_parameter('marker_topic', '/laser_profiler/ellipse_marker')
        self.declare_parameter('plane_size', 0.6)   # plane patch edge length (m)

        cloud_topic = self.get_parameter(
            'cloud_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter(
            'marker_topic').get_parameter_value().string_value
        self.plane_size = self.get_parameter(
            'plane_size').get_parameter_value().double_value

        # IMU (gravity in laser frame, approximated from linear acceleration)
        self.g_laser = None

        # Subscribers
        self.cloud_sub = self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_callback,
            qos_profile_sensor_data)

        self.imu_sub = self.create_subscription(
            Imu, '/inpipe_bot/imu/data', self.imu_callback,
            qos_profile_sensor_data)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, marker_topic, 10)
        self.corrected_cloud_pub = self.create_publisher(
            PointCloud2,
            '/inpipe_bot/laser_profiler/points_corrected',
            10
        )

        self.get_logger().info('ellipse_fitter node started.')

    # ----------------- IMU callback: estimate gravity in laser frame -----------------
    def imu_callback(self, msg: Imu):
        # Use linear acceleration as gravity (assuming quasi-static robot)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        g = np.array([ax, ay, az], dtype=np.float64)
        nrm = np.linalg.norm(g)
        if nrm < 1e-6:
            return
        self.g_laser = g / nrm

    # ----------------- main callback -----------------
    def cloud_callback(self, cloud: PointCloud2):
        # Extract xyz from cloud
        pts = []
        for p in pc2.read_points(cloud, field_names=('x', 'y', 'z'),
                                 skip_nans=True):
            pts.append([p[0], p[1], p[2]])

        if len(pts) < 10:
            self.get_logger().warn('Not enough points to fit ellipse.')
            return

        pts = np.asarray(pts, dtype=np.float64)

        # 1) PCA to find best-fit plane for the ring (laser plane)
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # smallest eigenvalue -> plane normal
        idx = np.argsort(eigvals)
        n = eigvecs[:, idx[0]]         # laser plane normal
        u = eigvecs[:, idx[1]]         # in-plane axis 1
        v = eigvecs[:, idx[2]]         # in-plane axis 2

        # normalize
        n /= np.linalg.norm(n)
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        # 2) Project points into this plane (2D coords in basis u,v)
        xy = np.stack([
            pts_centered.dot(u),
            pts_centered.dot(v)
        ], axis=1)  # shape (N, 2)

        # OpenCV fitEllipse expects N x 1 x 2 float32
        pts_cv = xy.astype(np.float32).reshape((-1, 1, 2))

        if pts_cv.shape[0] < 5:
            self.get_logger().warn('Not enough points for cv2.fitEllipse.')
            return

        try:
            (cx, cy), (len1, len2), angle_deg = cv2.fitEllipse(pts_cv)
        except cv2.error as e:
            self.get_logger().warn(f'cv2.fitEllipse failed: {e}')
            return

        # OpenCV does NOT guarantee len1 >= len2; enforce major >= minor
        if len2 > len1:
            len1, len2 = len2, len1
            angle_deg += 90.0  # swapping axes rotates ellipse by 90 degrees

        # Semi-axes (major = a, minor = b)
        a = len1 / 2.0
        b = len2 / 2.0

        if a < 1e-6:
            self.get_logger().warn('Major radius too small, skip plane estimation.')
            return

        # 3) 3D center of ellipse
        center_3d = centroid + cx * u + cy * v

        # 4) Sample points on fitted ellipse in 3D (for visualization)
        angle_rad = np.deg2rad(angle_deg)

        num_samples = 100
        ts = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=True)

        ellipse_points_3d = []
        for t in ts:
            xe = a * np.cos(t)
            ye = b * np.sin(t)

            # 2D rotation in plane (u,v)
            xr = xe * np.cos(angle_rad) - ye * np.sin(angle_rad)
            yr = xe * np.sin(angle_rad) + ye * np.cos(angle_rad)

            x2d = xr + cx
            y2d = yr + cy

            p3 = centroid + x2d * u + y2d * v
            ellipse_points_3d.append(p3)

        # 5) Compute unsigned tilt angle |phi|
        # Ideal pipe: minor radius b = pipe radius R, major a = R / cos(phi)
        # => cos(phi) = b/a
        ratio = np.clip(b / a, -1.0, 1.0)
        phi = np.arccos(ratio)   # unsigned tilt (>= 0)

        # Major/minor directions in plane (2D) after enforcing major >= minor
        major_dir_2d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        minor_dir_2d = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        # Lift to 3D
        major_dir_3d = major_dir_2d[0] * u + major_dir_2d[1] * v
        minor_dir_3d = minor_dir_2d[0] * u + minor_dir_2d[1] * v
        major_dir_3d /= np.linalg.norm(major_dir_3d)
        minor_dir_3d /= np.linalg.norm(minor_dir_3d)

        # 6) Use IMU gravity to choose sign of phi and the correct axis k
        if self.g_laser is None:
            # No IMU yet: choose +phi arbitrarily (phi_signed > 0, but orientation ambiguous)
            self.get_logger().warn_throttle(2000,
                                            'No IMU gravity yet, using unsigned phi (+).')
            cross_mn = np.cross(minor_dir_3d, n)
            k = n * np.cos(phi) - cross_mn * np.sin(phi)
            k /= np.linalg.norm(k)
            phi_signed = phi
        else:
            g = self.g_laser

            # Two possible pipe axes: rotate n around minor axis by ±phi
            cross_mn = np.cross(minor_dir_3d, n)
            k_plus  = n * np.cos(phi) - cross_mn * np.sin(phi)
            k_minus = n * np.cos(phi) + cross_mn * np.sin(phi)
            k_plus  /= np.linalg.norm(k_plus)
            k_minus /= np.linalg.norm(k_minus)

            # Horizontal direction in cross-section: h = normalized( g x minor )
            # This defines "left-right" relative to gravity and minor axis.
            h = np.cross(g, minor_dir_3d)
            if np.linalg.norm(h) < 1e-6:
                # Degenerate (minor almost parallel to gravity); fall back to g x n
                h = np.cross(g, n)
            h /= np.linalg.norm(h)

            # Choose axis whose component along +h is larger
            if np.dot(k_plus, h) >= np.dot(k_minus, h):
                k = k_minus
                phi_signed = phi
            else:
                k =  k_plus
                phi_signed = phi

        # k is now the "true" pipe-axis direction (up to your sign convention)
        k /= np.linalg.norm(k)

        # 7) Build a plane patch centered at ellipse center, normal = k
        size = self.plane_size / 2.0

        # Build an orthonormal basis (e1, e2) spanning the plane with normal k
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, k)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - np.dot(tmp, k) * k
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(k, e1)
        e2 /= np.linalg.norm(e2)

        corners = []
        corners.append(center_3d + size * e1 + size * e2)
        corners.append(center_3d - size * e1 + size * e2)
        corners.append(center_3d - size * e1 - size * e2)
        corners.append(center_3d + size * e1 - size * e2)

        # 8) PROJECT ORIGINAL POINTS TO NEW PLANE (pipe cross-section)
        #
        # Plane: normal = k, passes through center_3d.
        corrected_pts = []
        for p in pts:
            vec = p - center_3d
            d = np.dot(vec, k)
            p_corr = p - d * k
            corrected_pts.append(p_corr.tolist())

        # Build and publish corrected point cloud
        header = cloud.header
        header.stamp = self.get_clock().now().to_msg()
        corrected_cloud = pc2.create_cloud_xyz32(header, corrected_pts)
        self.corrected_cloud_pub.publish(corrected_cloud)

        # 9) Create markers

        # Ellipse marker
        ellipse_marker = Marker()
        ellipse_marker.header.frame_id = cloud.header.frame_id
        ellipse_marker.header.stamp = self.get_clock().now().to_msg()
        ellipse_marker.ns = 'fitted_ellipse'
        ellipse_marker.id = 0
        ellipse_marker.type = Marker.LINE_STRIP
        ellipse_marker.action = Marker.ADD
        ellipse_marker.scale.x = 0.01  # line width

        ellipse_marker.color.r = 0.0
        ellipse_marker.color.g = 1.0
        ellipse_marker.color.b = 0.0
        ellipse_marker.color.a = 1.0

        ellipse_marker.points = []
        for p3 in ellipse_points_3d:
            pt = Point()
            pt.x = float(p3[0])
            pt.y = float(p3[1])
            pt.z = float(p3[2])
            ellipse_marker.points.append(pt)
        ellipse_marker.points.append(ellipse_marker.points[0])  # close loop

        # Center marker
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
        center_marker.pose.orientation.w = 1.0

        center_marker.scale.x = 0.03
        center_marker.scale.y = 0.03
        center_marker.scale.z = 0.03

        center_marker.color.r = 1.0
        center_marker.color.g = 0.0
        center_marker.color.b = 0.0
        center_marker.color.a = 1.0

        # Plane marker (cross-section plane, normal = k)
        plane_marker = Marker()
        plane_marker.header.frame_id = cloud.header.frame_id
        plane_marker.header.stamp = ellipse_marker.header.stamp
        plane_marker.ns = 'pipe_plane'
        plane_marker.id = 2
        plane_marker.type = Marker.LINE_STRIP
        plane_marker.action = Marker.ADD

        plane_marker.scale.x = 0.01

        plane_marker.color.r = 0.0
        plane_marker.color.g = 0.0
        plane_marker.color.b = 1.0
        plane_marker.color.a = 0.7

        plane_marker.points = []
        for c in corners:
            pt = Point()
            pt.x = float(c[0])
            pt.y = float(c[1])
            pt.z = float(c[2])
            plane_marker.points.append(pt)
        plane_marker.points.append(plane_marker.points[0])  # close square

        # Optional: arrow showing pipe-axis direction k
        axis_marker = Marker()
        axis_marker.header.frame_id = cloud.header.frame_id
        axis_marker.header.stamp = ellipse_marker.header.stamp
        axis_marker.ns = 'pipe_axis'
        axis_marker.id = 3
        axis_marker.type = Marker.ARROW
        axis_marker.action = Marker.ADD
        axis_marker.scale.x = 0.03   # shaft diameter
        axis_marker.scale.y = 0.06   # head diameter
        axis_marker.scale.z = 0.1    # head length

        axis_marker.color.r = 1.0
        axis_marker.color.g = 1.0
        axis_marker.color.b = 0.0
        axis_marker.color.a = 1.0

        axis_marker.points = []
        p_start = Point()
        p_start.x = float(center_3d[0])
        p_start.y = float(center_3d[1])
        p_start.z = float(center_3d[2])
        p_end = Point()
        L = 0.3
        p_end.x = float(center_3d[0] + L * k[0])
        p_end.y = float(center_3d[1] + L * k[1])
        p_end.z = float(center_3d[2] + L * k[2])
        axis_marker.points.append(p_start)
        axis_marker.points.append(p_end)

        # Publish markers
        self.marker_pub.publish(ellipse_marker)
        self.marker_pub.publish(center_marker)
        self.marker_pub.publish(plane_marker)
        self.marker_pub.publish(axis_marker)

        # Log some useful info (with signed phi)
        self.get_logger().info(
            f'Ellipse center (3D): {center_3d}, '
            f'axes (a,b)=({a:.4f},{b:.4f}) [m], '
            f'tilt phi_signed={np.degrees(phi_signed):.2f} deg'
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

