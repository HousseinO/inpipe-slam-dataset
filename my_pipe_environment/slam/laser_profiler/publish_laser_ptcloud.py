#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import cv2

from rclpy.time import Time

from sensor_msgs.msg import PointCloud2, Imu
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped

import tf2_ros


def quat_from_R(R: np.ndarray):
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.2498 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.2498 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.2498 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.2498 * S
    return qx, qy, qz, qw


def tf_from_T(T: np.ndarray, frame_id: str, child_frame_id: str, stamp):
    """Build TransformStamped from 4x4 matrix T."""
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


class NewLaserProfilerFrame(Node):
    def __init__(self):
        super().__init__('new_laser_profiler_frame')

        # Parameters
        self.declare_parameter('cloud_topic', '/inpipe_bot/laser_profiler/points')
        self.declare_parameter('marker_topic', '/laser_profiler/ellipse_marker')
        self.declare_parameter('laser_frame', 'laser_profiler_link')
        self.declare_parameter('imu_frame', 'imu_link')
        self.declare_parameter('new_frame', 'new_laser_profiler')
        self.declare_parameter('plane_size', 0.6)

        cloud_topic = self.get_parameter(
            'cloud_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter(
            'marker_topic').get_parameter_value().string_value
        self.laser_frame = self.get_parameter(
            'laser_frame').get_parameter_value().string_value
        self.imu_frame = self.get_parameter(
            'imu_frame').get_parameter_value().string_value
        self.new_frame = self.get_parameter(
            'new_frame').get_parameter_value().string_value
        self.plane_size = self.get_parameter(
            'plane_size').get_parameter_value().double_value

        # TF2 (for static imu->laser rotation)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Broadcaster (same pattern as your icp node)
        self.tfb = tf2_ros.TransformBroadcaster(self)

        # Reference axis in laser frame, derived from IMU mount
        # IMU +X axis is pipe axis in imu frame when orientation is identity
        self.ref_axis_laser = None

        # Flags for "log once"
        self.imu_seen = False
        self.ref_axis_logged = False
        self.cloud_cb_logged = False

        # Subscribers / publishers
        self.cloud_sub = self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_callback,
            qos_profile_sensor_data)

        self.imu_sub = self.create_subscription(
            Imu, '/inpipe_bot/imu/data', self.imu_callback,
            qos_profile_sensor_data)

        self.marker_pub = self.create_publisher(Marker, marker_topic, 10)

        self.get_logger().info(
            f'new_laser_profiler_frame started. '
            f'Parent={self.laser_frame}, child={self.new_frame}, '
            f'cloud={cloud_topic}, imu={self.imu_frame}'
        )

    # IMU callback only to know IMU is alive
    def imu_callback(self, msg: Imu):
        if not self.imu_seen:
            self.get_logger().info(
                f'Received first IMU data. Orientation (x,y,z,w)=('
                f'{msg.orientation.x:.3f}, {msg.orientation.y:.3f}, '
                f'{msg.orientation.z:.3f}, {msg.orientation.w:.3f})'
            )
            self.imu_seen = True

    def _ensure_ref_axis(self):
        """Compute pipe axis direction in laser frame from IMU mount."""
        if self.ref_axis_laser is not None:
            return True

        try:
            # Transform: laser_frame <- imu_frame
            tf_imu_laser = self.tf_buffer.lookup_transform(
                self.laser_frame, self.imu_frame, Time())
        except Exception as e:
            # This might spam a bit, but better than silent failure
            self.get_logger().warn(
                f'Cannot get TF {self.laser_frame} <- {self.imu_frame}: {e}'
            )
            return False

        q = tf_imu_laser.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
        ], dtype=np.float64)

        e_x_imu = np.array([1.0, 0.0, 0.0])
        self.ref_axis_laser = R @ e_x_imu
        self.ref_axis_laser /= np.linalg.norm(self.ref_axis_laser)

        if not self.ref_axis_logged:
            self.get_logger().info(
                f'Reference pipe axis in laser frame (from IMU mount): '
                f'{self.ref_axis_laser}'
            )
            self.ref_axis_logged = True

        return True

    def cloud_callback(self, cloud: PointCloud2):
        if not self.cloud_cb_logged:
            self.get_logger().info('cloud_callback is running; will publish TF.')
            self.cloud_cb_logged = True

        # Make sure we know imu->laser relation (for sign of phi)
        self._ensure_ref_axis()

        # 1) Extract points
        pts = []
        for p in pc2.read_points(cloud, field_names=('x', 'y', 'z'),
                                 skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if len(pts) < 10:
            self.get_logger().warn('Not enough points to fit ellipse.')
            return
        pts = np.asarray(pts, dtype=np.float64)

        # 2) PCA to find laser plane
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)
        n = eigvecs[:, idx[0]]  # plane normal
        u = eigvecs[:, idx[1]]
        v = eigvecs[:, idx[2]]
        n /= np.linalg.norm(n)
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        # 3) Project into plane and fit ellipse in 2D
        xy = np.stack([pts_centered.dot(u),
                       pts_centered.dot(v)], axis=1)
        pts_cv = xy.astype(np.float32).reshape((-1, 1, 2))
        if pts_cv.shape[0] < 5:
            self.get_logger().warn('Not enough points for cv2.fitEllipse.')
            return

        try:
            (cx, cy), (len1, len2), angle_deg = cv2.fitEllipse(pts_cv)
        except cv2.error as e:
            self.get_logger().warn(f'cv2.fitEllipse failed: {e}')
            return

        # Ensure len1 is major, len2 is minor
        if len2 > len1:
            len1, len2 = len2, len1
            angle_deg += 90.0

        a = len1 / 2.0
        b = len2 / 2.0
        if a < 1e-6:
            self.get_logger().warn('Major radius too small.')
            return

        center_3d = centroid + cx * u + cy * v

        # 4) Major/minor directions in 3D
        angle_rad = np.deg2rad(angle_deg)
        major_dir_2d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        minor_dir_2d = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        major_dir_3d = major_dir_2d[0] * u + major_dir_2d[1] * v
        minor_dir_3d = minor_dir_2d[0] * u + minor_dir_2d[1] * v
        major_dir_3d /= np.linalg.norm(major_dir_3d)
        minor_dir_3d /= np.linalg.norm(minor_dir_3d)

        # 5) Compute |phi|
        ratio = np.clip(b / a, -1.0, 1.0)
        phi = np.arccos(ratio)   # unsigned

        # 6) Two candidate pipe axes around MINOR axis: ±phi
        cross_mn = np.cross(minor_dir_3d, n)
        k_plus  = n * np.cos(phi) - cross_mn * np.sin(phi)
        k_minus = n * np.cos(phi) + cross_mn * np.sin(phi)
        k_plus  /= np.linalg.norm(k_plus)
        k_minus /= np.linalg.norm(k_minus)

        # 7) Choose sign using IMU mount (ref_axis_laser)
        if self.ref_axis_laser is not None:
            dot_plus = np.dot(k_plus,  self.ref_axis_laser)
            dot_minus = np.dot(k_minus, self.ref_axis_laser)
            if dot_plus >= dot_minus:
                k = k_plus
                phi_signed = +phi
            else:
                k = k_minus
                phi_signed = -phi
        else:
            # No TF imu->laser: fall back to +phi
            self.get_logger().warn(
                'ref_axis_laser unknown; using +phi arbitrarily.'
            )
            k = k_plus
            phi_signed = phi

        k /= np.linalg.norm(k)

        # ---- Build new_laser_profiler frame axes ----
        z_new = k
        # y_new along minor axis, orthogonalized to z_new
        y_tmp = minor_dir_3d - np.dot(minor_dir_3d, z_new) * z_new
        if np.linalg.norm(y_tmp) < 1e-6:
            y_tmp = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(y_tmp, z_new)) > 0.9:
                y_tmp = np.array([1.0, 0.0, 0.0])
            y_tmp = y_tmp - np.dot(y_tmp, z_new) * z_new
        y_new = y_tmp / np.linalg.norm(y_tmp)
        x_new = np.cross(y_new, z_new)
        x_new /= np.linalg.norm(x_new)

        # 8) Build transform laser_profiler_link -> new_laser_profiler (4x4)
        R_new = np.column_stack((x_new, y_new, z_new))
        T_laser_new = np.eye(4, dtype=np.float64)
        T_laser_new[:3, :3] = R_new
        T_laser_new[:3, 3] = center_3d

        tf_msg = tf_from_T(
            T_laser_new,
            frame_id=self.laser_frame,
            child_frame_id=self.new_frame,
            stamp=self.get_clock().now().to_msg()
        )
        self.tfb.sendTransform(tf_msg)

        # 9) Optional: marker at center
        center_marker = Marker()
        center_marker.header.frame_id = self.laser_frame
        center_marker.header.stamp = tf_msg.header.stamp
        center_marker.ns = 'new_laser_profiler_center'
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
        center_marker.color.g = 1.0
        center_marker.color.b = 0.0
        center_marker.color.a = 1.0
        self.marker_pub.publish(center_marker)

        self.get_logger().info(
            f'center={center_3d}, a={a:.4f}, b={b:.4f}, '
            f'phi_signed={np.degrees(phi_signed):.2f} deg'
        )


def main(args=None):
    rclpy.init(args=args)
    node = NewLaserProfilerFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

