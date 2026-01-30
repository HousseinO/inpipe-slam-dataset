#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np

from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


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


class LaserProfilerOdomConstraint(Node):
    """
    Node that listens to the TF from laser_profiler_link -> new_laser_profiler
    (published by your NewLaserProfilerFrame node) and, for every pair of
    successive ellipses, publishes a relative pose constraint between the two
    laser poses, with translation projected onto the plane orthogonal to
    the pipe axis (z of new_laser_profiler).
    """

    def __init__(self):
        super().__init__('laser_profiler_odom_constraint')

        # Parameters
        self.declare_parameter('laser_frame', 'laser_profiler_link')
        self.declare_parameter('new_frame', 'new_laser_profiler')
        self.declare_parameter('output_topic', '/laser_profiler/odom_constraint')
        self.declare_parameter('update_rate', 20.0)  # Hz

        self.laser_frame = self.get_parameter(
            'laser_frame').get_parameter_value().string_value
        self.new_frame = self.get_parameter(
            'new_frame').get_parameter_value().string_value
        self.output_topic = self.get_parameter(
            'output_topic').get_parameter_value().string_value
        self.update_rate = self.get_parameter(
            'update_rate').get_parameter_value().double_value

        # TF2 listener (to read laser -> new_laser_profiler)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for odom-style relative constraint
        self.constraint_pub = self.create_publisher(
            PoseStamped, self.output_topic, 10
        )

        # Storage for previous ellipse info
        self.prev_stamp = None
        self.prev_axis = None   # a1  (cylinder axis in laser frame)
        self.prev_center = None # p1  (point on axis in laser frame)

        # Timer to poll TF for new ellipses
        period = 1.0 / max(self.update_rate, 1.0)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'laser_profiler_odom_constraint started. '
            f'Listening to TF {self.laser_frame} -> {self.new_frame}, '
            f'publishing constraints on {self.output_topic}'
        )

    def timer_callback(self):
        # Query the latest transform laser_frame -> new_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.laser_frame,
                self.new_frame,
                Time()  # latest
            )
        except Exception as e:
            self.get_logger().warn(
                f'Cannot get TF {self.laser_frame} -> {self.new_frame}: {e}',
                throttle_duration_sec=5.0
            )
            return

        stamp = tf.header.stamp

        # First ellipse: just store and return
        if self.prev_stamp is None:
            self._store_measurement(tf, stamp)
            return

        # If timestamp hasn't changed, no new ellipse yet
        if (stamp.sec == self.prev_stamp.sec and
                stamp.nanosec == self.prev_stamp.nanosec):
            return

        # New ellipse: compute constraint between prev and current
        a2, p2 = self._extract_axis_and_center(tf)
        if a2 is None:
            return

        a1 = self.prev_axis
        p1 = self.prev_center

        if a1 is None or p1 is None:
            self._store_measurement(tf, stamp)
            return

        # Radial vectors from axis to laser origin (origin = (0,0,0) in laser frame)
        r1 = p1 - np.dot(p1, a1) * a1
        r2 = p2 - np.dot(p2, a2) * a2

        if np.linalg.norm(r1) < 1e-6 or np.linalg.norm(r2) < 1e-6:
            self.get_logger().warn(
                'Radial vector too small; laser likely on axis. '
                'Cannot compute unique rotation.'
            )
            self._store_measurement(tf, stamp)
            return

        r1 /= np.linalg.norm(r1)
        r2 /= np.linalg.norm(r2)

        # Build 3x2 matrices of reference vectors
        # (axis + radial direction)
        A = np.column_stack((a1, r1))  # in frame at time 1
        B = np.column_stack((a2, r2))  # in frame at time 2

        # Solve for rotation R such that B ≈ R * A  (Wahba / Procrustes)
        M = B @ A.T  # 3x3
        U, _, Vt = np.linalg.svd(M)
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt

        # Project translation onto plane orthogonal to a2
        # P = I - a2 a2ᵀ  (projector onto plane ⟂ pipe axis)
        P = np.eye(3) - np.outer(a2, a2)
        t_perp = P @ (p2 - R @ p1)
        print('B = ', B)
        print('t_perp = ', t_perp)

        # Build and publish PoseStamped as relative constraint (frame1 -> frame2)
        qx, qy, qz, qw = quat_from_R(R)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        # NOTE: The constraint is a *delta* (relative pose) expressed in the
        # second laser frame's coordinates. We still set header.frame_id to
        # laser_frame; your SLAM node should treat this as a relative factor,
        # not an absolute pose.
        pose_msg.header.frame_id = self.laser_frame
        pose_msg.pose.position.x = float(t_perp[0])
        pose_msg.pose.position.y = float(t_perp[1])
        pose_msg.pose.position.z = float(t_perp[2])
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.constraint_pub.publish(pose_msg)

        self.get_logger().debug(
            f'Published constraint: t_perp={t_perp}, '
            f'a1={a1}, a2={a2}'
        )

        # Update stored measurement for next step
        self._store_measurement(tf, stamp)

    def _extract_axis_and_center(self, tf_msg):
        """
        From the TransformStamped laser_frame -> new_laser_profiler, extract:
        - axis a: z-axis of new_laser_profiler expressed in laser_frame
        - center p: translation (center of ellipse) expressed in laser_frame
        """
        # Translation: point on cylinder axis in laser frame
        t = tf_msg.transform.translation
        p = np.array([t.x, t.y, t.z], dtype=np.float64)

        # Rotation: get rotation matrix, then take z-axis
        q = tf_msg.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
        ], dtype=np.float64)

        a = R[:, 2]  # z-axis of new_laser_profiler in laser frame

        norm_a = np.linalg.norm(a)
        if norm_a < 1e-6:
            self.get_logger().warn('Axis norm too small when extracting from TF.')
            return None, None

        a /= norm_a
        return a, p

    def _store_measurement(self, tf_msg, stamp):
        a, p = self._extract_axis_and_center(tf_msg)
        if a is None:
            return
        self.prev_axis = a
        self.prev_center = p
        self.prev_stamp = stamp


def main(args=None):
    rclpy.init(args=args)
    node = LaserProfilerOdomConstraint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

