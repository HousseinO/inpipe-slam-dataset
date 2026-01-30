#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def fit_circle_kasa(y, z):
    """
    Fit a circle to 2D points (y, z) using the Kåsa least-squares method.

    Model: (y - yc)^2 + (z - zc)^2 = R^2
    Expanded: D*y + E*z + F + (y^2 + z^2) = 0
    Solve for D, E, F in least squares, then:
        yc = -D/2
        zc = -E/2
        R  = sqrt((D^2 + E^2)/4 - F)

    Returns (yc, zc, R) or None on failure.
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if y.size < 3:
        return None

    # Build least-squares system
    A = np.column_stack((y, z, np.ones_like(y)))          # shape (N,3)
    b = -(y**2 + z**2)                                    # shape (N,)

    try:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    D, E, F = params

    yc = -D / 2.0
    zc = -E / 2.0
    discriminant = (D**2 + E**2) / 4.0 - F

    if discriminant <= 0.0:
        return None

    R = np.sqrt(discriminant)

    if not (np.isfinite(yc) and np.isfinite(zc) and np.isfinite(R) and R > 0):
        return None

    return float(yc), float(zc), float(R)


class PipeCrossSectionsNode(Node):
    """
    Subscribes to /inpipe_bot/lidar/points_base (frame base_link),
    takes true planar cross-sections at given X positions, fits a 2D circle
    in each cross-section, and publishes the fitted circles and centers
    as markers for RViz.

    Pipe axis is assumed to be +X of base_link.
    """

    def __init__(self):
        super().__init__('pipe_cross_sections_circle')

        # Parameters
        self.declare_parameter('input_topic', '/inpipe_bot/lidar/points_base')
        self.declare_parameter('slice_centers', [0.4, 0.8, 0.7, 0.9, 0.5, 1.0, 1.5, 2.0])
        # Half-thickness of the slice along X (meters)
        self.declare_parameter('slice_half_thickness', 0.01)  # +/- 1 cm
        self.declare_parameter('min_points', 20)

        input_topic = self.get_parameter('input_topic').value
        self.slice_centers = list(self.get_parameter('slice_centers').value)
        self.slice_half_thickness = float(
            self.get_parameter('slice_half_thickness').value
        )
        self.min_points = int(self.get_parameter('min_points').value)

        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.cloud_callback, 10
        )

        # Publishers: one topic for all circles, one for all centers
        self.ellipse_pub = self.create_publisher(Marker, '/pipe_ellipses', 10)
        self.center_pub = self.create_publisher(Marker, '/pipe_ellipse_centers', 10)

        self.get_logger().info(
            f"[pipe_cross_sections_circle] Subscribing to {input_topic}, "
            f"slices at X = {self.slice_centers}, "
            f"half_thickness = {self.slice_half_thickness} m"
        )

    def cloud_callback(self, cloud: PointCloud2):
        # Read all xyz points from cloud
        pts_iter = point_cloud2.read_points(
            cloud, field_names=('x', 'y', 'z'), skip_nans=True
        )

        # Collect (y,z) per slice using *true planar* criterion |x - x_slice| < thickness
        slice_y = {c: [] for c in self.slice_centers}
        slice_z = {c: [] for c in self.slice_centers}

        for x, y, z in pts_iter:
            for x_s in self.slice_centers:
                if abs(x - x_s) <= self.slice_half_thickness:
                    slice_y[x_s].append(y)
                    slice_z[x_s].append(z)

        header = cloud.header  # should have frame_id = base_link

        # Colors for slices
        colors = [
            (1.0, 0.0, 0.0),   # red
            (0.0, 1.0, 0.0),   # green
            (0.0, 0.0, 1.0),   # blue
            (1.0, 1.0, 0.0),   # yellow
        ]

        ellipse_id = 0
        center_id = 0

        for i, x_s in enumerate(self.slice_centers):
            ys = slice_y[x_s]
            zs = slice_z[x_s]

            if len(ys) < self.min_points:
                # Not enough points in this slice
                continue

            # Fit a geometric circle to the cross-section points (y,z)
            result = fit_circle_kasa(ys, zs)
            if result is None:
                self.get_logger().warn(
                    f"Circle fit failed for slice at x={x_s:.2f} (N={len(ys)})"
                )
                continue

            yc, zc, radius = result

            # ---------- Circle marker (as LINE_STRIP) ----------
            ellipse_marker = Marker()
            ellipse_marker.header = header
            ellipse_marker.ns = 'pipe_cross_section'
            ellipse_marker.id = ellipse_id
            ellipse_marker.type = Marker.LINE_STRIP
            ellipse_marker.action = Marker.ADD
            ellipse_marker.pose.orientation.w = 1.0  # identity
            ellipse_marker.scale.x = 0.01  # line thickness

            r, g, b = colors[i % len(colors)]
            ellipse_marker.color.r = r
            ellipse_marker.color.g = g
            ellipse_marker.color.b = b
            ellipse_marker.color.a = 1.0

            num_samples = 100
            ts = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=True)

            for t in ts:
                ct = np.cos(t)
                st = np.sin(t)

                p = Point()
                p.x = float(x_s)
                p.y = float(yc + radius * ct)
                p.z = float(zc + radius * st)
                ellipse_marker.points.append(p)

            # close the circle
            if ellipse_marker.points:
                ellipse_marker.points.append(ellipse_marker.points[0])

            self.ellipse_pub.publish(ellipse_marker)
            ellipse_id += 1

            # ---------- Center marker (SPHERE) ----------
            center_marker = Marker()
            center_marker.header = header
            center_marker.ns = 'pipe_cross_section_center'
            center_marker.id = center_id
            center_marker.type = Marker.SPHERE
            center_marker.action = Marker.ADD
            center_marker.pose.position.x = float(x_s)
            center_marker.pose.position.y = float(yc)
            center_marker.pose.position.z = float(zc)
            center_marker.pose.orientation.w = 1.0

            center_marker.scale.x = 0.05
            center_marker.scale.y = 0.05
            center_marker.scale.z = 0.05

            center_marker.color.r = r
            center_marker.color.g = g
            center_marker.color.b = b
            center_marker.color.a = 1.0

            self.center_pub.publish(center_marker)
            center_id += 1


def main(args=None):
    rclpy.init(args=args)
    node = PipeCrossSectionsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

