#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32
import math

class DualSonarMaxRangeNode(Node):
    def __init__(self):
        super().__init__('dual_sonar_max_range_node')

        # Subscriptions
        self.sub_left = self.create_subscription(
            PointCloud2,
            '/inpipe_bot/sonar/left/range',
            self.left_callback,
            10)

        self.sub_right = self.create_subscription(
            PointCloud2,
            '/inpipe_bot/sonar/right/range',
            self.right_callback,
            10)

        # Publishers
        self.pub_left = self.create_publisher(
            Float32,
            '/inpipe_bot/sonar/left/max_range',
            10)

        self.pub_right = self.create_publisher(
            Float32,
            '/inpipe_bot/sonar/right/max_range',
            10)

        self.get_logger().info("Dual Sonar Max Range Node Started")

    def compute_max_distance(self, msg: PointCloud2):
        max_dist = 0.0
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            dist = math.sqrt(x*x + y*y + z*z)
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def left_callback(self, msg: PointCloud2):
        max_dist = self.compute_max_distance(msg)
        out = Float32()
        out.data = max_dist
        self.pub_left.publish(out)

    def right_callback(self, msg: PointCloud2):
        max_dist = self.compute_max_distance(msg)
        out = Float32()
        out.data = max_dist
        self.pub_right.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = DualSonarMaxRangeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

