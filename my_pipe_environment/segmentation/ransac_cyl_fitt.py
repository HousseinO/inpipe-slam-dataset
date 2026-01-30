#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class LidarToBaseCloudNode(Node):
    def __init__(self):
        super().__init__('lidar_to_base_cloud')

        # Parameters
        self.declare_parameter('input_topic', '/inpipe_bot/lidar/points')
        self.declare_parameter('output_topic', '/inpipe_bot/lidar/points_base')
        self.declare_parameter('target_frame', 'base_link')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        # TF buffer + listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriber & publisher
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.cloud_callback,
            10
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            output_topic,
            10
        )

        self.get_logger().info(
            f"LidarToBaseCloudNode started. Subscribing: {input_topic}, "
            f"Publishing: {output_topic}, Target frame: {self.target_frame}"
        )

    def cloud_callback(self, msg: PointCloud2):
        # Source frame is whatever the cloud says
        source_frame = msg.header.frame_id

        if not source_frame:
            self.get_logger().warn("Received PointCloud2 with empty frame_id, skipping.")
            return

        try:
            # Lookup transform: target_frame <- source_frame at time of msg
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                msg.header.stamp,
                timeout=Duration(seconds=0.1)
            )

        except Exception as e:
            self.get_logger().warn(
                f"Could not transform from {source_frame} to {self.target_frame}: {e}"
            )
            return

        try:
            # Apply the transform to the cloud
            transformed_cloud: PointCloud2 = do_transform_cloud(msg, transform)

            # Make sure frame_id is the target frame
            transformed_cloud.header.frame_id = self.target_frame

            self.publisher.publish(transformed_cloud)

        except Exception as e:
            self.get_logger().error(f"Error transforming PointCloud2: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LidarToBaseCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

