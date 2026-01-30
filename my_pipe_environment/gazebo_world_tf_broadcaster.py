#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class P3DTfBroadcaster(Node):
    def __init__(self):
        super().__init__('p3d_tf_broadcaster')
        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True
            )
        ])

        # Parameters (override in launch if you like)
        self.declare_parameter('parent_frame', 'ground_truth')
        self.declare_parameter('child_frame', 'base_link_truth')
        self.declare_parameter('topic', '/inpipe_bot/p3d_demo')

        self.parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value
        self.child_frame = self.get_parameter('child_frame').get_parameter_value().string_value
        topic = self.get_parameter('topic').get_parameter_value().string_value

        self.tf_broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, topic, self.odom_cb, 10)

        self.get_logger().info(
            f"Listening to '{topic}' and broadcasting TF: "
            f"{self.parent_frame} -> {self.child_frame}"
        )

    def odom_cb(self, msg: Odometry):
        # Build TransformStamped from Odometry pose
        t = TransformStamped()
        # Use the incoming timestamp if present; otherwise use now()
        t.header.stamp = msg.header.stamp if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0 else self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        # Broadcast
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = P3DTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

