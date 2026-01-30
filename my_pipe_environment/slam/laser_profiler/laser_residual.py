#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from geometry_msgs.msg import TransformStamped

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from rclpy.time import Time


class LaserProfilerTransformer(Node):
    """
    Node that:
      - Subscribes to /pipe/odom (nav_msgs/Odometry)
      - Broadcasts that odom as a TF transform (header.frame_id -> base_link)
      - Subscribes to the laser profiler point cloud in laser_profiler_link
      - Transforms the cloud into base_link frame and republishes it
    """

    def __init__(self):
        super().__init__('laser_profiler_transformer')

        # Parameters
        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link'
        ).value

        self.laser_topic = self.declare_parameter(
            'laser_topic', '/inpipe_bot/laser_profiler/points'
        ).value

        self.odom_topic = self.declare_parameter(
            'odom_topic', '/pipe/odom'
        ).value

        self.output_topic = self.declare_parameter(
            'output_topic', '/inpipe_bot/laser_profiler/points_in_base_link'
        ).value

        # TF broadcaster for odom->base_link
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # TF listener for transforming clouds
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.laser_topic,
            self.cloud_callback,
            10
        )

        # Publisher for transformed cloud
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10
        )

        self.get_logger().info(
            f"LaserProfilerTransformer started.\n"
            f"  Subscribing to odom: {self.odom_topic}\n"
            f"  Subscribing to laser cloud: {self.laser_topic}\n"
            f"  Publishing transformed cloud: {self.output_topic}\n"
            f"  Target frame: {self.target_frame}"
        )

    # -------------------- ODOM CALLBACK -------------------- #
    def odom_callback(self, msg: Odometry):
        """
        Convert /pipe/odom into a TF transform.
        Typically:
          msg.header.frame_id   -> world/pipe frame (e.g., 'pipe' or 'odom')
          msg.child_frame_id    -> 'base_link'
          msg.pose.pose         -> pose of base_link in world/pipe
        """

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id
        # Use child_frame_id from odom, or fallback to target_frame
        t.child_frame_id = msg.child_frame_id if msg.child_frame_id else self.target_frame

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        # Broadcast the transform: world/pipe -> base_link
        self.tf_broadcaster.sendTransform(t)

    # ------------------ CLOUD CALLBACK --------------------- #
    def cloud_callback(self, msg: PointCloud2):
        """
        Transform the incoming laser profiler cloud into target_frame (base_link).

        This uses the same math as in the PDF:
            p' = R * p + t
        but is implemented through tf2 and do_transform_cloud().
        """

        try:
            # Look up transform: target_frame (base_link) <- source_frame (cloud frame)
            # Using latest available transform (Time()).
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,           # target frame
                msg.header.frame_id,         # source frame (e.g. 'laser_profiler_link')
                Time()                       # get latest transform
            )

        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().warn(
                f"Could not transform from {msg.header.frame_id} to {self.target_frame}: {ex}"
            )
            return

        # Apply the transform to the point cloud
        transformed_cloud = do_transform_cloud(msg, transform)

        # Optionally, you can modify the header.stamp to match transform time
        transformed_cloud.header.stamp = transform.header.stamp

        # Publish the transformed cloud
        self.cloud_pub.publish(transformed_cloud)


def main(args=None):
    rclpy.init(args=args)

    node = LaserProfilerTransformer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

