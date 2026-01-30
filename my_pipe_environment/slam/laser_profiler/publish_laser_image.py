#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2

from cv_bridge import CvBridge
import cv2

import tf2_ros
from tf2_ros import TransformException
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sm


class LaserToImage(Node):
    def __init__(self):
        super().__init__('laser_to_image')
        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True
            )
        ])

        # Parameters (can be changed in launch file)
        self.declare_parameter('camera_frame', 'l515_color_optical_frame')
        self.declare_parameter('cloud_topic', '/inpipe_bot/laser_profiler/points')
        self.declare_parameter('image_topic', '/inpipe_bot/l515/l515_rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/inpipe_bot/l515/l515_rgb/camera_info')
        self.declare_parameter('output_image_topic', '/laser_profiler/image')

        self.camera_frame = self.get_parameter(
            'camera_frame').get_parameter_value().string_value
        cloud_topic = self.get_parameter(
            'cloud_topic').get_parameter_value().string_value
        image_topic = self.get_parameter(
            'image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter(
            'camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter(
            'output_image_topic').get_parameter_value().string_value

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics
        self.cam_info = None
        self.fx = self.fy = self.cx = self.cy = None

        # Last image from the camera
        self.last_image = None

        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data)

        self.cinfo_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback,
            qos_profile_sensor_data)

        self.cloud_sub = self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_callback,
            qos_profile_sensor_data)

        # Publisher (projected laser on image)
        self.image_pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info('laser_to_image node started.')

    # --------- Callbacks ---------

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        if self.cam_info is None:
            self.get_logger().info('Received first CameraInfo.')
        self.cam_info = msg

        # In ROS 2 CameraInfo the field is 'k' (lower-case)
        K = msg.k
        self.fx, self.fy = K[0], K[4]
        self.cx, self.cy = K[2], K[5]

    def image_callback(self, msg: Image):
        """Store the latest camera image."""
        self.last_image = msg

    def cloud_callback(self, cloud: PointCloud2):
        """Transform cloud to camera frame, project, draw, and publish image."""
        if self.cam_info is None or self.last_image is None:
            # Need both intrinsics and an image
            return

        # 1) Transform cloud into the camera frame
        try:
            # Time() with no args = "latest" in tf2
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,           # target
                cloud.header.frame_id,       # source
                Time(),                      # latest available transform
                timeout=Duration(seconds=0.1)
            )
            cloud_in_cam = tf2_sm.do_transform_cloud(cloud, transform)
        except TransformException as ex:
            self.get_logger().warn(f'TF transform failed: {ex}')
            return

        # 2) Convert base image to OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                self.last_image, desired_encoding='bgr8')
        except Exception as ex:
            self.get_logger().warn(f'cv_bridge conversion failed: {ex}')
            return

        height, width = cv_img.shape[:2]

        # 3) Project each 3D point and draw it
        for p in pc2.read_points(cloud_in_cam, skip_nans=True):
            X, Y, Z = float(p[0]), float(p[1]), float(p[2])

            if Z <= 0.0:
                continue

            u = int(self.fx * X / Z + self.cx)
            v = int(self.fy * Y / Z + self.cy)

            if 0 <= u < width and 0 <= v < height:
                # Draw a red pixel (or a small circle) for each laser point
                cv2.circle(cv_img, (u, v), 5, (0, 0, 255), -1)

        # 4) Publish the resulting image
        out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        out_msg.header = self.last_image.header  # keep original timestamp/frame
        self.image_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaserToImage()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

