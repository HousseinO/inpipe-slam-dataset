#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np


class PointCloudNoiseAndDensityNode(Node):
    def __init__(self):
        super().__init__('pointcloud_noise_and_density_node')

        # Parameters
        self.declare_parameter('input_topic', '/inpipe_bot/l515/l515_depth/points')
        self.declare_parameter('output_topic', '/inpipe_bot/l515/l515_depth/points_noisy')
        self.declare_parameter('noise_stddev', 0.01)     # meters
        self.declare_parameter('nan_prob', 0.02)          # 2% NaN dropout
        self.declare_parameter('keep_ratio', 0.1)         # keep 20% of points

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(PointCloud2, input_topic, self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

        self.get_logger().info(
            f"Subscribed to {input_topic}, publishing noisy/diluted cloud to {output_topic}\n"
            f"noise_stddev={self.get_parameter('noise_stddev').value}, "
            f"nan_prob={self.get_parameter('nan_prob').value}, "
            f"keep_ratio={self.get_parameter('keep_ratio').value}"
        )

    def callback(self, msg: PointCloud2):
        try:
            pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=False)
        except Exception as e:
            self.get_logger().warn(f"Failed to read cloud: {e}")
            return

        N = pts.shape[0]
        mask_valid = np.isfinite(pts).all(axis=1)
        num_valid = mask_valid.sum()
        if num_valid == 0:
            self.get_logger().warn("No valid points in cloud")
            return

        pts_noisy = pts.copy()

        # --- Gaussian noise ---
        sigma = float(self.get_parameter('noise_stddev').value)
        noise = np.random.normal(0.0, sigma, pts_noisy.shape)
        pts_noisy[mask_valid] += noise[mask_valid]

        # --- Random NaN dropout ---
        nan_prob = float(self.get_parameter('nan_prob').value)
        dropout_mask = (np.random.rand(N) < nan_prob)
        pts_noisy[dropout_mask] = np.nan

        # --- Random density reduction ---
        keep_ratio = float(self.get_parameter('keep_ratio').value)
        # Keep only a random subset of non-NaN points
        valid_idx = np.where(np.isfinite(pts_noisy).all(axis=1))[0]
        num_keep = int(len(valid_idx) * keep_ratio)
        if num_keep > 0:
            keep_idx = np.random.choice(valid_idx, size=num_keep, replace=False)
            new_pts = pts_noisy[keep_idx]
        else:
            new_pts = pts_noisy[np.isfinite(pts_noisy).all(axis=1)]  # fallback

        # --- Publish noisy & downsampled cloud ---
        noisy_msg = pc2.create_cloud_xyz32(msg.header, new_pts)
        noisy_msg.header.stamp = msg.header.stamp
        noisy_msg.header.frame_id = msg.header.frame_id
        self.pub.publish(noisy_msg)

        self.get_logger().debug(
            f"Input {N} pts → valid {num_valid} → kept {new_pts.shape[0]} (dropout {dropout_mask.sum()})"
        )


def main():
    rclpy.init()
    node = PointCloudNoiseAndDensityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

