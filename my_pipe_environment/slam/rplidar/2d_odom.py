#!/usr/bin/env python3
"""
ROS2 (Humble) minimal scan-to-scan matcher for 2D LiDAR point clouds.
- Subscribes:  sensor_msgs/PointCloud2 on /inpipe_bot/rplidar/points (configurable)
- Performs:    lightweight 2D ICP (point-to-point) between consecutive scans
- Publishes:   nav_msgs/Odometry on /odom
               nav_msgs/Path     on /scanpath
               TF: odom -> base_link

Notes:
- This is a small, dependency-free (no Open3D/Scipy) ICP suitable for small motions.
- It down-samples aggressively and assumes the motion between frames is small.
- For in-pipe use, this is a starting point; later you can replace the matcher with
  correlative matching + IMU fusion.
"""

from typing import Optional, Tuple
import math
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
    
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped

import tf_transformations
from tf2_ros import TransformBroadcaster


# ---------------------------- Utility math ---------------------------------

def se2_compose(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """Compose two SE(2) transforms (3x3)."""
    return Ta @ Tb


def vec2_to_se2(x: float, y: float, th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    T = np.array([[c, -s, x],
                  [s,  c, y],
                  [0., 0., 1.]])
    return T


def se2_to_pose(T: np.ndarray) -> Tuple[float, float, float]:
    x, y = T[0, 2], T[1, 2]
    th = math.atan2(T[1, 0], T[0, 0])
    return x, y, th
    
def voxel_downsample_xy(points: np.ndarray, voxel: float) -> np.ndarray:
    """Down-sample by voxel grid in XY (ignore Z)."""
    if len(points) == 0 or voxel <= 0:
        return points
    # integer voxel indices in XY
    idx = np.floor(points[:, :2] / voxel).astype(np.int64)
    # unique key per voxel
    keys = idx[:, 0] * 73856093 ^ idx[:, 1] * 19349663
    # keep first point per voxel
    _, first_indices = np.unique(keys, return_index=True)
    return points[first_indices]


# ----------------------------- Point-to-LINE ICP 2D --------------------------

def compute_normals_2d(points: np.ndarray, k: int = 8) -> np.ndarray:
    """Estimate 2D normals via local PCA on XY neighbors.
    Returns unit normals of shape (N,2)."""
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    P = points[:, :2]
    N = P.shape[0]
    normals = np.zeros((N, 2), dtype=np.float32)
    # Brute-force kNN (fine after voxel downsample)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    # large value on diagonal to exclude self
    np.fill_diagonal(d2, np.inf)
    for i in range(N):
        idx = np.argpartition(d2[i], k)[:k]
        nbrs = P[idx]
        c = nbrs.mean(axis=0)
        X = nbrs - c
        # 2x2 covariance
        C = X.T @ X / max(len(idx) - 1, 1)
        # eigenvectors of covariance: smallest eigenvalue -> normal
        w, v = np.linalg.eigh(C)
        n = v[:, 0]  # eigenvector for smallest eigenvalue
        n = n / (np.linalg.norm(n) + 1e-9)
        normals[i] = n
    return normals


def p2l_icp_2d(source: np.ndarray, target: np.ndarray, iters: int = 20, max_corr: float = 0.12) -> Tuple[np.ndarray, float]:
    """Point-to-line ICP in 2D (Gauss-Newton). Returns delta T (3x3) mapping source->target."""
    if len(source) < 5 or len(target) < 5:
        return np.eye(3), float('inf')

    # Precompute normals on TARGET
    tgt_normals = compute_normals_2d(target)

    # Precompute brute-force distance matrix for correspondences
    Axy = source[:, :2]
    Bxy = target[:, :2]
    T = np.eye(3)
    R = T[:2, :2]
    t = T[:2, 2]

    for _ in range(iters):
        # Transform source by current estimate
        S = (Axy @ R.T) + t
        # nearest neighbors in TARGET
        d2 = ((S[:, None, :] - Bxy[None, :, :]) ** 2).sum(axis=2)
        j = np.argmin(d2, axis=1)
        nn = Bxy[j]
        n = tgt_normals[j]
        d = np.sqrt(d2[np.arange(len(S)), j])
        mask = d < max_corr
        if mask.sum() < 5:
            break
        S = S[mask]
        A_sel = Axy[mask]
        nn = nn[mask]
        n = n[mask]

        # Build linear system J^T J delta = -J^T r
        # residual r = n^T (R s + t - q)
        # Jacobian wrt [dx, dy, dtheta]:
        # J = [n_x, n_y, n^T * (R * S * s)] where S = [[0,-1],[1,0]]
        Smat = np.array([[0.0, -1.0], [1.0, 0.0]])
        v = (R @ (Smat @ A_sel.T)).T  # shape (m,2)
        J = np.stack([n[:, 0], n[:, 1], (n * v).sum(axis=1)], axis=1)
        r = (n * (S - nn)).sum(axis=1)
        H = J.T @ J
        b = - J.T @ r
        try:
            delta = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break
        dx, dy, dth = delta.tolist()
        # Update R,t
        c, s = math.cos(dth), math.sin(dth)
        dR = np.array([[c, -s], [s, c]])
        R = dR @ R
        t = dR @ t + np.array([dx, dy])
        if np.linalg.norm(delta) < 1e-5:
            break

    T[:2, :2] = R
    T[:2, 2] = t
    # final mean orthogonal error
    S_final = (Axy @ R.T) + t
    d2 = ((S_final[:, None, :] - Bxy[None, :, :]) ** 2).sum(axis=2)
    j = np.argmin(d2, axis=1)
    n = tgt_normals[j]
    nn = Bxy[j]
    err = float(np.mean(np.abs((n * (S_final - nn)).sum(axis=1))))
    return T, err


# ----------------------------- ROS2 Node ------------------------------------
class ScanMatcherNode(Node):
    def __init__(self):
        super().__init__('scan_matcher_node')

        # Parameters
        self.declare_parameter('cloud_topic', '/inpipe_bot/rplidar/points')
        self.declare_parameter('voxel', 0.03)              # meters
        self.declare_parameter('max_corr_dist', 0.12)      # meters
        self.declare_parameter('max_icp_iters', 25)
        self.declare_parameter('frame_odom', 'odom_rplidar')
        self.declare_parameter('frame_base', 'base_link')

        self.cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.voxel = self.get_parameter('voxel').get_parameter_value().double_value
        self.max_corr = self.get_parameter('max_corr_dist').get_parameter_value().double_value
        self.max_icp_iters = int(self.get_parameter('max_icp_iters').get_parameter_value().integer_value)
        self.frame_odom = self.get_parameter('frame_odom').get_parameter_value().string_value
        self.frame_base = self.get_parameter('frame_base').get_parameter_value().string_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self.cloud_cb, qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom/rplidar', 10)
        self.path_pub = self.create_publisher(Path, '/scanpath', 10)
        self.tf_br = TransformBroadcaster(self)

        self.prev_pts: Optional[np.ndarray] = None
        self.global_T = np.eye(3)  # odom->base_link
        self.path = Path()
        self.path.header.frame_id = self.frame_odom

        self.get_logger().info(f"Listening to {self.cloud_topic}")

    # ---- PointCloud2 -> Nx3 float32 ----
    def cloud_to_points(self, msg: PointCloud2) -> np.ndarray:
        pts = []
        for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([x, y, z])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        arr = np.asarray(pts, dtype=np.float32)
        # Filter far/NaN (already skipped); enforce finite
        arr = arr[np.isfinite(arr).all(axis=1)]
        return arr

    def publish_tf_and_msgs(self, stamp):
        x, y, th = se2_to_pose(self.global_T)
        # TF odom -> base_link
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.frame_odom
        t.child_frame_id = self.frame_base
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(th))
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
#        self.tf_br.sendTransform(t)

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_odom
        odom.child_frame_id = self.frame_base
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        self.odom_pub.publish(odom)

        # Path
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.frame_odom
        ps.pose = odom.pose.pose
#        self.path.poses.append(ps)
#        self.path.header.stamp = stamp
#        self.path_pub.publish(self.path)

    def cloud_cb(self, msg: PointCloud2):
        # Convert & downsample
        pts = self.cloud_to_points(msg)
        if len(pts) == 0:
            return
        pts = voxel_downsample_xy(pts, self.voxel)

        if self.prev_pts is None:
            self.prev_pts = pts
            self.publish_tf_and_msgs(msg.header.stamp)
            return

                # Run point-to-LINE ICP (scan-to-scan)
        dT, err = p2l_icp_2d(pts.copy(), self.prev_pts, iters=self.max_icp_iters, max_corr=self.max_corr)

        # Compose into global pose (odom->base_link)
        self.global_T = se2_compose(self.global_T, dT)

        # Update previous cloud
        self.prev_pts = pts

        # Publish odom, path, and TF
        self.publish_tf_and_msgs(msg.header.stamp)

        # Debug
        if math.isfinite(err):
            self.get_logger().debug(f"ICP err={err:.4f}")


def main():
    rclpy.init()
    node = ScanMatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

