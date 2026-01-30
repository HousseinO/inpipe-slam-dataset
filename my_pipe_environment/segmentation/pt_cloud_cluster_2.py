#!/usr/bin/env python3
# pipe_elbow_state.py — ROS 2 Humble
# Real-time elbow/pipe segmentation + 3-class state:
#   ELBOW_PIPE, PIPE_ELBOW, PIPE_OPEN

import math
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros
import open3d as o3d

# ----------------- helpers (from your fast estimator) -----------------

def tf_to_matrix(tf):
    t = tf.transform.translation; q = tf.transform.rotation
    x,y,z,w = q.x,q.y,q.z,q.w
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64); T[:3,:3]=R; T[:3,3]=[t.x,t.y,t.z]; return T

def voxel_downsample_xyz(xyz: np.ndarray, voxel=0.015) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points)

def pca_axis(xyz: np.ndarray):
    ctr = xyz.mean(axis=0)
    X = xyz - ctr
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axis = Vt[0, :] / np.linalg.norm(Vt[0, :])
    return axis, ctr

def distance_points_to_line(xyz: np.ndarray, p0: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(xyz - p0, v), axis=1)

def yaw_pitch_from_dir(v: np.ndarray):
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw = math.atan2(vy, vx)
    pitch = math.atan2(-vz, math.hypot(vx, vy))
    return math.degrees(yaw), math.degrees(pitch)

def np_to_pc2(points_xyz: np.ndarray, header: Header) -> PointCloud2:
    if points_xyz.size == 0:
        return pc2.create_cloud_xyz32(header, [])
    return pc2.create_cloud_xyz32(header, points_xyz.astype(np.float32).tolist())

def smooth_centerline(C: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1 or len(C) < 3:
        return C
    pad = k // 2
    Cp = np.pad(C, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones((k, 1)) / k
    return (np.convolve(Cp[:, 0], kernel[:, 0], 'valid'),
            np.convolve(Cp[:, 1], kernel[:, 0], 'valid'),
            np.convolve(Cp[:, 2], kernel[:, 0], 'valid'))

# ----------------- node -----------------

class PipeElbowState(Node):
    def __init__(self):
        super().__init__('pipe_elbow_state')

        # I/O
        self.declare_parameter('cloud_topic', '/inpipe_bot/l515/l515_depth/points')
        self.declare_parameter('target_frame', 'base_link')   # '' to keep source frame

        # Speed/robustness
        self.declare_parameter('voxel', 0.015)                # 1.5 cm
        self.declare_parameter('num_bins', 80)
        self.declare_parameter('min_bin_points', 20)
        self.declare_parameter('centerline_smooth_bins', 5)

        # Elbow + straightness logic
        self.declare_parameter('elbow_band_bins', 3)          # how many bins around elbow center belong to "elbow"
        self.declare_parameter('straight_curv_deg_thresh', 5.0)
        self.declare_parameter('straight_min_bins', 4)
        self.declare_parameter('min_points_total', 400)

        # Classification proximity (how close the robot must be to call PIPE_ELBOW)
        self.declare_parameter('proximity_bins', 5)           # bin-distance threshold to elbow for PIPE_ELBOW

        topic = self.get_parameter('cloud_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        self.sub = self.create_subscription(PointCloud2, topic, self.cb, 1)
        self.pub_straight1 = self.create_publisher(PointCloud2, 'pipe/straight1', 1)
        self.pub_straight2 = self.create_publisher(PointCloud2, 'pipe/straight2', 1)
        self.pub_elbow     = self.create_publisher(PointCloud2, 'pipe/elbow', 1)
        self.pub_pipe_part = self.create_publisher(PointCloud2, 'pipe/pipe_part', 1)
        self.pub_state     = self.create_publisher(String, 'pipe/state', 1)

        self.tfbuf = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tflist = tf2_ros.TransformListener(self.tfbuf, self)

        self.get_logger().info(f"Listening on {topic}, target_frame={self.target_frame or '[source frame]'}")

    def cb(self, msg: PointCloud2):
        # ---- 1) Read + filter ----
        try:
            pts_src = pc2.read_points_numpy(msg, field_names=('x','y','z'), skip_nans=True)
        except Exception as e:
            self.get_logger().warn(f"read_points_numpy failed: {e}")
            return
        if pts_src.size == 0:
            return
        mask = np.isfinite(pts_src).all(axis=1)
        pts_src = pts_src[mask]
        if pts_src.shape[0] < self.get_parameter('min_points_total').value:
            return

        # ---- 2) Transform to target frame ----
        source = msg.header.frame_id
        target = self.target_frame or source
        if target != source:
            try:
                tf = self.tfbuf.lookup_transform(target, source, rclpy.time.Time(), timeout=Duration(seconds=0.2))
            except Exception as e:
                self.get_logger().warn(f"TF lookup {target} <- {source} failed: {e}")
                return
            T = tf_to_matrix(tf)
            pts = (T @ np.c_[pts_src.astype(np.float64), np.ones((pts_src.shape[0],1))].T).T[:, :3]
            out_frame = target
        else:
            pts = pts_src.astype(np.float64)
            out_frame = source

        # ---- 3) Voxel downsample ----
        voxel = float(self.get_parameter('voxel').value)
        pts_ds = voxel_downsample_xyz(pts, voxel=voxel)
        if pts_ds.shape[0] < 300:
            return

        # ---- 4) Global axis + binning ----
        u0, c0 = pca_axis(pts_ds)
        s = (pts_ds - c0) @ u0                                # scalar position along axis
        smin, smax = float(s.min()), float(s.max())
        if smax - smin < 1e-3:
            return

        num_bins = int(self.get_parameter('num_bins').value)
        edges = np.linspace(smin, smax, num_bins + 1)
        bin_ids = np.clip(np.digitize(s, edges) - 1, 0, num_bins - 1)

        counts = np.bincount(bin_ids, minlength=num_bins)
        valid = counts >= int(self.get_parameter('min_bin_points').value)
        if valid.sum() < 5:
            return

        # Coarse centerline (bin means)
        sum_x = np.bincount(bin_ids, weights=pts_ds[:,0], minlength=num_bins)
        sum_y = np.bincount(bin_ids, weights=pts_ds[:,1], minlength=num_bins)
        sum_z = np.bincount(bin_ids, weights=pts_ds[:,2], minlength=num_bins)
        C = np.vstack([sum_x[valid]/counts[valid],
                       sum_y[valid]/counts[valid],
                       sum_z[valid]/counts[valid]]).T

        # Optional smoothing
        k = int(self.get_parameter('centerline_smooth_bins').value)
        if k > 1:
            cx, cy, cz = smooth_centerline(C, k)
            C = np.stack([cx, cy, cz], axis=1)

        # ---- 5) Weighted curvature -> elbow bin ----
        t = np.gradient(C, axis=0)
        tn = t / np.linalg.norm(t, axis=1, keepdims=True).clip(min=1e-9)
        dots = np.sum(tn[:-1] * tn[1:], axis=1).clip(-1.0, 1.0)
        ang = np.arccos(dots)                                 # radians between successive tangents
        seg_len = np.linalg.norm(np.diff(C, axis=0), axis=1)
        w_ang = ang * seg_len
        if w_ang.size == 0:
            return

        m0, m1 = int(0.2*len(w_ang)), int(0.8*len(w_ang))
        rel_idx = np.argmax(w_ang[m0:m1])
        k_idx = m0 + int(rel_idx)

        valid_bins = np.nonzero(valid)[0]
        if len(valid_bins) < 2:
            return
        elbow_bin = int(0.5*(valid_bins[k_idx] + valid_bins[k_idx+1])) if k_idx+1 < len(valid_bins) else valid_bins[k_idx]

        # ---- 6) Assign bins to straight1 / elbow / straight2 ----
        band = int(self.get_parameter('elbow_band_bins').value)
        left_bins  = np.arange(0, elbow_bin - band + 1)
        mid_bins   = np.arange(max(0, elbow_bin - band + 1), min(num_bins, elbow_bin + band + 1))
        right_bins = np.arange(elbow_bin + band + 1, num_bins)

        left_mask  = np.isin(bin_ids, left_bins)
        mid_mask   = np.isin(bin_ids, mid_bins)
        right_mask = np.isin(bin_ids, right_bins)

        A_pts = pts_ds[left_mask]
        elbow_pts = pts_ds[mid_mask]
        B_pts = pts_ds[right_mask]

        # ---- 7) Decide whether a side is really "straight"; else merge to elbow ----
        def side_avg_deg(bin_range):
            if len(bin_range) < 2:
                return np.inf
            idxs = np.intersect1d(valid_bins[:-1], bin_range, assume_unique=False)
            if idxs.size == 0:
                return np.inf
            loc = np.searchsorted(valid_bins[:-1], idxs)
            deg = np.degrees(np.mean(ang[loc])) if loc.size > 0 else np.inf
            return deg

        curv_thresh_deg = float(self.get_parameter('straight_curv_deg_thresh').value)
        min_bins = int(self.get_parameter('straight_min_bins').value)

        left_deg  = side_avg_deg(left_bins)
        right_deg = side_avg_deg(right_bins)

        if right_deg > curv_thresh_deg or np.sum(right_mask) < self.get_parameter('min_bin_points').value * min_bins:
            elbow_pts = np.vstack([elbow_pts, B_pts]) if elbow_pts.size else B_pts
            B_pts = np.empty((0, 3))
        if left_deg > curv_thresh_deg or np.sum(left_mask) < self.get_parameter('min_bin_points').value * min_bins:
            elbow_pts = np.vstack([elbow_pts, A_pts]) if elbow_pts.size else A_pts
            A_pts = np.empty((0, 3))

        # ---- 8) Pipe geometry meta (optional but handy in logs) ----
        radius = np.nan
        elbow_angle_deg = np.nan
        report = []

        # A
        if A_pts.size:
            a_dir, a_ctr = pca_axis(A_pts)
            rad_a = float(np.median(distance_points_to_line(A_pts, a_ctr, a_dir)))
            yaw_a, pitch_a = yaw_pitch_from_dir(a_dir / np.linalg.norm(a_dir))
            report.append(f"A: r≈{rad_a:.3f}m yaw={yaw_a:.1f}° pitch={pitch_a:.1f}°")
            radius = rad_a if not np.isfinite(radius) else 0.5*(radius + rad_a)
        else:
            a_dir = None

        # B
        if B_pts.size:
            b_dir, b_ctr = pca_axis(B_pts)
            rad_b = float(np.median(distance_points_to_line(B_pts, b_ctr, b_dir)))
            yaw_b, pitch_b = yaw_pitch_from_dir(b_dir / np.linalg.norm(b_dir))
            report.append(f"B: r≈{rad_b:.3f}m yaw={yaw_b:.1f}° pitch={pitch_b:.1f}°")
            radius = rad_b if not np.isfinite(radius) else 0.5*(radius + rad_b)
        else:
            b_dir = None

        if a_dir is not None and b_dir is not None:
            u = a_dir / np.linalg.norm(a_dir)
            v = b_dir / np.linalg.norm(b_dir)
            if np.dot(u, v) > 0:
                v = -v
            elbow_angle_deg = math.degrees(math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0))))

        if np.isfinite(radius):
            report.insert(0, f"radius≈{radius:.3f}m")
        if np.isfinite(elbow_angle_deg):
            report.insert(1, f"elbow≈{elbow_angle_deg:.1f}°")

        self.get_logger().info("[FAST+STATE] " + " | ".join(report) if report else "[FAST+STATE] no straight segments detected")

        # ---- 9) Classification (ELBOW_PIPE / PIPE_ELBOW / PIPE_OPEN) ----
        # Map robot origin (0,0,0) in target frame onto axis bins
        s0 = float((np.zeros(3) - c0) @ u0)                    # origin projection
        # Place it in bin space of all points (not only valid), to compare with elbow_bin
        s0_bin = int(np.clip(np.digitize(s0, edges) - 1, 0, num_bins - 1))

        elbow_exists = elbow_pts.size > 0
        prox_bins = int(self.get_parameter('proximity_bins').value)

        in_elbow_band = s0_bin in mid_bins.tolist()
        near_elbow = (abs(s0_bin - elbow_bin) <= prox_bins)

        if in_elbow_band and elbow_exists:
            state = "ELBOW_PIPE"
        elif elbow_exists and near_elbow:
            state = "PIPE_ELBOW"
        else:
            state = "PIPE_OPEN"

        # Decide which straight side the robot is in, for publishing pipe_part
        if s0_bin in left_bins.tolist():
            pipe_part_pts = A_pts
        elif s0_bin in right_bins.tolist():
            pipe_part_pts = B_pts
        else:
            # inside elbow -> pick the nearer straight side (by bin distance)
            dL = abs(s0_bin - (left_bins[-1] if len(left_bins) else 0))
            dR = abs(s0_bin - (right_bins[0] if len(right_bins) else num_bins-1))
            pipe_part_pts = A_pts if dL <= dR else B_pts

        # ---- 10) Publish clouds ----
        hdr = Header(); hdr.stamp = msg.header.stamp; hdr.frame_id = out_frame
        self.pub_straight1.publish(np_to_pc2(A_pts, hdr))
        self.pub_straight2.publish(np_to_pc2(B_pts, hdr))
        self.pub_elbow.publish(np_to_pc2(elbow_pts, hdr))
        self.pub_pipe_part.publish(np_to_pc2(pipe_part_pts, hdr))

        # ---- 11) Publish state ----
        payload = {
            "state": state,
            "elbow_exists": bool(elbow_exists),
            "s0_bin": int(s0_bin),
            "elbow_bin": int(elbow_bin),
            "proximity_bins": prox_bins,
            "elbow_band_bins": band,
            "num_bins": num_bins,
            "radius_m": float(radius) if np.isfinite(radius) else None,
            "elbow_angle_deg": float(elbow_angle_deg) if np.isfinite(elbow_angle_deg) else None
        }
        msg_state = String()
        msg_state.data = json.dumps(payload, separators=(',', ':'))
        self.pub_state.publish(msg_state)

def main():
    rclpy.init()
    node = PipeElbowState()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

