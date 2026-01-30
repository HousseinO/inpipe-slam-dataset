#!/usr/bin/env python3
# ROS 2 Humble — Detect only PIPE vs ELBOW (adaptive, no meters)
# Method: voxel -> PCA ordering -> bin means centerline -> per-bin turn (deg)
#         elbow begins where cumulative turn exceeds threshold and
#         local median turn stays above a small angle for a few bins.

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import tf2_ros
import open3d as o3d

# ---------- helpers ----------
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

def np_to_pc2(points_xyz: np.ndarray, header: Header) -> PointCloud2:
    if points_xyz.size == 0:
        return pc2.create_cloud_xyz32(header, [])
    return pc2.create_cloud_xyz32(header, points_xyz.astype(np.float32).tolist())

def rolling_median(x, k):
    if k <= 1 or len(x) == 0: return x
    pad = k//2
    xp = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i+k])
    return out

# ---------- node ----------
class PipeVsElbowAdaptive(Node):
    def __init__(self):
        super().__init__('pipe_vs_elbow_adaptive')

        # Inputs / frames
        self.declare_parameter('cloud_topic', '/inpipe_bot/l515/l515_depth/points')
        self.declare_parameter('target_frame', 'base_link')   # '' to keep source frame

        # Speed / robustness
        self.declare_parameter('voxel', 0.015)                # 1.5 cm
        self.declare_parameter('num_bins', 15)  #15
        self.declare_parameter('min_bin_points', 20)
        self.declare_parameter('min_points_total', 400)

        # Adaptive elbow decision (bin-based, no meters)
        self.declare_parameter('cum_turn_deg', 10.0)          # total turning angle to call elbow
        self.declare_parameter('local_window_bins', 5)        # median window over per-bin turn
        self.declare_parameter('local_angle_thresh_deg', 3.0) # local turn threshold
        self.declare_parameter('persist_bins', 4)             # require this many consecutive bins
        self.declare_parameter('guard_bins', 2)               # small margin before elbow

        # Centerline marker
        self.declare_parameter('publish_centerline_marker', True)
        self.declare_parameter('centerline_width', 0.01)
        
        # Head crop (ignore near-sensor noisy part)
        self.declare_parameter('head_crop_mode', 'percent')  # 'percent' | 'bins' | 'meters'
        self.declare_parameter('head_crop_percent', 0.08)    # keep last 92% (default)
        self.declare_parameter('head_crop_bins', 6)          # if mode == 'bins'
        self.declare_parameter('head_crop_m', 0.30)          # if mode == 'meters' (along s)

        topic = self.get_parameter('cloud_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        # I/O
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, 1)
        self.pub_pipe  = self.create_publisher(PointCloud2, 'pipe/pipe', 1)
        self.pub_elbow = self.create_publisher(PointCloud2, 'pipe/elbow', 1)
        self.pub_centerline = self.create_publisher(Marker, 'pipe/elbow_centerline', 1)

        # TF
        self.tfbuf = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tflist = tf2_ros.TransformListener(self.tfbuf, self)

        self.get_logger().info(f"Listening on {topic}, target_frame={self.target_frame or '[source frame]'}")

    def cb(self, msg: PointCloud2):
        # 1) read cloud
        try:
            pts_src = pc2.read_points_numpy(msg, field_names=('x','y','z'), skip_nans=True)
        except Exception as e:
            self.get_logger().warn(f"read_points_numpy failed: {e}")
            return
        if pts_src.size == 0:
            return
        pts_src = pts_src[np.isfinite(pts_src).all(axis=1)]
        if pts_src.shape[0] < int(self.get_parameter('min_points_total').value):
            return

        # 2) transform to target frame
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
            pts = pts_src.astype(np.float64); out_frame = source

        # 3) voxel downsample
        voxel = float(self.get_parameter('voxel').value)
        pts_ds = voxel_downsample_xyz(pts, voxel)
        if pts_ds.shape[0] < 300:
            return

        # 4) PCA for ordering
        u0, c0 = pca_axis(pts_ds)
        s = (pts_ds - c0) @ u0
        smin, smax = float(s.min()), float(s.max())
        if smax - smin < 1e-3:
            return

        # 5) binning + centerline
        num_bins = int(self.get_parameter('num_bins').value)
        edges = np.linspace(smin, smax, num_bins + 1)
        
        # ---------- HEAD CROP: drop the first part along s ----------
        mode = self.get_parameter('head_crop_mode').value
        keep_mask = np.ones(len(s), dtype=bool)

        if mode == 'percent':
            frac = float(self.get_parameter('head_crop_percent').value)
            frac = float(np.clip(frac, 0.0, 0.9))  # avoid removing everything
            s_cut = smin + frac * (smax - smin)
            keep_mask &= (s >= s_cut)

        elif mode == 'bins':
            n_drop = int(self.get_parameter('head_crop_bins').value)
            n_drop = max(0, min(n_drop, num_bins - 2))
            if n_drop > 0:
                s_cut = edges[n_drop]             # boundary after the dropped bins
                keep_mask &= (s >= s_cut)

        elif mode == 'meters':
            L = float(self.get_parameter('head_crop_m').value)
            s_cut = smin + L                      # drop first L meters *along s*
            keep_mask &= (s >= s_cut)

        # Apply crop
        pts_ds = pts_ds[keep_mask]
        s = s[keep_mask]
        if pts_ds.shape[0] < 300:
            return

        # Recompute ranges/edges after crop
        smin, smax = float(s.min()), float(s.max())
        if smax - smin < 1e-3: return
        edges = np.linspace(smin, smax, num_bins + 1)



        bin_ids = np.clip(np.digitize(s, edges) - 1, 0, num_bins - 1)

        counts = np.bincount(bin_ids, minlength=num_bins)
        valid = counts >= int(self.get_parameter('min_bin_points').value)
        if valid.sum() < 5:
            return

        sum_x = np.bincount(bin_ids, weights=pts_ds[:,0], minlength=num_bins)
        sum_y = np.bincount(bin_ids, weights=pts_ds[:,1], minlength=num_bins)
        sum_z = np.bincount(bin_ids, weights=pts_ds[:,2], minlength=num_bins)
        C = np.vstack([sum_x[valid]/counts[valid],
                       sum_y[valid]/counts[valid],
                       sum_z[valid]/counts[valid]]).T

        # 6) per-bin turning angles (deg) along the centerline
        t = np.gradient(C, axis=0)
        tn = t / np.linalg.norm(t, axis=1, keepdims=True).clip(min=1e-9)
        # angle between successive tangents:
        dots = np.sum(tn[:-1] * tn[1:], axis=1).clip(-1.0, 1.0)
        turn_deg = np.degrees(np.arccos(dots))  # length N-1

        # 7) adaptive elbow start by cumulative + local persistence
        local_k = int(self.get_parameter('local_window_bins').value)
        turn_med = rolling_median(turn_deg, max(3, local_k))          # robust local turn
        cum_turn = np.cumsum(turn_med)                                 # total bend from start
        cum_thresh = float(self.get_parameter('cum_turn_deg').value)
        local_thresh = float(self.get_parameter('local_angle_thresh_deg').value)
        persist_bins = int(self.get_parameter('persist_bins').value)

        elbow_start_idx = None
        i = 0
        while i < len(turn_med):
            if turn_med[i] >= local_thresh and cum_turn[i] >= cum_thresh:
                # ensure it persists
                j = min(i + persist_bins, len(turn_med))
                if np.all(turn_med[i:j] >= local_thresh):
                    elbow_start_idx = i
                    break
                i = j
            else:
                i += 1

        guard_bins = int(self.get_parameter('guard_bins').value)
        valid_bins = np.nonzero(valid)[0]

        if elbow_start_idx is None:
            # everything is straight pipe
            pipe_mask = np.ones_like(bin_ids+1, dtype=bool)
            elbow_mask = ~pipe_mask
        else:
            # map local centerline-change index to global bin index
            # turn_deg has length (len(C)-1), its i aligns with segment between C[i] and C[i+1]
            elbow_start_idx=elbow_start_idx
#            start_bin = max(0, valid_bins[min(elbow_start_idx, len(valid_bins)-1)] - guard_bins)
#            pipe_mask  = bin_ids < start_bin
#            elbow_mask = bin_ids >= start_bin
            start_bin_mark = valid_bins[min(elbow_start_idx + 1, len(valid_bins)-1)]
            pipe_mask  = bin_ids < start_bin_mark
            elbow_mask = bin_ids >= start_bin_mark

        pipe_pts  = pts_ds[pipe_mask]
        elbow_pts = pts_ds[elbow_mask]

        # 8) publish
        hdr = Header(); hdr.stamp = msg.header.stamp; hdr.frame_id = out_frame
        self.pub_pipe.publish(np_to_pc2(pipe_pts, hdr))
        self.pub_elbow.publish(np_to_pc2(elbow_pts, hdr))

        # 9) centerline marker for tuning/visualization
        if self.get_parameter('publish_centerline_marker').value and C.shape[0] >= 2:
            # whole centerline (blue)
            m = Marker()
            m.header.frame_id = out_frame; m.header.stamp = msg.header.stamp
            m.ns = 'elbow_centerline'; m.id = 0
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = float(self.get_parameter('centerline_width').value)
            m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in C]
            self.pub_centerline.publish(m)

            if elbow_start_idx is not None:
                me = Marker()
                me.header.frame_id = out_frame; me.header.stamp = msg.header.stamp
                me.ns = 'elbow_centerline'; me.id = 1
                me.type = Marker.LINE_STRIP; me.action = Marker.ADD
                me.scale.x = float(self.get_parameter('centerline_width').value)
                me.color = ColorRGBA(r=0.95, g=0.45, b=0.1, a=0.95)
                me.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in C[elbow_start_idx+1:]]
                self.pub_centerline.publish(me)

        # log
        if elbow_start_idx is None:
            self.get_logger().info(f"[PIPE ONLY] pipe={pipe_pts.shape[0]} pts | cum_turn_end={cum_turn[-1]:.1f}°")
        else:
            self.get_logger().info(
                f"[PIPE+ELBOW] start@bin≈{valid_bins[min(elbow_start_idx+1,len(valid_bins)-1)]} | "
                f"pipe={pipe_pts.shape[0]} elbow={elbow_pts.shape[0]} | cum_turn@start≈{cum_turn[elbow_start_idx]:.1f}°"
            )

def main():
    rclpy.init()
    node = PipeVsElbowAdaptive()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

