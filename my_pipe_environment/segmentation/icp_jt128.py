#!/usr/bin/env python3
"""
hesai_icp_odom_debug.py
ICP odometry for Hesai PointCloud2 WITH VERY VERBOSE LOGGING.

What this version adds:
- Logs at every stage (RX, TF, filtering counts, voxel/cap counts, ICP stats, publish confirmation)
- Parameter dump on start
- Optional periodic TF availability check
- Uses pc2.read_points() so mixed datatype clouds (ring/timestamp) work

Run example:
python3 hesai_icp_odom_debug.py --ros-args \
  -p cloud_topic:=/hesai/pandar \
  -p target_frame:=hesai_lidar \
  -p odom_frame:=odom \
  -p base_frame:=hesai_lidar \
  -p z_min:=0.20 -p z_max:=5.0 \
  -p min_range:=0.5 -p max_range:=25.0 \
  -p voxel:=0.10 -p max_points:=40000 \
  -p icp_max_corr_dist:=0.25 -p icp_max_iter:=60 -p min_fitness:=0.25
"""

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import open3d as o3d


# ---------- SE3 helpers ----------
def quat_from_R(R: np.ndarray):
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s
    return float(x), float(y), float(z), float(w)


def odom_msg_from_T(T: np.ndarray, frame_id: str, child_frame_id: str, stamp):
    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = frame_id
    odom.child_frame_id = child_frame_id

    R = T[:3, :3]
    t = T[:3, 3]
    qx, qy, qz, qw = quat_from_R(R)

    odom.pose.pose.position.x = float(t[0])
    odom.pose.pose.position.y = float(t[1])
    odom.pose.pose.position.z = float(t[2])
    odom.pose.pose.orientation.x = qx
    odom.pose.pose.orientation.y = qy
    odom.pose.pose.orientation.z = qz
    odom.pose.pose.orientation.w = qw
    return odom


def tf_from_T(T: np.ndarray, frame_id: str, child_frame_id: str, stamp):
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id

    R = T[:3, :3]
    t = T[:3, 3]
    qx, qy, qz, qw = quat_from_R(R)

    tf.transform.translation.x = float(t[0])
    tf.transform.translation.y = float(t[1])
    tf.transform.translation.z = float(t[2])
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


def tf_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class HesaiICPOdomDebug(Node):
    def __init__(self):
        super().__init__("hesai_icp_odom")

        # ---- params ----
        self.declare_parameter("cloud_topic", "/hesai/pandar")
        self.declare_parameter("target_frame", "hesai_lidar")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "hesai_lidar")

        # filters
        self.declare_parameter("min_range", 0.5)
        self.declare_parameter("max_range", 25.0)
        self.declare_parameter("z_min", -100.0)
        self.declare_parameter("z_max", 100.0)

        # downsample / cap
        self.declare_parameter("voxel", 0.10)
        self.declare_parameter("max_points", 40000)

        # ICP
        self.declare_parameter("icp_max_corr_dist", 0.25)
        self.declare_parameter("icp_max_iter", 60)
        self.declare_parameter("min_fitness", 0.25)
        self.declare_parameter("use_point_to_plane", True)

        # logging controls
        self.declare_parameter("log_every_n_frames", 1)  # set 10 to reduce spam
        self.declare_parameter("tf_timeout_sec", 0.2)

        # QoS: keep RELIABLE by default (matches your rosbag), but explicit profile
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        topic = self.get_parameter("cloud_topic").value
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, qos)

        self.pub_odom = self.create_publisher(Odometry, "pipe/odom", 10)
        self.pub_path = self.create_publisher(Path, "pipe/path", 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.get_parameter("odom_frame").value

        self.tfbuf = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tflist = tf2_ros.TransformListener(self.tfbuf, self)
        self.tfb = tf2_ros.TransformBroadcaster(self)

        self.prev_pcd = None
        self.T_odom_base = np.eye(4, dtype=np.float64)
        self.T_last_delta = np.eye(4, dtype=np.float64)

        self.frame_count = 0

        self._log_params()
        self.get_logger().info(f"SUBSCRIBED topic={topic}")

    def _log_params(self):
        p = lambda k: self.get_parameter(k).value
        self.get_logger().info(
            "PARAMS: "
            f"cloud_topic={p('cloud_topic')} "
            f"target_frame={p('target_frame')} "
            f"odom_frame={p('odom_frame')} "
            f"base_frame={p('base_frame')} | "
            f"min_range={p('min_range')} max_range={p('max_range')} "
            f"z_min={p('z_min')} z_max={p('z_max')} | "
            f"voxel={p('voxel')} max_points={p('max_points')} | "
            f"icp_max_corr_dist={p('icp_max_corr_dist')} icp_max_iter={p('icp_max_iter')} "
            f"min_fitness={p('min_fitness')} use_point_to_plane={p('use_point_to_plane')} | "
            f"log_every_n_frames={p('log_every_n_frames')} tf_timeout_sec={p('tf_timeout_sec')}"
        )

    def read_xyz(self, msg: PointCloud2) -> np.ndarray:
        it = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        arr = np.fromiter((v for p in it for v in p), dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return arr.reshape(-1, 3)

    def preprocess(self, pts: np.ndarray):
        """Return filtered pts and a dict of stats."""
        stats = {}
        stats["in"] = int(pts.shape[0])

        # range on XY
        min_r = float(self.get_parameter("min_range").value)
        max_r = float(self.get_parameter("max_range").value)
        r = np.linalg.norm(pts, axis=1)   # sqrt(x^2 + y^2 + z^2)

        m_range = (r >= min_r) & (r <= max_r)
        stats["keep_range"] = int(m_range.sum())

        zmin = float(self.get_parameter("z_min").value)
        zmax = float(self.get_parameter("z_max").value)
        m_z = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        stats["keep_z"] = int(m_z.sum())

        m = m_range & m_z
        stats["keep_both"] = int(m.sum())

        out = pts[m]
        stats["out"] = int(out.shape[0])

        # quick distribution (helpful to see axis issues)
        if stats["in"] > 0:
            stats["z_min_obs"] = float(np.min(pts[:, 2]))
            stats["z_max_obs"] = float(np.max(pts[:, 2]))
            stats["z_mean_obs"] = float(np.mean(pts[:, 2]))
            stats["r_min_obs"] = float(np.min(r))
            stats["r_max_obs"] = float(np.max(r))
            stats["r_mean_obs"] = float(np.mean(r))

        return out, stats

    def to_o3d(self, pts: np.ndarray):
        stats = {}
        stats["pts_in"] = int(pts.shape[0])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        voxel = float(self.get_parameter("voxel").value)
        t0 = time.time()
        if voxel > 0.0:
            pcd = pcd.voxel_down_sample(voxel)
        stats["after_voxel"] = int(np.asarray(pcd.points).shape[0])
        stats["voxel_sec"] = float(time.time() - t0)

        max_pts = int(self.get_parameter("max_points").value)
        n = np.asarray(pcd.points).shape[0]
        if n > max_pts:
            idx = np.random.choice(n, size=max_pts, replace=False)
            pcd = pcd.select_by_index(idx)
        stats["after_cap"] = int(np.asarray(pcd.points).shape[0])

        use_p2l = bool(self.get_parameter("use_point_to_plane").value)
        stats["normals"] = 0
        if use_p2l:
            n2 = np.asarray(pcd.points).shape[0]
            if n2 >= 200:
                t1 = time.time()
                rad = max(voxel * 3.0, 0.3)
                pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=50)
                )
                stats["normals"] = int(len(pcd.normals))
                stats["normals_sec"] = float(time.time() - t1)
        return pcd, stats

    def cb(self, msg: PointCloud2):
        self.frame_count += 1
        nlog = int(self.get_parameter("log_every_n_frames").value)
        do_log = (nlog <= 1) or (self.frame_count % nlog == 0)

        if do_log:
            self.get_logger().info(
                f"RX frame#{self.frame_count} stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec} "
                f"frame_id={msg.header.frame_id} width={msg.width} point_step={msg.point_step}"
            )

        # ---- read xyz ----
        t_read0 = time.time()
        try:
            pts_src = self.read_xyz(msg)
        except Exception as e:
            self.get_logger().warn(f"READ_XYZ failed: {e}")
            return
        t_read = time.time() - t_read0

        if do_log:
            self.get_logger().info(f"READ_XYZ: n={pts_src.shape[0]} dtype={pts_src.dtype} sec={t_read:.4f}")

        if pts_src.shape[0] < 1000:
            if do_log:
                self.get_logger().warn(f"EARLY_RETURN: too few raw points ({pts_src.shape[0]} < 1000)")
            return

        # ---- TF transform ----
        source = msg.header.frame_id
        target = self.get_parameter("target_frame").value or source
        pts = pts_src

        if target != source:
            tf_timeout = float(self.get_parameter("tf_timeout_sec").value)
            try:
                tfm = self.tfbuf.lookup_transform(
                    target, source, msg.header.stamp, timeout=Duration(seconds=tf_timeout)
                )
                T = tf_to_matrix(tfm)
                pts_h = np.c_[pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)]
                pts = (T @ pts_h.T).T[:, :3]
                if do_log:
                    self.get_logger().info(f"TF OK: {target} <- {source} (applied to {pts.shape[0]} pts)")
            except Exception as e:
                self.get_logger().warn(f"TF FAIL: {target} <- {source}: {e}")
                return
        else:
            if do_log:
                self.get_logger().info(f"TF SKIP: target==source=={source}")

        # ---- preprocess filters ----
        pts_filt, fstats = self.preprocess(pts)
        if do_log:
            self.get_logger().info(
                "FILTER: "
                f"in={fstats.get('in')} keep_range={fstats.get('keep_range')} keep_z={fstats.get('keep_z')} "
                f"keep_both={fstats.get('keep_both')} out={fstats.get('out')} | "
                f"z_obs[min,max,mean]={fstats.get('z_min_obs',0):.3f},{fstats.get('z_max_obs',0):.3f},{fstats.get('z_mean_obs',0):.3f} "
                f"r_obs[min,max,mean]={fstats.get('r_min_obs',0):.3f},{fstats.get('r_max_obs',0):.3f},{fstats.get('r_mean_obs',0):.3f}"
            )

        if pts_filt.shape[0] < 1000:
            if do_log:
                self.get_logger().warn(f"EARLY_RETURN: too few filtered points ({pts_filt.shape[0]} < 1000)")
            return

        # ---- Open3D build + downsample ----
        pcd, pstats = self.to_o3d(pts_filt)
        npts = int(np.asarray(pcd.points).shape[0])
        if do_log:
            self.get_logger().info(
                "PCD: "
                f"pts_in={pstats['pts_in']} after_voxel={pstats['after_voxel']} "
                f"after_cap={pstats['after_cap']} voxel_sec={pstats.get('voxel_sec',0):.4f} "
                f"normals={pstats.get('normals',0)} normals_sec={pstats.get('normals_sec',0):.4f}"
            )

        if npts < 300:
            if do_log:
                self.get_logger().warn(f"EARLY_RETURN: too few PCD points ({npts} < 300)")
            return

        stamp = msg.header.stamp

        # ---- initialize ----
        if self.prev_pcd is None:
            self.prev_pcd = pcd
            self.publish_pose(stamp, do_log=do_log, note="INIT")
            if do_log:
                self.get_logger().info(f"INIT DONE: prev_pcd set with {npts} points")
            return

        # ---- ICP ----
        max_corr = float(self.get_parameter("icp_max_corr_dist").value)
        max_iter = int(self.get_parameter("icp_max_iter").value)
        min_fit = float(self.get_parameter("min_fitness").value)
        use_p2l = bool(self.get_parameter("use_point_to_plane").value)

        init = self.T_last_delta.copy()
        t_icp0 = time.time()

        try:
            if use_p2l and len(self.prev_pcd.normals) and len(pcd.normals):
                reg = o3d.pipelines.registration.registration_icp(
                    pcd, self.prev_pcd, max_corr, init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
                )
                method = "P2L"
            else:
                reg = o3d.pipelines.registration.registration_icp(
                    pcd, self.prev_pcd, max_corr, init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
                )
                method = "P2P"
        except Exception as e:
            self.get_logger().warn(f"ICP EXCEPTION: {e}")
            self.prev_pcd = pcd
            self.publish_pose(stamp, do_log=do_log, note="ICP_EXCEPTION")
            return

        t_icp = time.time() - t_icp0
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        T_prev_to_curr = np.asarray(reg.transformation, dtype=np.float64)

        # quick motion magnitude logs (helps detect crazy jumps)
        dt = T_prev_to_curr[:3, 3]
        trans_norm = float(np.linalg.norm(dt))
        # yaw approx from rotation
        yaw = float(math.atan2(T_prev_to_curr[1, 0], T_prev_to_curr[0, 0]))

        if do_log:
            self.get_logger().info(
                f"ICP {method}: fit={fitness:.3f} rmse={rmse:.4f} sec={t_icp:.3f} "
                f"delta_trans_norm={trans_norm:.3f}m yaw_delta={yaw:.3f}rad max_corr={max_corr} iter={max_iter}"
            )

        if (fitness < min_fit) or (not np.isfinite(rmse)):
            self.get_logger().warn(f"ICP REJECT: fitness {fitness:.3f} < {min_fit:.3f} or rmse nan")
            self.prev_pcd = pcd
            self.publish_pose(stamp, do_log=do_log, note="ICP_REJECT")
            return

        # ---- accumulate ----
        self.T_odom_base = self.T_odom_base @ T_prev_to_curr
        self.T_last_delta = T_prev_to_curr
        self.prev_pcd = pcd

        self.publish_pose(stamp, do_log=do_log, note="ICP_OK")

    def publish_pose(self, stamp, do_log: bool = True, note: str = ""):
        odom_frame = self.get_parameter("odom_frame").value
        base_frame = self.get_parameter("base_frame").value

        odom = odom_msg_from_T(self.T_odom_base, odom_frame, base_frame, stamp)
        self.pub_odom.publish(odom)
        self.tfb.sendTransform(tf_from_T(self.T_odom_base, odom_frame, base_frame, stamp))

        ps = PoseStamped()
        ps.header.frame_id = odom_frame
        ps.header.stamp = stamp
        ps.pose = odom.pose.pose

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(ps)
        if len(self.path_msg.poses) > 2000:
            self.path_msg.poses = self.path_msg.poses[-2000:]
        self.pub_path.publish(self.path_msg)

        if do_log:
            t = self.T_odom_base[:3, 3]
            self.get_logger().info(
                f"PUBLISH[{note}]: TF {odom_frame}->{base_frame} "
                f"pose_xyz=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) path_len={len(self.path_msg.poses)}"
            )


def main():
    rclpy.init()
    node = HesaiICPOdomDebug()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

