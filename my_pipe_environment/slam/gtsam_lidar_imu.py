# Directory layout (put this in a ROS 2 workspace, e.g., ~/ros2_ws/src)
#
# l515_imu_graph_slam/
# ├── package.xml
# ├── setup.cfg
# ├── setup.py
# ├── l515_imu_graph_slam
# │   ├── __init__.py
# │   └── l515_imu_graph_slam_node.py
# └── launch
#     └── l515_imu_graph_slam.launch.py
#
# --------------------------
# package.xml (minimal)
# --------------------------
# <?xml version="1.0"?>
# <package format="3">
#   <name>l515_imu_graph_slam</name>
#   <version>0.0.1</version>
#   <description>Graph-SLAM with L515 IMU + optional RGB-D odom using GTSAM</description>
#   <maintainer email="you@example.com">You</maintainer>
#   <license>Apache-2.0</license>
#   <buildtool_depend>ament_python</buildtool_depend>
#   <exec_depend>rclpy</exec_depend>
#   <exec_depend>sensor_msgs</exec_depend>
#   <exec_depend>geometry_msgs</exec_depend>
#   <exec_depend>nav_msgs</exec_depend>
#   <exec_depend>tf_transformations</exec_depend>
#   <exec_depend>python3-numpy</exec_depend>
#   <exec_depend>python3-opencv</exec_depend>
#   <exec_depend>python3-numpy</exec_depend>
#   <exec_depend>python3-yaml</exec_depend>
#   <!-- GTSAM is installed via pip; no ROS dep -->
# </package>
#
# --------------------------
# setup.cfg
# --------------------------
# [develop]
# script_dir=$base/lib/l515_imu_graph_slam
# [install]
# install_scripts=$base/lib/l515_imu_graph_slam
#
# --------------------------
# setup.py
# --------------------------
# from setuptools import setup
# import os
# from glob import glob
# package_name = 'l515_imu_graph_slam'
# setup(
#     name=package_name,
#     version='0.0.1',
#     packages=[package_name],
#     data_files=[
#         ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#         ('share/' + package_name + '/launch', glob('launch/*.py')),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='You',
#     maintainer_email='you@example.com',
#     description='Graph-SLAM with L515 IMU + optional RGB-D odom using GTSAM',
#     license='Apache-2.0',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'l515_imu_graph_slam_node = l515_imu_graph_slam.l515_imu_graph_slam_node:main',
#         ],
#     },
# )
#
# --------------------------
# launch/l515_imu_graph_slam.launch.py
# --------------------------
# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration
#
# def generate_launch_description():
#     imu_topic = LaunchConfiguration('imu_topic')
#     odom_topic = LaunchConfiguration('odom_topic')
#     return LaunchDescription([
#         DeclareLaunchArgument('imu_topic', default_value='/camera/imu'),
#         DeclareLaunchArgument('odom_topic', default_value='/rgbd/odom'),
#         Node(
#             package='l515_imu_graph_slam',
#             executable='l515_imu_graph_slam_node',
#             name='l515_imu_graph_slam_node',
#             output='screen',
#             parameters=[{
#                 'imu_topic': imu_topic,
#                 'odom_topic': odom_topic,
#                 'gravity': 9.80665,
#                 'imu_sigma_gyro': 0.0015,      # rad/sqrt(s)
#                 'imu_sigma_acc': 0.02,         # m/s^2/sqrt(s)
#                 'imu_sigma_bias_gyro': 1e-5,
#                 'imu_sigma_bias_acc': 1e-4,
#                 'keyframe_dt': 0.5,            # seconds between IMU keyframes
#             }]
#         )
#     ])
#
# --------------------------
# l515_imu_graph_slam/l515_imu_graph_slam_node.py
# --------------------------
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import numpy as np
import math

# GTSAM
import gtsam
from gtsam import symbol

class IMUGraphSLAM(Node):
    """
    Graph-SLAM back-end (Python + GTSAM 4.2) using L515 fused IMU topic (sensor_msgs/Imu)
    and optional RGB-D odometry as soft pose constraints.

    Variables per keyframe k: Xk (Pose3), Vk (Vel3), Bk (imu bias)
    Factors: Prior(X0,V0,B0), CombinedImuFactor(k-1 -> k), optional soft prior from odom.
    """

    def __init__(self):
        super().__init__('l515_imu_graph_slam_node')
        # --- Params
        self.imu_topic = self.declare_parameter('imu_topic', '/inpipe_bot/imu/data').value
        self.odom_topic = self.declare_parameter('odom_topic', '/pipe/odom').value
        self.g = float(self.declare_parameter('gravity', 9.80665).value)
        self.keyframe_dt = float(self.declare_parameter('keyframe_dt', 0.5).value)

        # IMU noise params
        sigma_gyro = float(self.declare_parameter('imu_sigma_gyro', 0.0015).value)
        sigma_acc = float(self.declare_parameter('imu_sigma_acc', 0.02).value)
        sigma_bias_gyro = float(self.declare_parameter('imu_sigma_bias_gyro', 1e-5).value)
        sigma_bias_acc = float(self.declare_parameter('imu_sigma_bias_acc', 1e-4).value)

        # --- GTSAM containers
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.isam_params = gtsam.ISAM2Params()
        

        self.X = lambda k: symbol('x', k)
        self.V = lambda k: symbol('v', k)
        self.B = lambda k: symbol('b', k)

        # --- IMU preintegration setup (GTSAM 4.2)
        imu_params = gtsam.PreintegrationParams.MakeSharedU(self.g)
        imu_params.setAccelerometerCovariance(np.eye(3) * (sigma_acc ** 2))
        imu_params.setGyroscopeCovariance(np.eye(3) * (sigma_gyro ** 2))
        imu_params.setIntegrationCovariance(np.eye(3) * 1e-6)
        bias_covar = np.diag([sigma_bias_acc**2]*3 + [sigma_bias_gyro**2]*3)

        self.current_bias = gtsam.imuBias_ConstantBias(np.zeros(3), np.zeros(3))
        self.preint = gtsam.PreintegratedImuMeasurements(imu_params, self.current_bias)

        # --- Noise models
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.deg2rad([1,1,1]), [0.05,0.05,0.05]])
        self.prior_vel_noise  = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.prior_bias_noise = gtsam.noiseModel.Gaussian.Covariance(bias_covar * 100.0)
        self.odom_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.deg2rad([2,2,2]), [0.1,0.1,0.1]])

        # --- State bookkeeping
        self.k = 0
        self.last_keyframe_time = None
        self.last_imu_time = None

        # --- Publishers / Subscribers
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, 200)
        self.sub_odom = self.create_subscription(PoseWithCovarianceStamped, self.odom_topic, self.cb_odom, 10)
        self.path_pub = self.create_publisher(Path, 'graph_slam/path', 10)

        # --- Initialize prior at k=0
        self.initialize_prior()
        self.get_logger().info(f"IMU Graph-SLAM (GTSAM 4.2) ready. Subscribed to {self.imu_topic} and {self.odom_topic}")

    def initialize_prior(self):
        # Pose prior (identity), velocity prior (zero), bias prior (zero)
        X0 = gtsam.Pose3()
        V0 = np.zeros(3)
        B0 = gtsam.imuBias_ConstantBias()

        self.values.insert(self.X(0), X0)
        self.values.insert(self.V(0), V0)
        self.values.insert(self.B(0), B0)

        self.graph.add(gtsam.PriorFactorPose3(self.X(0), X0, self.prior_pose_noise))
        self.graph.add(gtsam.PriorFactorVector(self.V(0), V0, self.prior_vel_noise))
        self.graph.add(gtsam.PriorFactorConstantBias(self.B(0), B0, self.prior_bias_noise))

        self.isam.update(self.graph, self.values)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.k = 0
        self.last_keyframe_time = None

    def cb_imu(self, msg: Imu):
        # Time handling
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = t
            return
        dt = max(1e-4, t - self.last_imu_time)
        self.last_imu_time = t

        # Extract gyro (rad/s) and accel (m/s^2)
        a = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.preint.integrateMeasurement(a, w, dt)

        # Trigger keyframe by time
        if self.last_keyframe_time is None:
            self.last_keyframe_time = t
        if (t - self.last_keyframe_time) >= self.keyframe_dt:
            self.add_imu_keyframe()
            self.last_keyframe_time = t

    def add_imu_keyframe(self):
        k1 = self.k + 1

        # Get current estimates (defaults if not present)
        est = self.isam.calculateEstimate()
        Xk = est.atPose3(self.X(self.k)) if est.exists(self.X(self.k)) else gtsam.Pose3()
        Vk = est.atVector(self.V(self.k)) if est.exists(self.V(self.k)) else np.zeros(3)
        Bk = est.atConstantBias(self.B(self.k)) if est.exists(self.B(self.k)) else gtsam.imuBias_ConstantBias()

        # Predict next state using preintegrated IMU (GTSAM 4.2 API): pim.predict(NavState, bias)
        nav_k = gtsam.NavState(Xk, Vk)
        nav_k1 = self.preint.predict(nav_k, Bk)
        Xk1_pred = nav_k1.pose()
        Vk1_pred = nav_k1.v()

        # Insert initial guesses
        self.values.insert(self.X(k1), Xk1_pred)
        self.values.insert(self.V(k1), Vk1_pred)
        self.values.insert(self.B(k1), Bk)  # propagate bias

        # Add CombinedImuFactor between k and k1
        imu_factor = gtsam.CombinedImuFactor(self.X(self.k), self.V(self.k), self.X(k1), self.V(k1), self.B(self.k), self.B(k1), self.preint)
        self.graph.add(imu_factor)

        # Optional: small random-walk on bias to keep it well-conditioned
        bias_rw = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.graph.add(gtsam.BetweenFactorConstantBias(self.B(self.k), self.B(k1), gtsam.imuBias_ConstantBias(), bias_rw))

        # Optimize and publish
        self.optimize_and_publish()

        # Prepare for next window
        self.k = k1
        self.preint.resetIntegrationAndSetBias(Bk)

    def cb_odom(self, msg: PoseWithCovarianceStamped):
        if self.k == 0:
            return
        # Convert msg pose to Pose3
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        Z_world = gtsam.Pose3(gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z), gtsam.Point3(p.x, p.y, p.z))
        soft_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.deg2rad([3,3,3]), [0.15,0.15,0.15]])
        self.graph.add(gtsam.PriorFactorPose3(self.X(self.k), Z_world, soft_prior_noise))
        self.optimize_and_publish()

    def optimize_and_publish(self):
        self.isam.update(self.graph, self.values)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.publish_path()

    def publish_path(self):
        est = self.isam.calculateEstimate()
        path = Path()
        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        path.header.frame_id = 'map'
        for i in range(self.k + 1):
            if not est.exists(self.X(i)):
                continue
            Xi = est.atPose3(self.X(i))
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = 'map'
            t = Xi.translation()
            q = Xi.rotation().toQuaternion()
            pose.pose.position.x = t.x()
            pose.pose.position.y = t.y()
            pose.pose.position.z = t.z()
            pose.pose.orientation.w = q.w()
            pose.pose.orientation.x = q.x()
            pose.pose.orientation.y = q.y()
            pose.pose.orientation.z = q.z()
            path.poses.append(pose)
        self.path_pub.publish(path)


def main():
    rclpy.init()
    node = IMUGraphSLAM()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == "__main__":
    main()

