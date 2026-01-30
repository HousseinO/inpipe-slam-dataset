#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import numpy as np
import matplotlib.pyplot as plt

from gtsam import (
    PreintegrationParams,
    PreintegratedImuMeasurements,
    Rot3,
    Point3,
    NavState,
    imuBias,
)
from geometry_msgs.msg import Twist


class ImuPreintegrationNode(Node):
    def __init__(self):
        super().__init__('imu_preintegration_node')
        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True
            )
        ])

        # -------------------- Parameters --------------------
        self.declare_parameter('imu_topic', '/inpipe_bot/imu/data')
        self.declare_parameter('traj_topic', 'preint_traj')
        self.declare_parameter('pose_topic', 'preint_pose')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('imu_rate', 200.0)        # Hz
        self.declare_parameter('keyframe_dt', 0.04)       # s
        self.last_cmd = Twist()
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

        

        imu_topic = self.get_parameter('imu_topic').value
        traj_topic = self.get_parameter('traj_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        imu_rate = float(self.get_parameter('imu_rate').value)
        self.keyframe_dt = float(self.get_parameter('keyframe_dt').value)

        dt_nominal = 1.0 / imu_rate

        # Noise params (tune for your IMU)
        acc_stddev = 0.03
        gyro_stddev = 0.001
        acc_bias_stddev = 0.001
        gyro_bias_stddev = 0.0001

        sigma_acc = acc_stddev / np.sqrt(dt_nominal)
        sigma_gyro = gyro_stddev / np.sqrt(dt_nominal)
        sigma_acc_bias = acc_bias_stddev / np.sqrt(dt_nominal)
        sigma_gyro_bias = gyro_bias_stddev / np.sqrt(dt_nominal)

        # -------------------- GTSAM preintegration --------------------
        g = 9.81
        self.params = PreintegrationParams.MakeSharedU(g)
        self.params.setAccelerometerCovariance((sigma_acc ** 2) * np.eye(3))
        self.params.setGyroscopeCovariance((sigma_gyro ** 2) * np.eye(3))
        self.params.setIntegrationCovariance((1e-4 ** 2) * np.eye(3))
        if hasattr(self.params, "setBiasAccCovariance"):
            self.params.setBiasAccCovariance((sigma_acc_bias ** 2) * np.eye(3))
        else:
            self.get_logger().warn("setBiasAccCovariance not available; skipping.")
        if hasattr(self.params, "setBiasOmegaCovariance"):
            self.params.setBiasOmegaCovariance((sigma_gyro_bias ** 2) * np.eye(3))
        else:
            self.get_logger().warn("setBiasOmegaCovariance not available; skipping.")

        # Bias calibration state
        self.calib_samples_needed = 1000 #4000 #2.5 x2 x2 = 5s  # ~2.5s at 200Hz 500
        self.calib_count = 0
        self.acc_sum = np.zeros(3)
        self.omega_sum = np.zeros(3)
        self.calibrated = False
        self.bias = imuBias.ConstantBias()

        # Preintegrator placeholder (will re-init after calibration)
        self.preintegrated = PreintegratedImuMeasurements(self.params, self.bias)

        # Initial state
        self.current_state = NavState(
            Rot3(),
            Point3(0.0, 0.0, 0.0),
            Point3(0.0, 0.0, 0.0),
        )

        self.last_time = None

        # Logs for optional plots
        self.t_log, self.x_log, self.y_log, self.z_log = [], [], [], []

        # -------------------- ROS I/O --------------------
        self.sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 200)
        self.traj_pub = self.create_publisher(Path, traj_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)

        # Path message that accumulates poses
        self.traj_msg = Path()
        self.traj_msg.header.frame_id = self.frame_id

        self.get_logger().info(
            f"IMU preintegration node started, listening on {imu_topic}"
        )
        self.get_logger().info(
            f"Publishing trajectory to '{traj_topic}' and pose to '{pose_topic}' in frame '{self.frame_id}'."
        )
        self.get_logger().info(
            f"Calibrating bias using first {self.calib_samples_needed} IMU samples; keep robot still."
        )

    # -------------------- Helpers --------------------
    def _get_delta_t(self):
        if hasattr(self.preintegrated, "deltaT"):
            return self.preintegrated.deltaT()
        if hasattr(self.preintegrated, "deltaTij"):
            return self.preintegrated.deltaTij()
        return 0.0
    def _cmd_cb(self, msg: Twist):
            self.last_cmd = msg
        
    def _moving_cmd(self):
        if self.last_cmd is None: return False
        return (abs(self.last_cmd.linear.x) > 0.02 or
                abs(self.last_cmd.linear.y) > 0.02 or
                abs(self.last_cmd.angular.z) > 0.02)

    @staticmethod
    def _position_components(p):
        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            return float(p.x()), float(p.y()), float(p.z())
        return float(p[0]), float(p[1]), float(p[2])

    @staticmethod
    def _rot_to_quat(R):
        """Convert 3x3 rotation matrix to quaternion (x,y,z,w) robustly."""
        R = np.asarray(R, dtype=float)
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0  # S = 4*qw
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return float(qx), float(qy), float(qz), float(qw)

    @staticmethod
    def _navstate_rot(state: NavState) -> Rot3:
        """Return a Rot3 from a NavState robustly across bindings."""
        if hasattr(state, "attitude"):
            return state.attitude()
        return state.pose().rotation()

    # -------------------- Main callback --------------------
    def imu_callback(self, msg: Imu):
        # Use IMU timestamp for consistency
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_time is None:
            self.last_time = t

        dt = t - self.last_time
        #print(dt)
        if dt <= 0.0:
            return
        self.last_time = t

        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=float)
        omega = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=float)

        # --------- 1) Bias calibration phase ---------
        if not self.calibrated:
            self.acc_sum += acc
            self.omega_sum += omega
            self.calib_count += 1

            if self.calib_count >= self.calib_samples_needed:
                mean_acc = self.acc_sum / self.calib_count
                mean_omega = self.omega_sum / self.calib_count

                # Adjust depending on your IMU convention:
                expected_acc = np.array([0.0, 0.0, 9.81])
                acc_bias = mean_acc - expected_acc
                gyro_bias = mean_omega

                self.bias = imuBias.ConstantBias(acc_bias, gyro_bias)
                self.preintegrated = PreintegratedImuMeasurements(self.params, self.bias)
                self.calibrated = True
                self.get_logger().info(
                    f"Calibration done.\n"
                    f"  mean_acc = {mean_acc}\n"
                    f"  mean_omega = {mean_omega}\n"
                    f"  acc_bias = {acc_bias}\n"
                    f"  gyro_bias = {gyro_bias}"
                )
            return  # don't integrate until calibrated

        # --------- 2) Normal preintegration after calibration ---------
        self.preintegrated.integrateMeasurement(acc, omega, dt)
        
        # Stationary detection thresholds (tune!)
        ang_thresh = 0.005      # rad/s
        lin_thresh = 0.08      # m/s^2
        g_thresh   = 0.08
        g = 9.81

        acc_norm_err = np.linalg.norm(acc - np.array([0.0, 0.0, 9.81]))
        acc_norm = np.linalg.norm(acc)
        is_stationary = (np.linalg.norm(omega) < ang_thresh) and (abs(acc_norm - g) < g_thresh)
        if is_stationary and not self._moving_cmd():
            # Zero the velocity and keep pose
            self.current_state = NavState(
                self._navstate_rot(self.current_state),
                self.current_state.position(),
                Point3(0.0, 0.0, 0.0)
            )

            # --- slow bias adaptation while still (ENABLE THIS) ---
            blend = 0.001  # small, stable
            new_acc_bias  = (1-blend)*self.bias.accelerometer() + blend*(acc - np.array([0,0,g]))
            new_gyro_bias = (1-blend)*self.bias.gyroscope()     + blend*omega
            self.bias = imuBias.ConstantBias(new_acc_bias, new_gyro_bias)

            # Recreate preintegrator with updated bias so future deltas are referenced properly
            self.preintegrated.resetIntegration()

#        if np.linalg.norm(omega) < ang_thresh and acc_norm_err < lin_thresh:
#            # ZUPT: clamp velocity and reset integrator
#            self.current_state = NavState(
#                self._navstate_rot(self.current_state),    # keep attitude
#                self.current_state.position(),             # keep position
#                Point3(0.0, 0.0, 0.0)                      # zero velocity
#            )
#            self.preintegrated.resetIntegration()
#            # (optional) very slow bias re-tuning while stationary:
#            # blend = 0.001
#            # self.bias = imuBias.ConstantBias(
#            #     (1-blend)*self.bias.accelerometer() + blend*(acc - np.array([0,0,9.81])),
#            #     (1-blend)*self.bias.gyroscope() + blend*omega
#            # )


        # Publish at keyframes
        if self._get_delta_t() >= self.keyframe_dt:
            new_state = self.preintegrated.predict(self.current_state, self.bias)
            

            # Position
            p = new_state.position()
            px, py, pz = self._position_components(p)

            # Orientation: Rot3 -> matrix -> quaternion
            Rg = self._navstate_rot(new_state)
            if hasattr(Rg, "rpy"):
                roll, pitch, yaw = Rg.rpy()              # radians
            elif hasattr(Rg, "xyz"):
                roll, pitch, yaw = Rg.xyz()              # radians (same ordering)
            else:
                R = np.asarray(Rg.matrix())
                # ZYX (yaw–pitch–roll)
                yaw   = np.arctan2(R[1, 0], R[0, 0])
                pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))  # clamp for numeric safety
                roll  = np.arctan2(R[2, 1], R[2, 2])

            self.get_logger().info(
                f"Euler (rad): roll={roll:.6f}, pitch={pitch:.6f}, yaw={yaw:.6f}"
            )
            self.get_logger().info(
                f"Euler (deg): roll={np.degrees(roll):.2f}, pitch={np.degrees(pitch):.2f}, yaw={np.degrees(yaw):.2f}"
            )
            qx, qy, qz, qw = self._rot_to_quat(Rg.matrix())

            # Log for plots
            self.t_log.append(t)
            self.x_log.append(px)
            self.y_log.append(py)
            self.z_log.append(pz)

            # ---- Publish PoseStamped (current pose) ----
            pose_msg = PoseStamped()
            pose_msg.header = msg.header  # keep IMU timestamp
            pose_msg.header.frame_id = self.frame_id
            pose_msg.pose.position.x = px
            pose_msg.pose.position.y = py
            pose_msg.pose.position.z = pz
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            self.pose_pub.publish(pose_msg)

            # ---- Update and publish Path (trajectory) ----
            self.traj_msg.header.stamp = msg.header.stamp
            self.traj_msg.header.frame_id = self.frame_id
            self.traj_msg.poses.append(pose_msg)
            self.traj_pub.publish(self.traj_msg)

            # Advance state & reset integrator
            self.current_state = new_state
            self.preintegrated.resetIntegration()

    # -------------------- Offline plotting --------------------
    def plot_results(self):
        n = min(len(self.t_log), len(self.x_log), len(self.y_log), len(self.z_log))
        if n == 0:
            self.get_logger().warn("No trajectory data collected; nothing to plot.")
            return

        t = np.asarray(self.t_log[:n])
        x = np.asarray(self.x_log[:n])
        y = np.asarray(self.y_log[:n])
        z = np.asarray(self.z_log[:n])

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("IMU Preintegrated Trajectory (X-Y)")
        plt.axis('equal')
        plt.grid(True)

        plt.figure()
        plt.plot(t, z)
        plt.xlabel("Time [s]")
        plt.ylabel("Z [m]")
        plt.title("IMU Preintegrated Z vs Time")
        plt.grid(True)

        plt.show()


def main():
    rclpy.init()
    node = ImuPreintegrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down, plotting results...")
        node.plot_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




##!/usr/bin/env python3
#import rclpy
#from rclpy.node import Node
#from sensor_msgs.msg import Imu

#import numpy as np
#import matplotlib.pyplot as plt

#import gtsam
#from gtsam import (
#    PreintegrationParams,
#    PreintegratedImuMeasurements,
#    Rot3,
#    Point3,
#    NavState,
#    imuBias,
#)


#class ImuPreintegrationNode(Node):
#    def __init__(self):
#        super().__init__('imu_preintegration_node')

#        imu_rate = 200.0
#        dt = 1.0 / imu_rate

#        # From your SDF
#        acc_stddev = 0.03
#        gyro_stddev = 0.001
#        acc_bias_stddev = 0.001
#        gyro_bias_stddev = 0.0001

#        sigma_acc = acc_stddev / np.sqrt(dt)
#        sigma_gyro = gyro_stddev / np.sqrt(dt)
#        sigma_acc_bias = acc_bias_stddev / np.sqrt(dt)
#        sigma_gyro_bias = gyro_bias_stddev / np.sqrt(dt)

#        g = 9.81
#        self.params = PreintegrationParams.MakeSharedU(g)

#        self.params.setAccelerometerCovariance((sigma_acc ** 2) * np.eye(3))
#        self.params.setGyroscopeCovariance((sigma_gyro ** 2) * np.eye(3))
#        self.params.setIntegrationCovariance((1e-4 ** 2) * np.eye(3))

#        if hasattr(self.params, "setBiasAccCovariance"):
#            self.params.setBiasAccCovariance((sigma_acc_bias ** 2) * np.eye(3))
#        else:
#            self.get_logger().warn("setBiasAccCovariance not available; skipping.")
#        if hasattr(self.params, "setBiasOmegaCovariance"):
#            self.params.setBiasOmegaCovariance((sigma_gyro_bias ** 2) * np.eye(3))
#        else:
#            self.get_logger().warn("setBiasOmegaCovariance not available; skipping.")

#        # ---- Bias calibration state ----
#        self.calib_samples_needed = 500  # use first 500 samples (~2.5s at 200Hz)
#        self.calib_count = 0
#        self.acc_sum = np.zeros(3)
#        self.omega_sum = np.zeros(3)
#        self.calibrated = False

#        # will be set after calibration
#        self.bias = imuBias.ConstantBias()

#        # Preintegrator placeholder (will re-init after calibration too)
#        self.preintegrated = PreintegratedImuMeasurements(self.params, self.bias)

#        # Initial state
#        self.current_state = NavState(
#            Rot3(),
#            Point3(0.0, 0.0, 0.0),
#            Point3(0.0, 0.0, 0.0),
#        )

#        self.last_time = None
#        self.keyframe_dt = 0.05

#        self.t_log, self.x_log, self.y_log, self.z_log = [], [], [], []

#        self.sub = self.create_subscription(
#            Imu,
#            '/inpipe_bot/imu/data',
#            self.imu_callback,
#            200,
#        )

#        self.get_logger().info(
#            "IMU preintegration node started, listening on /inpipe_bot/imu/data"
#        )
#        self.get_logger().info(
#            f"Calibrating bias using first {self.calib_samples_needed} IMU samples; keep robot still."
#        )

#    def _get_delta_t(self):
#        if hasattr(self.preintegrated, "deltaT"):
#            return self.preintegrated.deltaT()
#        if hasattr(self.preintegrated, "deltaTij"):
#            return self.preintegrated.deltaTij()
#        return 0.0

#    @staticmethod
#    def _position_components(p):
#        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
#            return float(p.x()), float(p.y()), float(p.z())
#        return float(p[0]), float(p[1]), float(p[2])

#    def imu_callback(self, msg: Imu):
#        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

#        if self.last_time is None:
#            self.last_time = t

#        dt = t - self.last_time
#        if dt <= 0.0:
#            return
#        self.last_time = t

#        acc = np.array([
#            msg.linear_acceleration.x,
#            msg.linear_acceleration.y,
#            msg.linear_acceleration.z,
#        ], dtype=float)
#        omega = np.array([
#            msg.angular_velocity.x,
#            msg.angular_velocity.y,
#            msg.angular_velocity.z,
#        ], dtype=float)

#        # --------- 1) Bias calibration phase ---------
#        if not self.calibrated:
#            self.acc_sum += acc
#            self.omega_sum += omega
#            self.calib_count += 1

#            if self.calib_count >= self.calib_samples_needed:
#                mean_acc = self.acc_sum / self.calib_count
#                mean_omega = self.omega_sum / self.calib_count

#                # IMPORTANT:
#                # Check what your IMU outputs when level & static:
#                # - If it's about [0, 0, 9.81], use expected = [0, 0, 9.81]
#                # - If it's about [0, 0, -9.81], flip the sign.
#                expected_acc = np.array([0.0, 0.0, 9.81])
#                acc_bias = mean_acc - expected_acc
#                gyro_bias = mean_omega

#                self.bias = imuBias.ConstantBias(acc_bias, gyro_bias)

#                # Recreate preintegrator with calibrated bias
#                self.preintegrated = PreintegratedImuMeasurements(
#                    self.params, self.bias
#                )

#                self.calibrated = True
#                self.get_logger().info(
#                    f"Calibration done.\n"
#                    f"  mean_acc = {mean_acc}\n"
#                    f"  mean_omega = {mean_omega}\n"
#                    f"  acc_bias = {acc_bias}\n"
#                    f"  gyro_bias = {gyro_bias}"
#                )
#            return  # don't integrate until calibrated

#        # --------- 2) Normal preintegration after calibration ---------
#        self.preintegrated.integrateMeasurement(acc, omega, dt)

#        if self._get_delta_t() >= self.keyframe_dt:
#            new_state = self.preintegrated.predict(self.current_state, self.bias)
#            p = new_state.position()
#            px, py, pz = self._position_components(p)

#            self.t_log.append(t)
#            self.x_log.append(px)
#            self.y_log.append(py)
#            self.z_log.append(pz)

#            self.current_state = new_state
#            self.preintegrated.resetIntegration()

#    def plot_results(self):
#        n = min(len(self.t_log), len(self.x_log), len(self.y_log), len(self.z_log))
#        if n == 0:
#            self.get_logger().warn("No trajectory data collected; nothing to plot.")
#            return

#        t = np.asarray(self.t_log[:n])
#        x = np.asarray(self.x_log[:n])
#        y = np.asarray(self.y_log[:n])
#        z = np.asarray(self.z_log[:n])

#        plt.figure()
#        plt.plot(x, y)
#        plt.xlabel("X [m]")
#        plt.ylabel("Y [m]")
#        plt.title("IMU Preintegrated Trajectory (X-Y)")
#        plt.axis('equal')
#        plt.grid(True)

#        plt.figure()
#        plt.plot(t, z)
#        plt.xlabel("Time [s]")
#        plt.ylabel("Z [m]")
#        plt.title("IMU Preintegrated Z vs Time")
#        plt.grid(True)

#        plt.show()


#def main():
#    rclpy.init()
#    node = ImuPreintegrationNode()
#    try:
#        rclpy.spin(node)
#    except KeyboardInterrupt:
#        pass
#    finally:
#        node.get_logger().info("Shutting down, plotting results...")
#        node.plot_results()
#        node.destroy_node()
#        rclpy.shutdown()


#if __name__ == '__main__':
#    main()

