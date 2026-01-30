#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


def quat_normalize(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return x/n, y/n, z/n, w/n


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return quat_normalize(x, y, z, w)


def quat_from_omega(wx, wy, wz, dt):
    angle = math.sqrt(wx*wx + wy*wy + wz*wz) * dt
    if angle < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    ax, ay, az = wx, wy, wz
    inv = 1.0 / math.sqrt(ax*ax + ay*ay + az*az)
    ax *= inv; ay *= inv; az *= inv
    half = 0.5 * angle
    s = math.sin(half)
    c = math.cos(half)
    return quat_normalize(ax*s, ay*s, az*s, c)


def rotate_body_to_world(vx, vy, vz, q):
    # Rotate v (body) by q (world->body) into world frame
    x, y, z, w = q
    tx = 2.0 * (y*vz - z*vy)
    ty = 2.0 * (z*vx - x*vz)
    tz = 2.0 * (x*vy - y*vx)
    vwx = vx + w*tx + (y*tz - z*ty)
    vwy = vy + w*ty + (z*tx - x*tz)
    vwz = vz + w*tz + (x*ty - y*tx)
    return vwx, vwy, vwz


class CmdVelToGazebo(Node):
    """
    Kinematic 6-DoF controller:
    - Reads /cmd_vel (body twist)
    - Integrates pose in world
    - Calls /gazebo/set_entity_state so Gazebo + IMU plugin see motion
    """
    def __init__(self):
        super().__init__("cmd_vel_to_gazebo")
        
        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True
            )
        ])

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("model_name", "inpipe_bot_sensors")
        self.declare_parameter("rate", 100.0)

        cmd_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        rate = self.get_parameter("rate").get_parameter_value().double_value

        # /cmd_vel subscriber
        self.cmd_sub = self.create_subscription(Twist, cmd_topic, self.cmd_cb, 10)

        # SetEntityState client
        self.cli = self.create_client(SetEntityState, "/set_entity_state")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /set_entity_state...")

        # State: world pose of base_link/model
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0

        # Last cmd_vel (body frame)
        self.vx = self.vy = self.vz = 0.0
        self.wx = self.wy = self.wz = 0.0

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0/rate, self.update)

        self.get_logger().info(
            f"cmd_vel_to_gazebo running. Controlling model '{self.model_name}' from {cmd_topic}"
        )

    def cmd_cb(self, msg: Twist):
        self.vx = float(msg.linear.x)
        self.vy = float(msg.linear.y)
        self.vz = float(msg.linear.z)
        self.wx = float(msg.angular.x)
        self.wy = float(msg.angular.y)
        self.wz = float(msg.angular.z)

    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        # 1) integrate orientation from body rates
        dq = quat_from_omega(self.wx, self.wy, self.wz, dt)
        self.qx, self.qy, self.qz, self.qw = quat_multiply(
            (self.qx, self.qy, self.qz, self.qw), dq
        )

        # 2) rotate linear vel (body) into world
        vwx, vwy, vwz = rotate_body_to_world(
            self.vx, self.vy, self.vz,
            (self.qx, self.qy, self.qz, self.qw)
        )

        # 3) integrate position
        self.x += vwx * dt
        self.y += vwy * dt
        self.z = 0.5 #+= 0.5 # vwz * dt

        # 4) push state into Gazebo
        msg = EntityState()
        msg.name = self.model_name
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = self.z
        msg.pose.orientation.x = self.qx
        msg.pose.orientation.y = self.qy
        msg.pose.orientation.z = self.qz
        msg.pose.orientation.w = self.qw

        msg.twist.linear.x = vwx
        msg.twist.linear.y = vwy
        msg.twist.linear.z = vwz
        msg.twist.angular.x = self.wx
        msg.twist.angular.y = self.wy
        msg.twist.angular.z = self.wz

        req = SetEntityState.Request()
        req.state = msg
        self.cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToGazebo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

