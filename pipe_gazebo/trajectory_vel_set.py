#!/usr/bin/env python3
import math, time, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState

def q_from_rpy(roll, pitch, yaw):
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    return Quaternion(
        x=sr*cp*cy - cr*sp*sy,
        y=cr*sp*cy + sr*cp*sy,
        z=cr*cp*sy - sr*sp*cy,
        w=cr*cp*cy + sr*sp*sy
    )

def wrap_pi(a):  # [-pi, pi]
    return (a + math.pi) % (2.0*math.pi) - math.pi

class Trajectory6DoFVel(Node):
    """
    Drives a model in 6-DoF by setting BOTH pose and twist via /set_entity_state,
    so the IMU sees realistic angular velocity and linear acceleration.
    """
    def __init__(self, entity_name='inpipe_bot_sensors'):
        super().__init__('trajectory_6dof_set_twist')
        self.entity_name = entity_name
        self.cli = self.create_client(SetEntityState, '/set_entity_state')
        self.get_logger().info('Waiting for /set_entity_state ...')
        self.cli.wait_for_service()
        self.get_logger().info('Connected to /set_entity_state')
        self.dt = 1.0/50.0  # 50 Hz

        z = 0.5
        # (x, y, z, roll, pitch, yaw, duration)
        self.waypoints = [
            (0.0, 0.0, z, 0.0, 0.0, 0.0,     2.0),
            (3.0, 0.0, z, 0.0, 0.0, 0.0,    50.0),
            (3.707, 0.292, z, 0.0, 0.0, 0.7, 10.0),       # ~40°
            (4.0, 0.5,   z, 0.0, 0.0, 1.5708, 10.0),      # 90°
            (4.0, 4.0,   z, 0.0, 0.0, 1.5708, 50.0),
            (4.292, 4.707, z, 0.0, 0.0, 0.7,  2.0),
            (4.5, 5.0,   z, 0.0, 0.0, 0.0,   2.0),
            (8.0, 5.0,   z, 0.0, 0.0, 0.0,  50.0),
            (8.707, 5.292, z, 0.0, 0.0, 0.7, 2.0),
            (9.0, 6.0,   z, 0.0, 0.0, 1.5708, 2.0),
            (9.0, 14.0,  z, 0.0, 0.0, 1.5708, 130.0),
            (9.292, 14.707, z, 0.0, 0.0, 0.7, 2.0),
            (10.0, 15.0, z, 0.0, 0.0, 0.0,  2.0),
            (13.0, 15.0, z, 0.0, 0.0, 0.0, 20.0),
        ]
        self.run_path()

    def send_state(self, pose: Pose, twist: Twist):
        req = SetEntityState.Request()
        req.state = EntityState(
            name=self.entity_name,
            pose=pose,
            twist=twist,
            reference_frame='world'
        )
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=0.2)

    def run_path(self):
        for i in range(1, len(self.waypoints)):
            x0,y0,z0,r0,p0,yaw0,_ = self.waypoints[i-1]
            x1,y1,z1,r1,p1,yaw1,T = self.waypoints[i]

            # constant velocities over segment
            dx, dy, dz = (x1-x0), (y1-y0), (z1-z0)
            droll  = wrap_pi(r1 - r0)
            dpitch = wrap_pi(p1 - p0)
            dyaw   = wrap_pi(yaw1 - yaw0)

            vx = dx / max(T, 1e-6)
            vy = dy / max(T, 1e-6)
            vz = dz / max(T, 1e-6)
            wx = droll  / max(T, 1e-6)
            wy = dpitch / max(T, 1e-6)
            wz = dyaw   / max(T, 1e-6)

            t0 = time.time()
            while True:
                s = (time.time() - t0) / T
                if s >= 1.0: s = 1.0

                # interpolate pose (for visualization continuity)
                x = x0 + dx*s
                y = y0 + dy*s
                z = z0 + dz*s
                roll  = r0 + droll *s
                pitch = p0 + dpitch*s
                yaw   = yaw0 + dyaw  *s

                pose = Pose(position=Point(x=x,y=y,z=z),
                            orientation=q_from_rpy(roll,pitch,yaw))
                twist = Twist()
                twist.linear.x,  twist.linear.y,  twist.linear.z  = vx, vy, vz
                twist.angular.x, twist.angular.y, twist.angular.z = wx, wy, wz

                self.send_state(pose, twist)
                if s >= 1.0:
                    break
                time.sleep(self.dt)

        self.get_logger().info('Trajectory complete.')
        rclpy.shutdown()

def main():
    rclpy.init()
    Trajectory6DoFVel('inpipe_bot_sensors')
    rclpy.shutdown()

if __name__ == '__main__':
    main()

