#!/usr/bin/env python3
import math, time, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist



def q_from_rpy(roll, pitch, yaw):
    """Convert roll, pitch, yaw to quaternion."""
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    return Quaternion(
        x=sr*cp*cy - cr*sp*sy,
        y=cr*sp*cy + sr*cp*sy,
        z=cr*cp*sy - sr*sp*cy,
        w=cr*cp*cy + sr*sp*cy
    )

class Trajectory6DOF(Node):
    def __init__(self, entity_name='inpipe_bot_sensors'):
        super().__init__('trajectory_6dof_set_state')
        self.entity_name = entity_name
        srv_name = '/set_entity_state'
        self.cli = self.create_client(SetEntityState, srv_name)
        self.get_logger().info(f'Waiting for {srv_name} ...')
        self.cli.wait_for_service()
        self.get_logger().info(f'Connected to {srv_name}')
        self.z=0.5
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        

        # -------- Waypoints: x y z roll pitch yaw time(s)
        self.waypoints = [
            # straight 1:
            (0.0, 0.0, self.z,  0.0, 0.0, 0.0,   2.0),
            (3.0, 0.0, self.z,  0.0, 0.0, 0.0,   50.0),
            # elbow
            (3.0+0.707, 0.292, self.z,  0.0, 0.0, 0.7,   100.0),
            # straight
            (4.0, 0.5, self.z,  0.0, 0.0, 1.5708,   30.0),
            (4.0, 4.0, self.z,  0.0, 0.0, 1.5708,   50.0),
            #elbow
            (4.292, 4.707, self.z,  0.0, 0.0, 0.7,   2.0),
            #straight
            (4.5, 5.0, self.z,  0.0, 0.0, 0.0,   2.0),
            (8.0, 5.0, self.z,  0.0, 0.0, 0.0,   50.0),
            #elbow
            (8.707, 5.292, self.z,  0.0, 0.0, 0.7,   2.0),
            #straight
            (9.0, 6.0, self.z,  0.0, 0.0, 1.5708,   2.0),
            (9.0, 6.0+8.0, self.z,  0.0, 0.0, 1.5708,   130.0),
            #elbow
            (9.292, 0.707+6.0+8.0, self.z,  0.0, 0.0, 0.7,   2.0),
            #straight
            (10.0, 1.0+6.0+8.0, self.z,  0.0, 0.0, 0.0,   2.0),
            (10.0+3.0, 1.0+6.0+8.0, self.z,  0.0, 0.0, 0.0,   20.0),
        ]
        self.dt = 1.0 / 50.0  # 50 Hz update
        self.run_path()

    def send_pose(self, x,y,z, roll,pitch,yaw):
        pose = Pose(position=Point(x=x,y=y,z=z), orientation=q_from_rpy(roll,pitch,yaw))
        req = SetEntityState.Request()
        req.state = EntityState(name=self.entity_name, pose=pose, reference_frame='world')
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=0.2)

    def run_path(self):
        for i in range(1, len(self.waypoints)):
            x0,y0,z0,r0,p0,yaw0,_ = self.waypoints[i-1]
            x1,y1,z1,r1,p1,yaw1,T = self.waypoints[i]
            t0 = time.time()
            while True:
                s = (time.time() - t0) / T
                if s >= 1.0: s = 1.0
                # linear interpolation
                x = x0 + (x1 - x0)*s
                y = y0 + (y1 - y0)*s
                z = z0 + (z1 - z0)*s
                roll  = r0 + (r1 - r0)*s
                pitch = p0 + (p1 - p0)*s
                yaw   = yaw0 + ((yaw1 - yaw0 + math.pi) % (2*math.pi) - math.pi)*s
                self.send_pose(x,y,z, roll,pitch,yaw)
                if s >= 1.0:
                    break
                time.sleep(self.dt)
        self.get_logger().info('Trajectory complete.')
        rclpy.shutdown()

def main():
    rclpy.init()
    Trajectory6DOF('inpipe_bot_sensors')

if __name__ == '__main__':
    main()

