#!/usr/bin/env python3
"""
Live-plot IMU linear_acceleration.x vs time (ms)
for /inpipe_bot/imu/data without dropping spikes.

ROS2 Humble + matplotlib, using a separate thread for rclpy.spin
so all IMU messages (e.g. 200 Hz) are handled.
"""

import threading
import collections

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Imu

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ImuAccelPlotNode(Node):
    def __init__(self):
        super().__init__('imu_accel_plot_node')

        # Parameters
        self.declare_parameter('topic', '/inpipe_bot/imu/data')
        topic = self.get_parameter('topic').get_parameter_value().string_value

        # Big enough to hold several seconds at 200 Hz
        self.maxlen = 5000  # 5000 / 200 Hz = 25 s
        self.times = collections.deque(maxlen=self.maxlen)
        self.ax_vals = collections.deque(maxlen=self.maxlen)

        self.t0 = None

        # QoS: reliable, keep_last with big depth to avoid drops
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2000,
        )

        self.sub = self.create_subscription(Imu, topic, self.imu_callback, qos)
        self.get_logger().info(f"Subscribed to {topic}")

        # Matplotlib setup
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], linewidth=1.5)
        self.ax.set_xlabel('Time [ms]')
        self.ax.set_ylabel('Linear Accel X [m/s²]')
        self.ax.set_title('IMU linear_acceleration.x')
        self.ax.grid(True)

        # Animation: redraw ~50 Hz (independent from IMU rate)
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=20,
            blit=False
        )

    def imu_callback(self, msg: Imu):
        # This must be VERY cheap: just push to buffers.

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t

        rel_t_ms = (t - self.t0) * 1000.0
        ax = msg.linear_acceleration.x

        self.times.append(rel_t_ms)
        self.ax_vals.append(ax)

    def update_plot(self, _frame):
        # Called by matplotlib thread/timer.
        if not self.times:
            return self.line,

        # Copy references once (avoid racing while plotting)
        x = list(self.times)
        y = list(self.ax_vals)

        self.line.set_data(x, y)

        # X axis: from oldest to newest with small margin
        self.ax.set_xlim(x[0], x[-1] + 1.0)

        # Y axis: autoscale with margin
        ymin = min(y)
        ymax = max(y)
        if ymin == ymax:
            ymin -= 0.1
            ymax += 0.1
        else:
            margin = 0.1 * (ymax - ymin)
            ymin -= margin
            ymax += margin
        self.ax.set_ylim(ymin, ymax)

        return self.line,


def ros_spin(node):
    # Run rclpy.spin in its own thread so callbacks keep up with 200 Hz
    rclpy.spin(node)


def main(args=None):
    rclpy.init(args=args)
    node = ImuAccelPlotNode()

    # Start ROS spinning in background
    spin_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()







##!/usr/bin/env python3
#"""
#ROS2 node to live-plot IMU linear acceleration x-axis vs time.
#Topic: /inpipe_bot/imu/data
#"""

#import rclpy
#from rclpy.node import Node
#from sensor_msgs.msg import Imu
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import collections
#import time


#class ImuAccelPlotNode(Node):
#    def __init__(self):
#        super().__init__('imu_accel_plot_node')

#        # Parameters
#        self.declare_parameter('topic', '/inpipe_bot/imu/data')
#        topic = self.get_parameter('topic').get_parameter_value().string_value

#        # Buffer for plotting (keep last N samples)
#        self.maxlen = 500
#        self.times = collections.deque(maxlen=self.maxlen)
#        self.ax_vals = collections.deque(maxlen=self.maxlen)

#        # Subscribe to IMU
#        self.sub = self.create_subscription(Imu, topic, self.imu_callback, 10)
#        self.get_logger().info(f"Subscribed to {topic}")

#        # Setup matplotlib figure
#        plt.ion()
#        self.fig, self.ax = plt.subplots()
#        self.line, = self.ax.plot([], [], 'b-', lw=2)
#        self.ax.set_xlabel('Time [s]')
#        self.ax.set_ylabel('Linear Accel X [m/s²]')
#        self.ax.set_title('IMU linear_acceleration.x')
#        self.ax.grid(True)

#        self.start_time = time.time()
#        self.ani = animation.FuncAnimation(
#            self.fig, self.update_plot, interval=100
#        )

#    def imu_callback(self, msg: Imu):
#        # Compute relative timestamp
#        t = time.time() - self.start_time
#        ax = msg.linear_acceleration.x

#        self.times.append(t)
#        self.ax_vals.append(ax)

#    def update_plot(self, frame):
#        if not self.times:
#            return self.line,
#        self.line.set_data(self.times, self.ax_vals)
#        self.ax.set_xlim(self.times[0], self.times[-1] + 0.01)
#        self.ax.set_ylim(
#            min(self.ax_vals) - 0.5, max(self.ax_vals) + 0.5
#        )
#        self.fig.canvas.draw()
#        return self.line,


#def main(args=None):
#    rclpy.init(args=args)
#    node = ImuAccelPlotNode()

#    try:
#        plt.show(block=False)
#        while rclpy.ok():
#            rclpy.spin_once(node, timeout_sec=0.01)
#            plt.pause(0.01)
#    except KeyboardInterrupt:
#        pass
#    finally:
#        node.destroy_node()
#        rclpy.shutdown()


#if __name__ == '__main__':
#    main()

