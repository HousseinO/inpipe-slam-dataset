#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
import csv
from rclpy.time import Time


class TfLogger(Node):
    def __init__(self, parent_frame, child_frame, output_csv):
        super().__init__("tf_logger")

        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.output_csv = output_csv

        self.get_logger().info(
            f"Logging TF {self.parent_frame} -> {self.child_frame} into {self.output_csv}"
        )

        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Prepare CSV file
        self.csv_file = open(self.output_csv, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "sec", "nanosec",
            "parent_frame", "child_frame",
            "x", "y", "z",
            "qx", "qy", "qz", "qw"
        ])

        # Run at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                Time()
            )
        except TransformException as ex:
            self.get_logger().warn(str(ex))
            return

        stamp = tf.header.stamp

        self.csv_writer.writerow([
            stamp.sec, stamp.nanosec,
            self.parent_frame, self.child_frame,
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ])
        self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main():
    # Expect 3 arguments
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python3 evaluation_node.py <parent_frame> <child_frame> <output_csv>")
        return

    parent_frame = sys.argv[1]
    child_frame = sys.argv[2]
    output_csv = sys.argv[3]

    rclpy.init()
    node = TfLogger(parent_frame, child_frame, output_csv)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

