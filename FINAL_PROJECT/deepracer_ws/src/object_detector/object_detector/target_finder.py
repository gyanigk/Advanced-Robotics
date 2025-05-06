import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np

class TargetFinder(Node):
    def __init__(self):
        super().__init__('target_finder')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 1)
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 1)
        self.mask_pub = self.create_publisher(Image, "/target_mask", 1)

        # PID gains (tune on your robot!)
        self.k_ang = 0.01
        self.k_lin = 1e-5

        # area thresholds (pixels²)
        self.min_area = 200    # ignore tiny blobs
        self.goal_area = 100000  # stop when contour large enough
        self.get_logger().info("TargetFinder ready.")

    def image_callback(self, msg):
        # --- 1. Vision ---
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, (35,80,80), (85,255,255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)
        # find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            # no target in view → stop
            self.cmd_pub.publish(Twist())
            return

        # pick largest
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        self.get_logger().info(f"area {area}")
        if area < self.min_area:
            # too small → ignore
            self.cmd_pub.publish(Twist())
            return

        # centroid
        M  = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        # image center
        img_cx = frame.shape[1] / 2

        # --- 2. Control ---
        err_x = cx - img_cx
        ang_z = -self.k_ang * err_x

        # drive forward until area ≈ goal_area
        lin_x = self.k_lin * (self.goal_area - area)
        self.get_logger().info(f"speed {lin_x}")

        twist = Twist()
        twist.angular.z = -np.clip(ang_z, -3.0, 3.0)
        # only drive forward; once area > goal, lin_x → zero or negative
        twist.linear.x  = -max(0.0, min(lin_x, 0.085))
        self.get_logger().info(f"speed {twist.linear.x}")

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TargetFinder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()