import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np

class BlueLineFollower(Node):
    def __init__(self):
        super().__init__('blue_line_follower')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.mask_pub = self.create_publisher(Image, "/blue_mask", 1)

        # PID gains
        self.k_ang = 0.01
        self.k_lin = 0.127

        # minimum contour area to be considered valid
        self.min_area = 200

        self.get_logger().info("BlueLineFollower ready.")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define blue range and mask
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Morphological operations to clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Publish the mask for visualization
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        # Find contours in the mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self.cmd_pub.publish(Twist())  # stop if no line
            return

        # Select the largest contour
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            self.cmd_pub.publish(Twist())  # ignore small areas
            return

        # Calculate centroid of the contour
        M = cv2.moments(c)
        if M['m00'] == 0:
            return  # avoid division by zero
        cx = int(M['m10'] / M['m00'])

        # Compute angular error (how far from center)
        img_cx = frame.shape[1] / 2
        err_x = cx - img_cx
        ang_z = -self.k_ang * err_x

        # Constant forward velocity
        lin_x = self.k_lin

        # Create and publish Twist message
        twist = Twist()
        twist.angular.z = -np.clip(ang_z, -1.0, 1.0)
        twist.linear.x = -lin_x
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = BlueLineFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
