import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class BlueLineFollower(Node):
    def __init__(self):
        super().__init__('blue_line_follower')
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.mask_pub = self.create_publisher(Image, "/line_mask", 1)
        self.bridge = CvBridge()
        self.twist = Twist()

        # PID controller parameters
        self.Kp = 0.008
        self.Ki = 0.0002
        self.Kd = 0.0003
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.last_time = self.get_clock().now()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Crop to lower region of interest (bottom 30% of the frame)
        h, w = cv_image.shape[:2]
        roi_y = int(h * 0.7)
        cv_image = cv_image[roi_y:, :]
        roi_h = cv_image.shape[0]  # Height of ROI

        # Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define blue color range
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Remove small noise and fill gaps
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Publish mask for debugging (uncomment if needed)
        # mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        # mask_msg.header = msg.header
        # self.mask_pub.publish(mask_msg)
        
        # Initialize variables
        cx = None
        self.twist = Twist()

        # Scan the mask horizontally to find left and right edges
        # Use a row near the bottom of the ROI for stability
        scan_y = int(roi_h * 0.8)  # Scan at 80% down the ROI
        row = mask[scan_y, :]

        # Find left and right edges where mask transitions from 0 to 255
        left_edge = None
        right_edge = None
        for x in range(w):
            if row[x] == 255 and left_edge is None:
                left_edge = x
            if row[x] == 0 and left_edge is not None and right_edge is None:
                right_edge = x - 1
                break
        # If right edge not found, check until end
        if left_edge is not None and right_edge is None:
            for x in range(w - 1, left_edge, -1):
                if row[x] == 255:
                    right_edge = x
                    break

        if left_edge is not None and right_edge is not None and right_edge > left_edge:
            # Calculate center of the line
            cx = (left_edge + right_edge) // 2
            # Calculate error from center of image
            error = cx - w / 2

            # PID control
            current_time = self.get_clock().now()
            dt = (current_time - self.last_time).nanoseconds * 1e-9
            if dt <= 0.0:
                dt = 1e-6

            self.integral_error += error * dt
            derivative = (error - self.prev_error) / dt

            # PID output
            angular_z = -(self.Kp * error +
                          self.Ki * self.integral_error +
                          self.Kd * derivative)

            # Clamp angular speed
            max_ang = 1.0 # rad/s
            angular_z = -float(np.clip(angular_z, -max_ang, max_ang))

            # Log for debugging
            self.get_logger().info(f"Left Edge: {left_edge}, Right Edge: {right_edge}, Center: {cx}")
            self.get_logger().info(f"Error: {error}, Angular Z: {angular_z}")
            self.get_logger().info(f"Integral Error: {self.integral_error}, Derivative: {derivative}")

            # Update state
            self.prev_error = error
            self.last_time = current_time

            # Set velocities
            self.twist.linear.x = -0.13  # Constant forward speed
            self.twist.angular.z = angular_z +0.25
            self.last_angular_z = angular_z
        else:
            # No valid line detected, continue turning
            self.twist.linear.x = -0.125  # Reduced speed
            self.twist.angular.z = self.last_angular_z if hasattr(self, 'last_angular_z') and abs(self.last_angular_z) > 1.0 else 1.0
            self.get_logger().info("No valid line detected, searching...")

        self.cmd_vel_pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = BlueLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()