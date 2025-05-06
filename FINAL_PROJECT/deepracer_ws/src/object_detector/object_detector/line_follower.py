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
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.bridge = CvBridge()
        self.twist = Twist()

        # Proportional control parameters
        self.Kp = 0.015       # Proportional gain for normal conditions
        self.Kp_turn = 0.025  # Proportional gain for L-turns
        self.angular_z_bias = 0.25  # Bias to compensate for wheel misalignment

        # State variables
        self.last_time = self.get_clock().now()
        self.last_cx = None
        self.last_cy = None
        self.last_line_time = self.last_time
        self.last_error_direction = 1.0  # For search direction
        self.reverse_start_time = None
        self.prev_cy = None  # For L-turn detection

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image (no ROI cropping)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = cv_image.shape[:2]

        # Convert to HSV color space and apply blur
        hsv = cv2.GaussianBlur(cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV), (5, 5), 0)
        
        # Define blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Select largest contour by area
        cx, cy = None, None
        max_area = 100  # Minimum area to reject noise
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > max_area:
                # Compute centroid
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

        # Detect L-turns based on centroid movement
        is_l_turn = False
        if cx is not None and cy is not None and self.prev_cy is not None:
            cy_change = abs(cy - self.prev_cy) / h  # Normalized change in y
            is_l_turn = cy_change > 0.1  # Threshold for rapid vertical shift (L-turn)

        # Publish debug image
        debug_image = cv_image.copy()
        if cx is not None and cy is not None:
            cv2.circle(debug_image, (cx, cy), 5, (0, 255, 0), -1)  # Centroid
            if contours:
                cv2.drawContours(debug_image, [largest_contour], -1, (255, 0, 0), 2)  # Contour
            if is_l_turn:
                cv2.putText(debug_image, "L-TURN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

        self.twist = Twist()
        current_time = self.get_clock().now()
        time_since_line = (current_time - self.last_line_time).nanoseconds * 1e-9

        if cx is not None:
            # Calculate error
            error = cx - w / 2

            # Proportional control with bias for straight movement
            kp = self.Kp_turn if is_l_turn else self.Kp
            angular_z = -kp * error
            angular_limit = 1.5 if is_l_turn else 0.5  # Smooth turns unless L-turn
            if abs(error) < 10:  # Apply bias when centered
                angular_z = self.angular_z_bias
                self.get_logger().info(f"Applying angular_z_bias: {self.angular_z_bias}")
            angular_z = float(np.clip(angular_z, -angular_limit, angular_limit))

            # Dynamic linear speed with braking
            if abs(error) > w / 4 or is_l_turn:  # Brake for large errors or L-turns
                self.twist.linear.x = 0.0
                self.get_logger().info("Braking due to large error or L-turn")
            else:
                self.twist.linear.x = -0.15 * (1 - abs(error) / (w / 2))
                self.twist.linear.x = max(self.twist.linear.x, -0.13)
            self.twist.angular.z = angular_z

            # Update state
            self.last_cx = cx
            self.last_cy = cy
            self.prev_cy = cy
            self.last_line_time = current_time
            self.last_error_direction = np.sign(error) if error != 0 else self.last_error_direction
            self.reverse_start_time = None

            self.get_logger().info(f"Error: {error:.2f}, Kp*Error: {kp*error:.4f}, L-Turn: {is_l_turn}, Angular Z: {angular_z:.4f}")
        else:
            if time_since_line < 0.8 and hasattr(self, 'last_cx'):
                # Use last known center for brief line loss
                cx = self.last_cx
                error = cx - w / 2
                angular_z = -self.Kp * error
                angular_z = float(np.clip(angular_z, -0.85, 0.85))  # Smooth for non-L-turn
                self.twist.linear.x = -0.15
                self.twist.angular.z = angular_z
                self.get_logger().info("Using last known center")
            elif time_since_line > 1.0 and self.reverse_start_time is None:
                # Initiate reverse maneuver
                self.reverse_start_time = current_time
                self.twist.linear.x = 0.15  # Reverse
                self.twist.angular.z = 0.8 * self.last_error_direction
                self.get_logger().info("Reversing to find line")
            elif self.reverse_start_time and (current_time - self.reverse_start_time).nanoseconds * 1e-9 < 1.0:
                # Continue reversing
                self.twist.linear.x = 0.15
                self.twist.angular.z = 0.8 * self.last_error_direction
            else:
                # Search mode with braking
                self.twist.linear.x = 0.0 # Brake while searching
                self.twist.angular.z = 0.8 * self.last_error_direction
                self.get_logger().info("Searching for line")

        self.last_time = current_time
        self.cmd_vel_pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = BlueLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()