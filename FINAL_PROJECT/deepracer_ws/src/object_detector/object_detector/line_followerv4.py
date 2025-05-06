import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data


class BlueLineFollower(Node):
    def __init__(self):
        super().__init__("blue_line_follower")

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, "/image_raw", self.image_callback, 10
        )
        self.stop_sign_sub = self.create_subscription(
            Bool, "/stop_sign_detected", self.stop_sign_callback, 10
        )
        self.distance_sub = self.create_subscription(
            Float32, "/stop_sign_distance", self.distance_callback, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.bridge = CvBridge()
        self.twist = Twist()

        # PID controller parameters
        self.Kp = 0.01445
        self.Ki = 0.0001
        self.Kd = 0.0003

        self.integral_error = 0.0
        self.prev_error = 0.0
        self.last_time = self.get_clock().now()

        # Stop sign detection state
        self.stop_sign_detected = False
        self.stop_sign_distance = float("inf")  # Default to large distance
        self.is_stopped = False
        self.stop_start_time = None
        # Ensure stop sign action only once
        self.stop_handled = False

        # School-zone & odom subscriptions
        self.red_zone_sub = self.create_subscription(
            Float32, "/school_distance", self.red_zone_callback, 10
        )
        self.green_zone_sub = self.create_subscription(
            Float32, "/clear_distance", self.green_zone_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile_sensor_data
        )

        # Zone distances
        self.red_zone_distance = float("inf")
        self.green_zone_distance = float("inf")
        # Drive speeds
        self.base_speed = -0.13
        self.slow_speed = -0.125
        # Green-zone state
        self.green_zone_active = False
        # Latch red zone slowdown
        self.red_zone_active = False
        self.green_zone_start_x = None
        # Current odom position
        self.current_x = 0.0
        self.current_y = 0.0

    def stop_sign_callback(self, msg):
        """Callback for stop sign detection flag."""
        # Only update detection if not already handled
        if not self.stop_handled:
            self.stop_sign_detected = msg.data
            self.get_logger().info(f"Stop sign detected: {self.stop_sign_detected}")

    def distance_callback(self, msg):
        """Callback for stop sign distance."""
        self.stop_sign_distance = msg.data
        self.get_logger().info(f"Stop sign distance: {self.stop_sign_distance} m")

    def red_zone_callback(self, msg):
        self.red_zone_distance = msg.data
        # On first entry into red zone (and not already slowed or in green)
        if msg.data < 1 and not self.red_zone_active and not self.green_zone_active:
            self.red_zone_active = True
            self.get_logger().info("Red zone entered: slow mode latched")

    def green_zone_callback(self, msg):
        # Always update green distance
        self.green_zone_distance = msg.data
        # On first entry into green zone, latch it and clear red-zone latch
        if msg.data < 1.5 and not self.green_zone_active:
            self.green_zone_active = True
            self.green_zone_start_x = self.current_x
            # self.red_zone_active = False
            self.get_logger().info("Green zone entered: cleared red-zone latch")

    def odom_callback(self, msg):
        # Track current x,y position
        pos = msg.pose.pose.position
        self.current_x = pos.x
        self.current_y = pos.y

    def image_callback(self, msg):
        # Check if robot is in stopped state due to stop sign
        current_time = self.get_clock().now()
        if self.is_stopped:
            # Check if 3 seconds have passed
            elapsed_time = (current_time - self.stop_start_time).nanoseconds * 1e-9
            if elapsed_time >= 3.5:
                self.is_stopped = False
                self.stop_start_time = None
                self.get_logger().info("Resuming line following after 3-second stop")
            else:
                # Remain stopped
                self.twist = Twist()
                self.cmd_vel_pub.publish(self.twist)
                return

        self.get_logger().info(f"red dist: {self.red_zone_distance}, green dist: {self.green_zone_distance}")
        # Check for stop sign within 15 cm (0.15 m)
        if self.stop_sign_detected and self.stop_sign_distance <= 1.5:
            self.get_logger().info("Stop sign within 15 cm, stopping for 3 seconds")
            self.is_stopped = True
            self.stop_handled = True
            self.stop_start_time = current_time
            self.twist = Twist()  # Stop the robot
            self.cmd_vel_pub.publish(self.twist)
            return

        # Determine speed based on school/green zone
        if self.green_zone_active:
            self.get_logger().info("green zone active")
            # still within green zone distance?
            dist = abs(self.current_x - self.green_zone_start_x)
            if dist < 4.572:
                self.get_logger().info("green zone active: slow speed")
                speed = self.slow_speed
            else:
                self.get_logger().info("green zone passed: resume speed")
                self.green_zone_active = False
                speed = self.base_speed
        elif self.red_zone_active:
            # latched red zone: always slow
            self.get_logger().info("Red zone active: latched slow speed")
            speed = self.slow_speed
        else:
            speed = self.base_speed


        # Proceed with line following
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Crop to lower region of interest (bottom 30% of the frame)
        h, w = cv_image.shape[:2]
        roi_y = int(h * 0.7)  # Updated to bottom 30%
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
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Initialize variables
        cx = None
        self.twist = Twist()

        # Scan the mask horizontally to find left and right edges
        scan_y = int(roi_h * 0.7)  # Scan at 70% down the ROI for stability
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
            dt = (current_time - self.last_time).nanoseconds * 1e-9
            if dt <= 0.0:
                dt = 0.01  # Use a reasonable minimum dt

            self.integral_error += error * dt
            # Clamp integral to prevent windup
            self.integral_error = float(np.clip(self.integral_error, -1000, 1000))
            derivative = (error - self.prev_error) / dt

            # PID output
            angular_z = -(
                self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative
            )

            # Clamp angular speed
            max_ang = 0.8  # rad/s
            angular_z = -float(np.clip(angular_z, -max_ang, max_ang))

            # Update state
            self.prev_error = error
            self.last_time = current_time

            # Set velocities
            self.twist.linear.x = speed
            self.twist.angular.z = angular_z
            self.last_angular_z = angular_z
        else:
            # No valid line detected, continue turning
            self.twist.linear.x = self.slow_speed  # Reduced speed
            self.twist.angular.z = (
                self.last_angular_z if hasattr(self, "last_angular_z") else 0.5
            )
            self.get_logger().info("No valid line detected, searching...")

        self.cmd_vel_pub.publish(self.twist)


def main(args=None):
    rclpy.init(args=args)
    node = BlueLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
