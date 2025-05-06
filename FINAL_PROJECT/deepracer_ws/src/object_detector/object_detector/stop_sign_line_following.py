import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist

INF = float('inf')

class StopSignLineFollower(Node):
    def __init__(self):
        super().__init__("stop_sign_line_follower")
        self.bridge = CvBridge()
        
        # Focal length for stop sign distance calculation
        self.fx = 551.29646
        self.get_logger().info(f"fx = {self.fx}.")

        # Stop sign width (meters)
        self.declare_parameter("sign_width", 0.1524)
        self.real_width = self.get_parameter("sign_width").get_parameter_value().double_value

        # Debug image window
        self.declare_parameter("show_debug_image", False)
        self.show_debug = self.get_parameter("show_debug_image").get_parameter_value().bool_value

        # Subscribe to camera image
        self.sub = self.create_subscription(Image, "/image_raw", self.image_callback, 1)

        # Publishers for stop sign detection
        self.stop_sign_pub = self.create_publisher(Bool, "/stop_sign_detected", 1)
        self.mask_pub = self.create_publisher(Image, "/stop_mask", 1)
        self.dist_pub = self.create_publisher(Float32, "/stop_sign_distance", 1)

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.twist = Twist()

        # State variables for stop sign stopping behavior
        self.stopping = False
        self.stop_start_time = None
        self.stop_duration = 3.0  # Stop for 3 seconds

        self.get_logger().info("StopSignLineFollower ready.")

    def image_callback(self, msg: Image):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # --- Stop Sign Detection ---
        # Convert to HSV for red octagon detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        distance = INF
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 8:
                found = True
                x, y, w, h = cv2.boundingRect(approx)
                if self.fx:
                    distance = (self.fx * self.real_width) / float(w)
                    self.get_logger().info(f"Stop sign distance: {distance:.2f} m")
                if self.show_debug:
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{distance:.2f} m",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        # Publish stop sign detection flag and distance
        self.stop_sign_pub.publish(Bool(data=found))
        dist_val = distance if (found and self.fx is not None) else INF
        self.dist_pub.publish(Float32(data=dist_val))

        # --- Handle Stop Sign Stopping Logic ---
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

        if found and not self.stopping:
            # Start stopping when a stop sign is detected
            self.stopping = True
            self.stop_start_time = current_time
            self.get_logger().info("Stop sign detected, stopping for 3 seconds.")

        if self.stopping:
            # Check if 3 seconds have passed
            if current_time - self.stop_start_time >= self.stop_duration:
                self.stopping = False
                self.stop_start_time = None
                self.get_logger().info("Stop duration complete, resuming line following.")
            # Stop the robot while in stopping phase
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
        else:
            # --- Line Following Logic ---
            # Convert to HSV for blue line detection
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Find contours in the blue mask
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if blue_contours:
                # Find largest blue contour
                largest_contour = max(blue_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)

                if M['m00'] > 0:
                    # Calculate centroid of the contour
                    cx = int(M['m10'] / M['m00'])

                    # Calculate error from center
                    error = cx - frame.shape[1] / 2

                    # Proportional control for steering
                    Kp = 0.002
                    angular_z = -Kp * error

                    # Set velocity commands
                    self.twist.linear.x = -0.06  # Constant forward velocity
                    # self.twist.angular.z = angular_z
                    self.twist.angular.z = 2.0

                    if self.show_debug:
                        cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)
                else:
                    # Stop if no valid centroid
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
            else:
                # Stop if no blue line detected
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0

        # Publish velocity commands
        self.cmd_vel_pub.publish(self.twist)

        # Display debug image if enabled
        if self.show_debug:
            cv2.imshow("StopSignLineFollower", frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = StopSignLineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()