import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32

INF = float('inf')


class SchoolZoneDetector(Node):
    def __init__(self):
        super().__init__("school_zone_detector")
        self.bridge = CvBridge()
        # focal length
        self.fx = 551.29646
        self.get_logger().info(f"fx = {self.fx}.")

        # stop sign width (meters)
        self.declare_parameter("sign_width", 0.1524)
        self.real_width = (
            self.get_parameter("sign_width").get_parameter_value().double_value
        )

        # debug image window
        self.declare_parameter("show_debug_image", False)
        self.show_debug = (
            self.get_parameter("show_debug_image").get_parameter_value().bool_value
        )

        # Subscribe
        self.sub = self.create_subscription(
            Image, "/image_raw", self.image_callback, 1
        )

        # Publish detection flag
        self.school_pub = self.create_publisher(Bool, "/school_zone_detected", 1)
        self.school_mask_pub = self.create_publisher(Image, "/school_mask", 1)
        self.clear_pub = self.create_publisher(Bool, "/all_clear_detected", 1)
        self.clear_mask_pub = self.create_publisher(Image, "/clear_mask", 1)
        # Publisher for distance
        self.school_dist_pub = self.create_publisher(Float32, "/school_distance", 1)
        self.clear_dist_pub = self.create_publisher(Float32, "/clear_distance", 1)
        self.get_logger().info("SchoolZoneDetector ready.")

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold red school‐zone sign (#ed0018)
        red_mask1 = cv2.inRange(hsv, (0, 60, 60), (20, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 60, 60), (179, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        red_mask_msg = self.bridge.cv2_to_imgmsg(red_mask, encoding='mono8')
        red_mask_msg.header = msg.header
        self.school_mask_pub.publish(red_mask_msg)

        # Threshold green all-clear sign (#cbfb04)
        green_mask = cv2.inRange(hsv, (25, 80, 80), (85, 255, 255))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        green_mask_msg = self.bridge.cv2_to_imgmsg(green_mask, encoding='mono8')
        green_mask_msg.header = msg.header
        self.clear_mask_pub.publish(green_mask_msg)

        # Helper to detect a sign color mask
        def detect_sign(mask):
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 500: 
                    continue
                peri   = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)

                if len(approx) == 5:
                    x,y,w,h = cv2.boundingRect(approx)
                    # distance via pinhole model
                    if self.fx:
                        dist = (self.fx * self.real_width) / float(w)
                        self.get_logger().info(f"distance : {dist}")
                    else:
                        dist = float('inf')
                    return True, dist, approx
            return False, float('inf'), None

        red_found, red_dist, red_poly   = detect_sign(red_mask)
        green_found, green_dist, green_poly = detect_sign(green_mask)
        self.get_logger().info(f"red : {red_found}, green: {green_found}")
        self.get_logger().info(f"school distance : {red_dist}")
        self.get_logger().info(f"clear distance : {green_dist}")

        self.school_pub.publish(Bool(data=red_found))
        self.clear_pub.publish(Bool(data=green_found))
        self.school_dist_pub.publish(Float32(data=red_dist))
        self.clear_dist_pub.publish(Float32(data=green_dist))

        if self.show_debug:
            cv2.imshow("School Zone", frame)
            cv2.waitKey(1)

    def info_callback(self, msg: CameraInfo):
        # grab focal length fx (pixels) once from camera_info
        if self.fx is None:
            self.fx = msg.k[0]
            self.get_logger().info(f"Camera fx set to {self.fx}")


def main(args=None):
    rclpy.init(args=args)
    node = SchoolZoneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
