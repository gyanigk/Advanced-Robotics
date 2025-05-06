import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32

INF = float('inf')


class StopSignDetector(Node):
    def __init__(self):
        super().__init__("stop_sign_detector")
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
        self.pub = self.create_publisher(Bool, "/stop_sign_detected", 1)
        self.mask_pub = self.create_publisher(Image, "/mask", 1)
        # Publisher for distance
        self.dist_pub = self.create_publisher(Float32, "/stop_sign_distance", 1)
        self.get_logger().info("StopSignDetector ready.")


    def image_callback(self, msg: Image):
        # convert ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # red octagon detection: stop sign
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
            # self.get_logger().info(f"approx: {len(approx)}")
            if len(approx) == 8:
                found = True
                self.get_logger().info(f"found!")
                x, y, w, h = cv2.boundingRect(approx)
                # self.get_logger().info(f"w : {w}")
                distance = (self.fx * self.real_width) / float(w)
                self.get_logger().info(f"distance : {distance}")
                break

        # publish detection flag
        self.pub.publish(Bool(data=found))
        # publish distance
        dist_val = distance if (found and self.fx is not None) else INF
        self.dist_pub.publish(Float32(data=dist_val))

        if self.show_debug:
            cv2.imshow("StopSign Detection", frame)
            cv2.waitKey(1)

    def info_callback(self, msg: CameraInfo):
        # grab focal length fx (pixels) once from camera_info
        if self.fx is None:
            self.fx = msg.k[0]
            self.get_logger().info(f"Camera fx set to {self.fx}")
        


def main(args=None):
    rclpy.init(args=args)
    node = StopSignDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

