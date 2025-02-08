import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import time

class LatencyPublisher(Node):
    def __init__(self):
        super().__init__('latency_publisher')
        self.publisher = self.create_publisher(Header, 'latency_topic', 10)
        self.timer = self.create_timer(0.1, self.publish_timestamp)

    def publish_timestamp(self):
        msg = Header()
        msg.stamp.sec = int(time.time())
        msg.stamp.nanosec = int((time.time() % 1) * 1e9)
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = LatencyPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
