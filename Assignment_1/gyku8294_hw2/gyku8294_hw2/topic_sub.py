import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import time
import numpy as np
import matplotlib.pyplot as plt

class LatencySubscriber(Node):
    def __init__(self):
        super().__init__('latency_subscriber')
        self.subscription = self.create_subscription(Header, 'latency_topic', self.callback, 10)
        self.latencies = []
        self.msg_count = 0

    def callback(self, msg):
        receive_time = time.time()
        publish_time = msg.stamp.sec + msg.stamp.nanosec / 1e9
        latency = receive_time - publish_time
        self.latencies.append(latency)
        self.msg_count += 1

        if self.msg_count >= 400:
            self.generate_histogram()
            rclpy.shutdown()

    def generate_histogram(self):
        plt.hist(self.latencies, bins=30, alpha=0.75, color='red', edgecolor='black')
        plt.xlabel("Latency (s)")
        plt.ylabel("Frequency")
        plt.title("ROS 2 Topic Latencies")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

def main():
    rclpy.init()
    node = LatencySubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
