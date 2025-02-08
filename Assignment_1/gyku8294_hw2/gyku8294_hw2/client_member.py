import sys

# from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node
from gyku8294_service.srv import Gyku8294Service
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(Gyku8294Service, 'gyku8294_service')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Gyku8294Service.Request()

    def send_request(self, input_string):
        self.req.msg_input = input_string 
        start_time = time.time()

        recall_value = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, recall_value)
        total_time = time.time() - start_time
        service_time = recall_value.result().time_taken
        return total_time, service_time


def main():
    rclpy.init()
    minimal_client = MinimalClientAsync()

    service_latencies = []

    for i in range(400):
        curr_total_time, curr_service_time = minimal_client.send_request("This is Gyanig!")
        service_latencies.append(curr_total_time-curr_service_time)

    #plot
    # plt.hist(service_latencies, bins=30, edgecolor='black')
    # plt.xlabel('Latency (s)')
    # plt.ylabel('Frequency')
    # plt.title('Service Call Latency Distribution')
    # plt.grid(True)
    # plt.savefig('service_latency_histogram.png')
    # plt.show()
    # minimal_client.get_logger().info('Histogram_generated!')

    # Create a larger figure
    plt.figure(figsize=(10, 6))

    # Plot histogram with improved styling
    plt.hist(service_latencies, bins=30, edgecolor='black', alpha=0.75, linewidth=1.2)

    # Labels and title with enhanced font sizes
    plt.xlabel('Latency (s)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Service Call Latency Distribution', fontsize=16, fontweight='bold')

    # Enable grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save and show the plot
    plt.savefig('service_latency_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Log completion
    minimal_client.get_logger().info('Histogram generated!')

    # future = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    # rclpy.spin_until_future_complete(minimal_client, future)
    # response = future.result()
    # minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()