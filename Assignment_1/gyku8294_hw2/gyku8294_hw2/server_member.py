# from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node
from gyku8294_service.srv import Gyku8294Service
import time


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        # self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
        self.srv = self.create_service(Gyku8294Service,'gyku8294_service',self.reverse_callback)
        self.get_logger().info("gyku8294 Server Ready...")
    
    def reverse_callback(self, request, response):
        # response.sum = request.a + request.b
        # self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        start_time = time.time()
        response.msg_output = request.msg_input[::-1] # reversing the text directly
        response.time_taken = time.time() - start_time # compute the only execution time
        return response


def main():
    rclpy.init()
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()