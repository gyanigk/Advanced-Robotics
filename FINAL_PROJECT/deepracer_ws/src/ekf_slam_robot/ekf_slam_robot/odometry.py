#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import math
import tf_transformations

class SimpleOdometry(Node):
    def __init__(self):
        super().__init__('simple_odometry')
        
        # Robot parameters
        self.max_linear_vel = 0.4  # Max linear velocity (m/s), based on prior throttle range
        self.max_angular_vel = 1.0  # Max angular velocity (rad/s), approximate
        
        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.last_time = self.get_clock().now()
        
        # Subscribers and publishers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timer for updating odometry at 10 Hz
        self.timer = self.create_timer(0.1, self.update_odometry)
        
        self.get_logger().info('Simple odometry node started')

    def cmd_vel_callback(self, msg):
        """Handle incoming cmd_vel messages."""
        # Constrain velocities
        self.linear_vel = max(min(msg.linear.x, self.max_linear_vel), 0.0)
        self.angular_vel = max(min(msg.angular.z, self.max_angular_vel), -self.max_angular_vel)
        
        self.get_logger().debug(f'Received cmd_vel: linear={self.linear_vel:.2f} m/s, '
                               f'angular={self.angular_vel:.2f} rad/s')

    def update_odometry(self):
        """Update and publish odometry."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        # Update pose
        delta_theta = self.angular_vel * dt
        delta_x = self.linear_vel * math.cos(self.theta) * dt
        delta_y = self.linear_vel * math.sin(self.theta) * dt
        
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        self.theta = self.normalize_angle(self.theta)
        
        # Publish odometry and transform
        self.publish_odom(current_time)
        self.broadcast_transform(current_time)
        
        self.get_logger().debug(f'Published odometry: x={self.x:.2f}, y={self.y:.2f}, '
                               f'theta={self.theta:.2f}')

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def publish_odom(self, current_time):
        """Publish odometry message."""
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Set position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]
        
        # Set velocity
        odom.twist.twist.linear.x = self.linear_vel
        odom.twist.twist.angular.z = self.angular_vel
        
        self.odom_pub.publish(odom)

    def broadcast_transform(self, current_time):
        """Broadcast transform from odom to base_link."""
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down simple odometry node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()