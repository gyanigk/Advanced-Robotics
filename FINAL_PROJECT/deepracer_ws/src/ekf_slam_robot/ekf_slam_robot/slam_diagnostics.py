#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
import time

class SlamDiagnostic(Node):
    def __init__(self):
        super().__init__('slam_diagnostic')
        
        # Setup QoS profiles to match those used by the system
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Setup subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, reliable_qos)
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, sensor_qos)
        
        self.tf_sub = self.create_subscription(
            TFMessage, '/tf', self.tf_callback, sensor_qos)
        
        # Diagnostic data
        self.last_odom_time = None
        self.last_scan_time = None
        self.last_tf_time = None
        self.last_position = [0.0, 0.0]
        self.odom_count = 0
        self.scan_count = 0
        self.tf_count = 0
        
        # Diagnostic timer (print status every 5 seconds)
        self.timer = self.create_timer(5.0, self.diagnostic_callback)
        
        self.get_logger().info('SLAM Diagnostic Node started')
    
    def odom_callback(self, msg):
        current_time = self.get_clock().now()
        if self.last_odom_time is not None:
            time_diff = (current_time - self.last_odom_time).nanoseconds / 1e9
            
            # Check position change
            new_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            position_diff = ((new_position[0] - self.last_position[0])**2 + 
                            (new_position[1] - self.last_position[1])**2)**0.5
            
            # Log significant movement
            if position_diff > 0.05:  # 5cm threshold
                self.get_logger().info(f'Robot moved: {position_diff:.3f}m')
                self.last_position = new_position
        
        self.last_odom_time = current_time
        self.odom_count += 1
    
    def scan_callback(self, msg):
        self.last_scan_time = self.get_clock().now()
        self.scan_count += 1
        
        # Check scan quality occasionally
        if self.scan_count % 20 == 0:
            valid_ranges = sum(1 for r in msg.ranges if r > 0.1 and r < msg.range_max)
            self.get_logger().info(f'Scan quality: {valid_ranges}/{len(msg.ranges)} valid points')
    
    def tf_callback(self, msg):
        self.last_tf_time = self.get_clock().now()
        self.tf_count += 1
        
        # Look for specific transforms of interest
        for transform in msg.transforms:
            if (transform.header.frame_id == 'odom' and 
                transform.child_frame_id == 'base_link'):
                # Found an odom->base_link transform
                self.get_logger().debug('Received odom->base_link transform')
            
            if (transform.header.frame_id == 'map' and 
                transform.child_frame_id == 'odom'):
                # Found a map->odom transform (from SLAM)
                self.get_logger().info('Received map->odom transform - SLAM is working!')
    
    def diagnostic_callback(self):
        current_time = self.get_clock().now()
        self.get_logger().info('--- SLAM Diagnostic Report ---')
        
        # Check if we're receiving data
        if self.last_odom_time is not None:
            time_diff = (current_time - self.last_odom_time).nanoseconds / 1e9
            self.get_logger().info(f'Odometry: {self.odom_count} msgs, last received {time_diff:.1f}s ago')
        else:
            self.get_logger().warn('No odometry messages received!')
        
        if self.last_scan_time is not None:
            time_diff = (current_time - self.last_scan_time).nanoseconds / 1e9
            self.get_logger().info(f'Laser Scan: {self.scan_count} msgs, last received {time_diff:.1f}s ago')
        else:
            self.get_logger().warn('No laser scan messages received!')
            
        if self.last_tf_time is not None:
            time_diff = (current_time - self.last_tf_time).nanoseconds / 1e9
            self.get_logger().info(f'Transforms: {self.tf_count} msgs, last received {time_diff:.1f}s ago')
        else:
            self.get_logger().warn('No transform messages received!')
        
        self.get_logger().info('-----------------------------')

def main(args=None):
    rclpy.init(args=args)
    node = SlamDiagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down SLAM diagnostic node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()