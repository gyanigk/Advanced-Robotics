#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
import math
import tf_transformations
import numpy as np
from scipy.spatial import KDTree

class SimpleOdometry(Node):
    def __init__(self):
        super().__init__('simple_odometry')
        
        # Declare parameters
        self.declare_parameter('max_linear_vel', 0.85)
        self.declare_parameter('max_steering_angle', 45.0)  # in degrees
        self.declare_parameter('wheelbase', 0.16)
        self.declare_parameter('use_encoders', False)
        self.declare_parameter('ticks_per_meter', 1000.0)
        self.declare_parameter('lidar_weight', 1.0)  # Weight for LIDAR in fusion (0 to 1)
        
        # Robot parameters
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_steering_angle = math.radians(self.get_parameter('max_steering_angle').value)
        self.wheelbase = self.get_parameter('wheelbase').value
        self.use_encoders = self.get_parameter('use_encoders').value
        self.ticks_per_meter = self.get_parameter('ticks_per_meter').value
        self.lidar_weight = self.get_parameter('lidar_weight').value
        
        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.steering_angle = 0.0
        self.angular_velocity = 0.0
        self.last_time = self.get_clock().now()
        
        # Encoder state
        self.last_left_ticks = 0.0
        self.last_right_ticks = 0.0
        self.first_encoder_reading = True
        
        # LIDAR state
        self.last_scan = None
        self.last_scan_points = None
        self.first_scan = True
        
        # QoS profiles
        cmd_vel_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers and publishers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, cmd_vel_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, scan_qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', odom_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        if self.use_encoders:
            self.left_encoder_sub = self.create_subscription(
                Float32, '/left_encoder_ticks', self.left_encoder_callback, 10)
            self.right_encoder_sub = self.create_subscription(
                Float32, '/right_encoder_ticks', self.right_encoder_callback, 10)
        
        # Timer for updating odometry at 20 Hz
        self.timer = self.create_timer(0.05, self.update_odometry)
        
        self.get_logger().info('Simple odometry node with LIDAR started with parameters:')
        self.get_logger().info(f'  - max_linear_vel: {self.max_linear_vel} m/s')
        self.get_logger().info(f'  - max_steering_angle: {math.degrees(self.max_steering_angle)} degrees')
        self.get_logger().info(f'  - wheelbase: {self.wheelbase} m')
        self.get_logger().info(f'  - use_encoders: {self.use_encoders}')
        self.get_logger().info(f'  - lidar_weight: {self.lidar_weight}')
        if self.use_encoders:
            self.get_logger().info(f'  - ticks_per_meter: {self.ticks_per_meter}')

    def cmd_vel_callback(self, msg):
        """Handle incoming cmd_vel messages."""
        self.v = max(min(msg.linear.x, self.max_linear_vel), -self.max_linear_vel)
        desired_angular_vel = msg.angular.z
        if abs(self.v) > 1e-4:
            self.steering_angle = math.atan2(desired_angular_vel * self.wheelbase, self.v)
            self.steering_angle = max(min(self.steering_angle, self.max_steering_angle), -self.max_steering_angle)
        else:
            self.steering_angle = 0.0
        if abs(self.steering_angle) > 1e-4:
            self.angular_velocity = self.v / self.wheelbase * math.tan(self.steering_angle)
        else:
            self.angular_velocity = 0.0
        self.get_logger().debug(f'Received cmd_vel: linear={self.v:.2f} m/s, '
                              f'steering_angle={self.steering_angle:.2f} rad, '
                              f'angular_vel={self.angular_velocity:.2f} rad/s')

    def left_encoder_callback(self, msg):
        """Handle left encoder ticks."""
        if self.first_encoder_reading:
            self.last_left_ticks = msg.data
            self.first_encoder_reading = False
            return
        self.compute_velocity_from_encoders(left_ticks=msg.data)

    def right_encoder_callback(self, msg):
        """Handle right encoder ticks."""
        if self.first_encoder_reading:
            self.last_right_ticks = msg.data
            self.first_encoder_reading = False
            return
        self.compute_velocity_from_encoders(right_ticks=msg.data)

    def compute_velocity_from_encoders(self, left_ticks=None, right_ticks=None):
        """Compute velocity from encoder ticks."""
        if left_ticks is not None:
            delta_left = (left_ticks - self.last_left_ticks) / self.ticks_per_meter
            self.last_left_ticks = left_ticks
        else:
            delta_left = 0.0
        if right_ticks is not None:
            delta_right = (right_ticks - self.last_right_ticks) / self.ticks_per_meter
            self.last_right_ticks = right_ticks
        else:
            delta_right = 0.0
        dt = 0.05
        self.v = (delta_left + delta_right) / (2.0 * dt)
        if abs(self.steering_angle) > 1e-4:
            self.angular_velocity = self.v / self.wheelbase * math.tan(self.steering_angle)
        else:
            self.angular_velocity = 0.0
        self.get_logger().debug(f'Encoder-based velocity: linear={self.v:.2f} m/s, '
                              f'angular={self.angular_velocity:.2f} rad/s')

    def scan_callback(self, msg):
        """Process LIDAR scan data."""
        if self.first_scan:
            self.last_scan = msg
            self.last_scan_points = self.scan_to_points(msg)
            self.first_scan = False
            return
        current_points = self.scan_to_points(msg)
        if len(current_points) > 0 and len(self.last_scan_points) > 0:
            # Estimate pose change using ICP
            dx, dy, dtheta = self.estimate_pose_change(self.last_scan_points, current_points)
            # Update odometry with LIDAR data
            self.fuse_lidar_pose(dx, dy, dtheta)
        self.last_scan = msg
        self.last_scan_points = current_points

    def scan_to_points(self, scan):
        """Convert LaserScan to 2D points."""
        points = []
        for i, r in enumerate(scan.ranges):
            if r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append([x, y])
        return np.array(points)

    def estimate_pose_change(self, prev_points, curr_points):
        """Estimate pose change using point-to-point ICP."""
        # Simple ICP: find nearest neighbors and estimate transformation
        tree = KDTree(prev_points)
        distances, indices = tree.query(curr_points)
        
        # Filter matches based on distance threshold
        max_distance = 0.1  # meters
        valid = distances < max_distance
        if np.sum(valid) < 10:  # Minimum number of valid matches
            return 0.0, 0.0, 0.0
        
        src = curr_points[valid]
        dst = prev_points[indices[valid]]
        
        # Estimate transformation (rotation + translation)
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        
        # Compute rotation
        W = np.dot(src_centered.T, dst_centered)
        U, _, Vt = np.linalg.svd(W)
        R = np.dot(Vt.T, U.T)
        dtheta = math.atan2(R[1, 0], R[0, 0])
        
        # Compute translation
        t = dst_mean - np.dot(R, src_mean)
        dx, dy = t[0], t[1]
        
        return dx, dy, dtheta

    def fuse_lidar_pose(self, dx_lidar, dy_lidar, dtheta_lidar):
        """Fuse LIDAR pose estimate with wheel odometry."""
        # Wheel odometry pose change
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt < 0.001:
            return
        
        delta_theta_odom = self.angular_velocity * dt
        if abs(delta_theta_odom) > 1e-4:
            if abs(self.angular_velocity) > 1e-4:
                radius = self.v / self.angular_velocity
                dx_odom = radius * (math.sin(self.theta + delta_theta_odom) - math.sin(self.theta))
                dy_odom = radius * (math.cos(self.theta) - math.cos(self.theta + delta_theta_odom))
            else:
                dx_odom = self.v * math.cos(self.theta + delta_theta_odom/2) * dt
                dy_odom = self.v * math.sin(self.theta + delta_theta_odom/2) * dt
        else:
            dx_odom = self.v * math.cos(self.theta) * dt
            dy_odom = self.v * math.sin(self.theta) * dt
        
        # Simple weighted average for fusion
        w_lidar = self.lidar_weight
        w_odom = 1.0 - w_lidar
        dx = w_lidar * dx_lidar + w_odom * dx_odom
        dy = w_lidar * dy_lidar + w_odom * dy_odom
        dtheta = w_lidar * dtheta_lidar + w_odom * delta_theta_odom
        
        # Update pose
        self.x += dx
        self.y += dy
        self.theta += dtheta
        self.theta = self.normalize_angle(self.theta)

    def update_odometry(self):
        """Update and publish odometry."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt < 0.001:
            return
        self.last_time = current_time
        
        if not self.use_encoders:
            if abs(self.steering_angle) > 1e-4:
                self.angular_velocity = self.v / self.wheelbase * math.tan(self.steering_angle)
            else:
                self.angular_velocity = 0.0
        
        # Publish odometry and transform
        self.publish_odom(current_time)
        self.broadcast_transform(current_time)
        
        self.get_logger().debug(f'Updated odometry: x={self.x:.2f}, y={self.y:.2f}, '
                              f'theta={self.theta:.2f}, dt={dt:.3f}s')

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def publish_odom(self, current_time):
        """Publish odometry message."""
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]
        
        # Reduced covariance due to LIDAR fusion
        pose_covariance = np.diag([0.05, 0.05, 0.0, 0.0, 0.0, 0.05])
        odom.pose.covariance = pose_covariance.flatten().tolist()
        
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.angular_velocity
        
        twist_covariance = np.diag([0.02, 0.0, 0.0, 0.0, 0.0, 0.02])
        odom.twist.covariance = twist_covariance.flatten().tolist()
        
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