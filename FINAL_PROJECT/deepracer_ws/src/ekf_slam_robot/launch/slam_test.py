#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('ekf_slam_robot')  # Replace with your actual package name
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # SLAM Toolbox parameters file path
    slam_params_file = os.path.join(pkg_dir, 'config', 'slam_toolbox_params.yaml')
    
    # Define nodes
    
    # Static transform from base_link to laser
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_broadcaster',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser'],
        output='screen'
    )
    
    # Custom odometry node
    odometry_node = Node(
        package='ekf_slam_robot',
        executable='odometryv2',
        name='simple_odometry',
        parameters=[
            {'max_linear_vel': 0.5},
            {'max_steering_angle': 30.0},
            {'wheelbase': 0.25},
            {'use_encoders': True},
            {'ticks_per_meter': 1000.0},
        ]
    )
    
    # RPLIDAR node
    rplidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        name='rplidar_node',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 115200,
            'frame_id': 'laser',
            'angle_compensate': True,
            'scan_mode': 'Standard'
        }],
        output='screen'
    )
    
    # SLAM Toolbox node (direct configuration instead of including launch)
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[slam_params_file, {'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Diagnostic node
    diagnostic_node = Node(
        package='ekf_slam_robot',  # Use your actual package name
        executable='slam_diagnostics',  # You'll need to create this
        name='slam_diagnostics',
        output='screen'
    )
    
    # Information about parameter file status
    param_info = LogInfo(
        msg=["SLAM parameter file: ", slam_params_file]
    )
    
    # Return the launch description
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'),
        
        # Information messages
        param_info,
        
        # Nodes
        static_tf_node,
        odometry_node,
        rplidar_node,
        slam_toolbox_node,
        diagnostic_node
    ])