#!/usr/bin/python3

import numpy as np
from easydict import EasyDict as edict
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import DBSCAN
from .util import quaternion_to_yaw, get_center_radius, angle_between_yaw
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2
from typing import List, Tuple

def quaternion_from_yaw(yaw: float) -> Quaternion:
    """Convert yaw angle to quaternion (assuming 2D rotation)."""
    q = Quaternion()
    q.w = np.cos(yaw / 2.0)
    q.z = np.sin(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q

class SlamEkf(Node):
    # Constants
    DBSCAN_EPS = 0.2
    SLAM_UPDATE_RATE = 0.01
    LANDMARK_DISTANCE_THRESHOLD = 1.0
    TIMEOUT_DURATION = 0.21
    PREDICTION_RATE = 0.2
    WHEELBASE = 0.175  # Distance between front and back wheels in meters
    CMD_VEL_TIMEOUT = 1.0  # Warn if no /cmd_vel messages for 1 second

    def __init__(self):
        super().__init__('ekf_slam')

        # Declare parameters
        self.declare_parameter('fixed_frame', 'map')
        self.declare_parameter('alpha', [0.005, 0.005, 0.1, 0.1])
        self.declare_parameter('beta', [0.01, 0.001])
        self.declare_parameter('landmark_marker_size', 0.2)
        self.declare_parameter('landmark_marker_color_r', 1.0)
        self.declare_parameter('landmark_marker_color_g', 0.0)
        self.declare_parameter('landmark_marker_color_b', 0.0)
        self.declare_parameter('landmark_marker_color_a', 1.0)

        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.marker_size = self.get_parameter('landmark_marker_size').value
        self.marker_color = (
            self.get_parameter('landmark_marker_color_r').value,
            self.get_parameter('landmark_marker_color_g').value,
            self.get_parameter('landmark_marker_color_b').value,
            self.get_parameter('landmark_marker_color_a').value
        )

        # Initialize SLAM components
        self.dbscan = DBSCAN(eps=self.DBSCAN_EPS, min_samples=1)
        self.mu = np.zeros((3, 1))  # Robot pose [x, y, yaw]
        self.sigma = np.zeros((3, 3))
        self.landmark_count = 0
        self.cmd_vel = []  # Store [v, w, timestamp] from /cmd_vel
        self.robot_pose_odom = [[0.0, 0.0, 0.0]]  # Integrated odometry [x, y, yaw]
        self.landmark_measurements = edict(data=None, timestamp=None)
        self.lidar_pts_fixedframe = None
        self.slam_last_ts = None
        self.last_cmd_vel_ts = None  # Track last /cmd_vel message time

        # Store trajectory for Path message
        self.trajectory = []

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )

        # Publishers for RViz
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/slam/pose', 10
        )
        self.landmark_pub = self.create_publisher(
            MarkerArray, '/slam/landmarks', 10
        )
        self.trajectory_pub = self.create_publisher(
            Path, '/slam/trajectory', 10
        )
        self.lidar_pub = self.create_publisher(
            PointCloud2, '/slam/lidar_points', 10
        )

        # Timer for SLAM updates
        cb_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(
            self.SLAM_UPDATE_RATE, self.slam, callback_group=cb_group
        )

    @property
    def robot_state(self) -> np.ndarray:
        return self.mu[:3]

    def update_robot_state(self, new_state: np.ndarray) -> None:
        self.mu[:3] = new_state

    @property
    def landmark_state(self) -> np.ndarray:
        return self.mu[3:].reshape((-1, 2))

    @property
    def sigma_r(self) -> np.ndarray:
        return self.sigma[:3, :3]

    @property
    def sigma_rm(self) -> np.ndarray:
        return self.sigma[:3, 3:]

    @property
    def sigma_mr(self) -> np.ndarray:
        return self.sigma[3:, :3]

    @property
    def sigma_m(self) -> np.ndarray:
        return self.sigma[3:, 3:]

    def cmd_vel_callback(self, msg: Twist) -> None:
        """Callback for /cmd_vel topic. Store velocity commands and integrate odometry."""
        v = msg.linear.x  # Linear velocity (m/s)
        w = msg.angular.z  # Angular velocity (rad/s)
        if np.isnan(v) or np.isnan(w):
            self.get_logger().warn("Received NaN in /cmd_vel, skipping.")
            return
        timestamp = self.get_curr_time()
        self.cmd_vel.append([v, w, timestamp])
        self.last_cmd_vel_ts = timestamp

        # Integrate odometry
        if len(self.cmd_vel) >= 2:
            prev_v, prev_w, prev_t = self.cmd_vel[-2]
            dt = timestamp - prev_t
            if dt <= 0:
                self.get_logger().warn("Invalid timestamp in /cmd_vel, skipping.")
                return
            x, y, yaw = self.robot_pose_odom[-1]
            # Assume constant velocity over dt
            x_new = x + v * dt * np.cos(yaw)
            y_new = y + v * dt * np.sin(yaw)
            yaw_new = yaw + w * dt
            self.robot_pose_odom.append([x_new, y_new, yaw_new])

        # Clear old velocity commands to prevent memory buildup
        while len(self.cmd_vel) > 10:  # Keep last 10 commands
            self.cmd_vel.pop(0)

    def scan_callback(self, msg: LaserScan) -> None:
        if not msg.ranges or len(msg.ranges) < 1:
            self.get_logger().warn("Empty or invalid LiDAR scan received.")
            return

        nscan = round((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1
        angles = np.linspace(msg.angle_min, msg.angle_max, nscan, endpoint=True)
        rng = np.array(msg.ranges)
        valid = (rng >= msg.range_min) & (rng <= msg.range_max) & (~np.isnan(rng))
        if not np.any(valid):
            self.get_logger().warn("No valid LiDAR points in scan.")
            return

        x, y, yaw = self.robot_state[:, 0]
        pt_x = x + rng[valid] * np.cos(yaw + angles[valid])
        pt_y = y + rng[valid] * np.sin(yaw + angles[valid])
        pts = np.hstack([pt_x[:, np.newaxis], pt_y[:, np.newaxis]])
        self.lidar_pts_fixedframe = pts

        pts_clu = self.cluster_pts(pts)
        all_center = []
        for clu in pts_clu:
            center, r = get_center_radius(clu)
            if center is not None:
                all_center.append(center)
        if len(all_center) == 0:
            return
        all_center = np.array(all_center)

        data = []
        for center in all_center:
            dx = center[0] - x
            dy = center[1] - y
            angle = angle_between_yaw(yaw, np.arctan2(dy, dx))
            data.append([np.sqrt(dx**2 + dy**2), angle])
        data = np.array(data)
        self.landmark_measurements.data = data
        self.landmark_measurements.timestamp = self.get_curr_time()

    def cluster_pts(self, pts: np.ndarray) -> List[np.ndarray]:
        if len(pts) == 0:
            return []
        valid_pts = pts[~np.any(np.isnan(pts) | np.isinf(pts), axis=1)]
        if len(valid_pts) == 0:
            return []
        clu = self.dbscan.fit(valid_pts)
        labels = clu.labels_
        nclu = np.max(labels) + 1
        return [valid_pts[labels == i] for i in range(nclu)]

    def get_curr_time(self) -> float:
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        return float(sec) + float(nanosec) / 1.0e9

    def slam_timing(self) -> Tuple[bool, bool]:
        curr_ts = self.get_curr_time()
        if self.slam_last_ts is None:
            self.slam_last_ts = curr_ts

        pred_only = True
        proceed = True
        use_own_timing = True
        landmark_measurements = self.landmark_measurements
        if landmark_measurements.timestamp is not None:
            if landmark_measurements.timestamp > self.slam_last_ts + 1e-3:
                pred_only = False
                use_own_timing = False
                self.slam_last_ts = landmark_measurements.timestamp
            else:
                if curr_ts >= self.slam_last_ts + self.TIMEOUT_DURATION:
                    use_own_timing = True
                else:
                    proceed = False
                    return proceed, pred_only
        else:
            use_own_timing = True

        if use_own_timing:
            if curr_ts - self.slam_last_ts >= self.PREDICTION_RATE:
                pred_only = True
                self.slam_last_ts = curr_ts
            else:
                proceed = False
                return proceed, pred_only
        return proceed, pred_only

    def odom_to_control(self) -> List[float]:
        """Compute control inputs [rot1, tr, rot2] from integrated odometry."""
        if len(self.robot_pose_odom) < 2:
            return [0.0, 0.0, 0.0]  # No movement if insufficient data

        x0, y0, yaw0 = self.robot_pose_odom[0]
        x1, y1, yaw1 = self.robot_pose_odom[-1]

        # Pop all poses except the latest
        for _ in range(len(self.robot_pose_odom) - 1):
            self.robot_pose_odom.pop(0)

        tr = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if tr < 0.05:
            yaw0_ = yaw0
        else:
            yaw0_ = np.arctan2(y1 - y0, x1 - x0)
        rot1 = angle_between_yaw(yaw1=yaw0, yaw2=yaw0_)
        rot2 = angle_between_yaw(yaw1=yaw0_, yaw2=yaw1)
        return [rot1, tr, rot2]

    def motion_jacobian(self, x: np.ndarray, u: List[float]) -> np.ndarray:
        yaw = x[-1, 0]
        rot1, tr, rot2 = u
        phi = yaw + rot1
        return np.array([
            [1.0, 0.0, -tr * np.sin(phi)],
            [0.0, 1.0, tr * np.cos(phi)],
            [0.0, 0.0, 1.0]
        ])

    def noise_jacobian(self, x: np.ndarray, u: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        rot1, tr, rot2 = u
        yaw = x[-1, 0]
        Rn = np.zeros((3, 3))
        a1, a2, a3, a4 = self.alpha
        Rn[0, 0] = a1 * rot1**2 + a2 * tr**2
        Rn[1, 1] = a3 * (rot1**2 + rot2**2) + a4 * tr**2
        Rn[2, 2] = a1 * rot2**2 + a2 * tr**2
        phi = yaw + rot1
        s = np.sin(phi)
        c = np.cos(phi)
        Jn = np.array([
            [-tr * s, c, 0.0],
            [tr * c, s, 0.0],
            [1.0, 0.0, 1.0]
        ])
        return Jn, Rn

    def motion_model(self, x: np.ndarray, u: List[float]) -> np.ndarray:
        x, y, yaw = x[:, 0]
        rot1, tr, rot2 = u
        phi = yaw + rot1
        return np.array([
            [x + tr * np.cos(phi)],
            [y + tr * np.sin(phi)],
            [yaw + rot1 + rot2]
        ])

    def compute_cov_pred(self, J_motion: np.ndarray, J_noise: np.ndarray, Rn: np.ndarray) -> None:
        tmp1 = np.linalg.multi_dot((J_motion, self.sigma_r, J_motion.T))
        tmp2 = np.linalg.multi_dot((J_noise, Rn, J_noise.T))
        self.sigma[:3, :3] = tmp1 + tmp2
        if self.sigma.shape[0] > 3:
            tmp = np.linalg.multi_dot((self.sigma_mr, J_motion.T))
            self.sigma[3:, :3] = tmp
            self.sigma[:3, 3:] = tmp.T

    def convert_to_fixed_frame(self, landmark_measurements: np.ndarray) -> np.ndarray:
        x, y, yaw = self.robot_state[:, 0]
        landmark_xy = []
        for p in landmark_measurements:
            x_ = x + p[0] * np.cos(yaw + p[1])
            y_ = y + p[0] * np.sin(yaw + p[1])
            landmark_xy.append([x_, y_])
        return np.array(landmark_xy)

    def find_association(self, l: np.ndarray) -> int:
        for j, ls in enumerate(self.landmark_state):
            if np.linalg.norm(ls - l) < self.LANDMARK_DISTANCE_THRESHOLD:
                return j
        return -1

    def sensor_cov(self, r: float) -> np.ndarray:
        b1, b2 = self.beta
        return np.array([[b1 * r, 0.0], [0.0, b2]])

    def initialize_landmark(self, l_xy: np.ndarray, measurement: np.ndarray) -> None:
        rng, bearing = measurement
        self.mu = np.vstack((self.mu, l_xy.reshape((2, 1))))
        x, y, yaw = self.robot_state[:, 0]
        t = yaw + bearing
        c = np.cos(t)
        s = np.sin(t)
        J1 = np.array([[1.0, 0.0, -rng * s], [0.0, 1.0, rng * c]])
        J2 = np.array([[c, -rng * s], [s, rng * c]])
        Rs = self.sensor_cov(rng)
        cov_ll = np.linalg.multi_dot((J1, self.sigma_r, J1.T)) + np.linalg.multi_dot((J2, Rs, J2.T))
        cov_lx = np.linalg.multi_dot((J1, self.sigma[:3, :]))
        self.sigma = np.vstack((self.sigma, cov_lx))
        self.sigma = np.hstack((self.sigma, np.vstack((cov_lx.T, cov_ll))))
        self.landmark_count += 1
        self.get_logger().info(
            f"New landmark added at ({l_xy[0]:.02f}, {l_xy[1]:.02f}). "
            f"Total landmarks: {self.landmark_count}"
        )

    def compute_obs(self, landmark_ind: int, rng: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        j = landmark_ind
        indices = [0, 1, 2, 3 + 2 * j, 3 + 2 * j + 1]
        cov_ = self.sigma[np.ix_(indices, indices)]
        xr, yr, yaw = self.robot_state[:, 0]
        xl, yl = self.landmark_state[j, :]
        dx, dy = xl - xr, yl - yr
        rho = dx**2 + dy**2
        rho_inv = 1.0 / rho
        rho_inv_sqrt = np.sqrt(rho_inv)
        c = np.cos(yaw)
        s = np.sin(yaw)
        dx_r = c * dx + s * dy
        dy_r = -s * dx + c * dy
        rho_x = -dy_r / (dx_r**2 + dy_r**2)
        rho_y = dx_r / (dx_r**2 + dy_r**2)
        h00 = -rho_inv_sqrt * dx
        h01 = -rho_inv_sqrt * dy
        h10 = rho_x * (-c) + rho_y * s
        h11 = rho_x * (-s) + rho_y * (-c)
        h12 = rho_x * dy_r + rho_y * (-dx_r)
        H = np.array([
            [h00, h01, 0.0, -h00, -h01],
            [h10, h11, h12, -h10, -h11]
        ])
        s = np.sin(yaw)
        c = np.cos(yaw)
        R = np.array([[c, -s], [s, c]])
        xl_r, yl_r = R.T.dot(np.array([dx, dy])[:, np.newaxis]).flatten()
        angle_pred = np.arctan2(yl_r, xl_r)
        Z = np.linalg.multi_dot((H, cov_, H.T)) + self.sensor_cov(rng)
        z_pred = np.array([np.sqrt(rho), angle_pred])
        return H, Z, z_pred

    def publish_rviz_visualizations(self) -> None:
        timestamp = self.get_clock().now().to_msg()

        # Publish robot pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.fixed_frame
        pose_msg.pose.pose.position.x = float(self.robot_state[0, 0])
        pose_msg.pose.pose.position.y = float(self.robot_state[1, 0])
        pose_msg.pose.pose.position.z = 0.0
        pose_msg.pose.pose.orientation = quaternion_from_yaw(float(self.robot_state[2, 0]))

        # Convert 3x3 covariance [x, y, yaw] to 6x6 [x, y, z, roll, pitch, yaw]
        cov_6x6 = [0.0] * 36  # Initialize 6x6 covariance matrix as flat list
        sigma_r_flat = self.sigma_r.flatten()
        # Map x, y, yaw covariances
        cov_6x6[0] = sigma_r_flat[0]   # x variance
        cov_6x6[1] = sigma_r_flat[1]   # xy covariance
        cov_6x6[5] = sigma_r_flat[2]   # x-yaw covariance
        cov_6x6[6] = sigma_r_flat[3]   # yx covariance
        cov_6x6[7] = sigma_r_flat[4]   # y variance
        cov_6x6[11] = sigma_r_flat[5]  # y-yaw covariance
        cov_6x6[30] = sigma_r_flat[6]  # yaw-x covariance
        cov_6x6[31] = sigma_r_flat[7]  # yaw-y covariance
        cov_6x6[35] = sigma_r_flat[8]  # yaw variance
        # Set small variance for unused dimensions (z, roll, pitch)
        cov_6x6[14] = 1e-6  # z variance
        cov_6x6[21] = 1e-6  # roll variance
        cov_6x6[28] = 1e-6  # pitch variance
        pose_msg.pose.covariance = cov_6x6
        self.pose_pub.publish(pose_msg)

        # Update and publish trajectory
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = timestamp
        pose_stamped.header.frame_id = self.fixed_frame
        pose_stamped.pose.position.x = float(self.robot_state[0, 0])
        pose_stamped.pose.position.y = float(self.robot_state[1, 0])
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation = quaternion_from_yaw(float(self.robot_state[2, 0]))
        self.trajectory.append(pose_stamped)
        path_msg = Path()
        path_msg.header.stamp = timestamp
        path_msg.header.frame_id = self.fixed_frame
        path_msg.poses = self.trajectory
        self.trajectory_pub.publish(path_msg)

        # Publish landmarks
        marker_array = MarkerArray()
        for i, landmark in enumerate(self.landmark_state):
            marker = Marker()
            marker.header.stamp = timestamp
            marker.header.frame_id = self.fixed_frame
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(landmark[0])
            marker.pose.position.y = float(landmark[1])
            marker.pose.position.z = 0.0
            marker.scale.x = marker.scale.y = marker.scale.z = self.marker_size
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.marker_color
            marker_array.markers.append(marker)
        self.landmark_pub.publish(marker_array)

        # Publish LiDAR points
        if self.lidar_pts_fixedframe is not None and len(self.lidar_pts_fixedframe) > 0:
            points = np.hstack([
                self.lidar_pts_fixedframe,
                np.zeros((self.lidar_pts_fixedframe.shape[0], 1))  # Add z=0
            ])
            # Optional: Downsample if too many points
            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                points = points[indices]
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            point_cloud = pc2.create_cloud(
                header=std_msgs.msg.Header(stamp=timestamp, frame_id=self.fixed_frame),
                fields=fields,
                points=points
            )
            self.lidar_pub.publish(point_cloud)

    def slam(self) -> None:
        curr_ts = self.get_curr_time()
        proceed, pred_only = self.slam_timing()

        # Check for /cmd_vel timeout
        if self.last_cmd_vel_ts is None or (curr_ts - self.last_cmd_vel_ts > self.CMD_VEL_TIMEOUT):
            self.get_logger().warn("No recent /cmd_vel messages received.")
            self.publish_rviz_visualizations()
            return

        if not proceed:
            self.publish_rviz_visualizations()
            return
        if len(self.robot_pose_odom) < 2:
            self.get_logger().warn("Insufficient odometry data from /cmd_vel.")
            self.publish_rviz_visualizations()
            return

        u = self.odom_to_control()
        J_motion = self.motion_jacobian(x=self.robot_state, u=u)
        J_noise, Rn = self.noise_jacobian(x=self.robot_state, u=u)
        self.update_robot_state(self.motion_model(x=self.robot_state, u=u))
        self.compute_cov_pred(J_motion, J_noise, Rn)

        if pred_only:
            self.publish_rviz_visualizations()
            return

        l_measurement_polar = self.landmark_measurements.data
        if l_measurement_polar is None:
            self.publish_rviz_visualizations()
            return
        l_measurement_xy = self.convert_to_fixed_frame(l_measurement_polar)

        for z, l_xy in zip(l_measurement_polar, l_measurement_xy):
            j = self.find_association(l_xy)
            if j == -1:
                self.initialize_landmark(l_xy, measurement=z)
                j = self.landmark_count - 1
            H, Z, z_pred = self.compute_obs(landmark_ind=j, rng=z[0])
            ind = [0, 1, 2, 3 + 2 * j, 3 + 2 * j + 1]
            cov_ = self.sigma[:, ind]
            try:
                K = np.linalg.multi_dot((cov_, H.T, np.linalg.pinv(Z)))
            except np.linalg.LinAlgError:
                self.get_logger().warn(f"Failed to compute Kalman gain for landmark {j}.")
                continue
            innovation = (z - z_pred)
            innovation[1] = angle_between_yaw(yaw1=z_pred[1], yaw2=z[1])
            innovation = innovation[:, np.newaxis]
            self.mu = self.mu + K.dot(innovation)
            self.sigma = self.sigma - np.linalg.multi_dot((K, Z, K.T))

        self.publish_rviz_visualizations()

def main(args=None):
    rclpy.init(args=args)
    ekf_slam = SlamEkf()
    ekf_slam.get_logger().info("EKF SLAM with /cmd_vel odometry and RViz visualization started!")
    try:
        rclpy.spin(ekf_slam)
    except KeyboardInterrupt:
        ekf_slam.get_logger().info("Shutting down EKF SLAM...")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()