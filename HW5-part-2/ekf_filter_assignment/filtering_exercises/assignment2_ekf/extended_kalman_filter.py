from typing import Tuple

import numpy as np

from filtering_exercises.environments.nonlinear_world import NonlinearWorld


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter implementation for robot localization.
    Handles nonlinear dynamics and measurement models.

    State: [x, y, heading]
    Control: [forward_velocity, angular_velocity]
    """

    def __init__(self, env: NonlinearWorld, initial_state=None):
        """Initialize EKF with environment and optional initial state."""
        self.env = env
        self.n_states = 3  # x, y, heading
        self.n_measurements = env.n_beams
        
        # Ray casting parameters
        self.ray_step_size = 0.05
        self.max_ray_steps = int(env.max_beam_length / self.ray_step_size)
        self.beam_angles = np.linspace(0, 2*np.pi, env.n_beams, endpoint=False)
        
        # Initialize state and covariance
        if initial_state is not None:
            self.mu = initial_state
        else:
            self.mu = np.array([5.0, 2.0, 0.0])
        
        # Initialize with higher uncertainty
        self.Sigma = np.diag([1.0, 1.0, np.pi/2])  # Much higher initial uncertainty
        
        # Motion model noise - higher base values
        self.base_pos_noise = 0.1 ** 2  # Increased base position noise
        self.base_heading_noise = 0.2 ** 2  # Increased base heading noise
        self.Q = np.diag([self.base_pos_noise, self.base_pos_noise, self.base_heading_noise])
        
        # Measurement model noise
        self.base_measurement_noise = max(env.sensor_noise_std ** 2, 0.1 ** 2)  # Increased base noise
        self.R = np.eye(self.n_measurements) * self.base_measurement_noise
        
        # Allow much wider variance range
        self.min_variance = np.array([0.1, 0.1, 0.1])  # Higher minimum variance
        self.max_variance = np.array([4.0, 4.0, 2*np.pi])  # Much higher maximum variance
        
        # Store history for debugging
        self.state_history = [self.mu.copy()]
        self.covariance_history = [self.Sigma.copy()]
        
        # Store last control inputs
        self.last_forward_vel = 0.0
        self.last_angular_vel = 0.0
        self.last_dt = env.dt
        
        # Update parameters - allow larger updates
        self.max_position_update = 1.0  # Much larger position updates allowed
        self.max_heading_update = np.pi  # Allow full heading updates
        
        # Innovation thresholds
        self.innovation_threshold = 1.0  # More permissive threshold
        self.max_outlier_ratio = 0.7  # Allow more outliers

    def _motion_model(self, state: np.ndarray, forward_vel: float, angular_vel: float, dt: float) -> np.ndarray:
        """Predict next state using nonlinear motion model."""
        x, y, heading = state
        
        # Get velocity scaling based on position
        vel_scale = self.env._get_velocity_scaling(state[:2])
        effective_vel = forward_vel * vel_scale
        
        # Speed-dependent turning
        speed = np.abs(effective_vel) / self.env.max_velocity
        speed_factor = np.clip(speed, 0.5, 1.5)
        effective_angular_vel = angular_vel / speed_factor
        
        # Predict next state
        dx = effective_vel * np.cos(heading) * dt
        dy = effective_vel * np.sin(heading) * dt
        dheading = effective_angular_vel * dt
        
        # Check boundary constraints
        next_x = np.clip(x + dx, 0.0, self.env.size[0])
        next_y = np.clip(y + dy, 0.0, self.env.size[1])
        
        # If we hit a boundary, reflect the heading
        if next_x in (0.0, self.env.size[0]) or next_y in (0.0, self.env.size[1]):
            if next_x in (0.0, self.env.size[0]):
                heading = np.pi - heading  # Reflect x
            if next_y in (0.0, self.env.size[1]):
                heading = -heading  # Reflect y
            heading = heading % (2 * np.pi)  # Normalize
        
        next_state = np.array([
            next_x,
            next_y,
            (heading + dheading) % (2 * np.pi)
        ])
        
        return next_state

    def compute_motion_jacobian(self, state: np.ndarray = None, forward_vel: float = None, angular_vel: float = None, dt: float = 0.1) -> np.ndarray:
        """Compute Jacobian of motion model with respect to state."""
        if state is None:
            state = self.mu
        if forward_vel is None:
            forward_vel = self.last_forward_vel
        if angular_vel is None:
            angular_vel = self.last_angular_vel
            
        return self._motion_jacobian(state, forward_vel, angular_vel, dt)

    def _motion_jacobian(self, state: np.ndarray, forward_vel: float, angular_vel: float, dt: float) -> np.ndarray:
        """Compute Jacobian of motion model with respect to state."""
        x, y, heading = state
        
        # Get velocity scaling and its derivatives
        vel_scale = self.env._get_velocity_scaling(state[:2])
        freq = self.env.terrain_frequency
        dvs_dx = 0.5 * freq * np.cos(freq * x) * np.cos(freq * y)
        dvs_dy = -0.5 * freq * np.sin(freq * x) * np.sin(freq * y)
        
        # Effective velocity and its derivatives
        effective_vel = forward_vel * vel_scale
        
        # Speed-dependent turning factor
        speed = np.abs(effective_vel) / self.env.max_velocity
        speed_factor = np.clip(speed, 0.5, 1.5)
        effective_angular_vel = angular_vel / speed_factor
        
        # Compute Jacobian elements
        F = np.eye(3)
        
        # Position derivatives
        F[0, 0] = 1 + dt * forward_vel * dvs_dx * np.cos(heading)
        F[0, 1] = dt * forward_vel * dvs_dy * np.cos(heading)
        F[0, 2] = -dt * effective_vel * np.sin(heading)
        
        F[1, 0] = dt * forward_vel * dvs_dx * np.sin(heading)
        F[1, 1] = 1 + dt * forward_vel * dvs_dy * np.sin(heading)
        F[1, 2] = dt * effective_vel * np.cos(heading)
        
        # Heading derivatives (affected by speed-dependent turning)
        F[2, 0] = -dt * angular_vel * dvs_dx / (speed_factor**2)
        F[2, 1] = -dt * angular_vel * dvs_dy / (speed_factor**2)
        F[2, 2] = 1
        
        return F

    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        """Predict measurements using nonlinear measurement model."""
        x, y, heading = state
        measurements = np.zeros(self.env.n_beams)
        
        # Compute beam angles in global frame
        global_angles = (self.beam_angles + heading) % (2 * np.pi)
        
        # Ray casting with adaptive step size
        for i, angle in enumerate(global_angles):
            # Ray direction
            direction = np.array([np.cos(angle), np.sin(angle)])
            curr_pos = np.array([x, y])
            distance = 0.0
            
            # Use adaptive step size - smaller near obstacles
            base_step = self.ray_step_size
            curr_step = base_step
            hit_something = False
            
            for _ in range(self.max_ray_steps):
                prev_pos = curr_pos.copy()
                curr_pos += direction * curr_step
                distance += curr_step
                
                # Check bounds
                if not (0 <= curr_pos[0] <= self.env.size[0] and 
                       0 <= curr_pos[1] <= self.env.size[1]):
                    # Back up to boundary
                    if curr_pos[0] < 0 or curr_pos[0] > self.env.size[0]:
                        t = (np.clip(curr_pos[0], 0, self.env.size[0]) - prev_pos[0]) / direction[0]
                        curr_pos = prev_pos + direction * t
                        distance = np.linalg.norm(curr_pos - np.array([x, y]))
                    if curr_pos[1] < 0 or curr_pos[1] > self.env.size[1]:
                        t = (np.clip(curr_pos[1], 0, self.env.size[1]) - prev_pos[1]) / direction[1]
                        curr_pos = prev_pos + direction * t
                        distance = np.linalg.norm(curr_pos - np.array([x, y]))
                    hit_something = True
                    break
                
                # Check if near any obstacle
                near_obstacle = False
                for obstacle in self.env.obstacles:
                    min_dist = float('inf')
                    for j in range(len(obstacle)):
                        p1 = np.array(obstacle[j])
                        p2 = np.array(obstacle[(j + 1) % len(obstacle)])
                        # Distance to line segment
                        line_dir = p2 - p1
                        line_len = np.linalg.norm(line_dir)
                        if line_len > 0:
                            line_dir = line_dir / line_len
                            v = curr_pos - p1
                            t = np.clip(np.dot(v, line_dir), 0, line_len)
                            proj = p1 + line_dir * t
                            dist = np.linalg.norm(curr_pos - proj)
                            min_dist = min(min_dist, dist)
                    if min_dist < 0.5:  # If near obstacle
                        near_obstacle = True
                        curr_step = base_step * 0.1  # Reduce step size
                        break
                
                if not near_obstacle:
                    curr_step = base_step  # Reset step size
                
                # Check if hitting any obstacle
                if self.env._check_collision(curr_pos):
                    # Back up to get more accurate collision point
                    while curr_step > base_step * 0.01:
                        curr_step *= 0.5
                        test_pos = prev_pos + direction * curr_step
                        if not self.env._check_collision(test_pos):
                            curr_pos = test_pos
                            distance = np.linalg.norm(curr_pos - np.array([x, y]))
                    hit_something = True
                    break
                
                if distance >= self.env.max_beam_length:
                    distance = self.env.max_beam_length
                    hit_something = True
                    break
            
            # If we didn't hit anything, use max beam length
            if not hit_something:
                distance = self.env.max_beam_length
            
            measurements[i] = distance
        
        return measurements

    def compute_measurement_jacobian(self, state: np.ndarray = None) -> np.ndarray:
        """Compute Jacobian of measurement model with respect to state."""
        if state is None:
            state = self.mu
        return self._measurement_jacobian(state)

    def _measurement_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute Jacobian of measurement model with respect to state."""
        x, y, heading = state
        H = np.zeros((self.env.n_beams, 3))
        
        # Get current measurements to determine which beams hit obstacles
        measurements = self._measurement_model(state)
        global_angles = (self.beam_angles + heading) % (2 * np.pi)
        
        for i, (angle, measurement) in enumerate(zip(global_angles, measurements)):
            if measurement < self.env.max_beam_length:
                # Beam hits an obstacle - compute Jacobian
                H[i, 0] = -np.cos(angle) / measurement  # dx
                H[i, 1] = -np.sin(angle) / measurement  # dy
                # Effect of heading on measurement
                H[i, 2] = -measurement * (
                    -np.sin(angle) * H[i, 0] + np.cos(angle) * H[i, 1]
                )
        
        return H


    def _constrain_covariance(self):
        """Apply minimum and maximum constraints to the covariance diagonal."""
        diag_idx = np.diag_indices_from(self.Sigma)
        self.Sigma[diag_idx] = np.clip(
            self.Sigma[diag_idx],
            self.min_variance,
            self.max_variance
        )

    def _normalize_heading(self):
        """Normalize heading to [0, 2π)"""
        self.mu[2] = self.mu[2] % (2 * np.pi)

    def predict(self, forward_vel: float, angular_vel: float, dt: float):
        """
        TODO: Implement the Extended Kalman Filter prediction step.
        
        This method should implement the prediction step of the EKF using a nonlinear motion model
        that accounts for the robot's unicycle dynamics and terrain-dependent velocity scaling.
        
        Implementation steps:
        1. Store control inputs for later use:
           - self.last_forward_vel = action[0]
           - self.last_angular_vel = action[1]
           - self.last_dt = dt
        
        2. Update state using nonlinear motion model:
           - Call self._motion_model(self.mu, action[0], action[1], dt)
           - This gives you the predicted next state
           - Store result in self.mu
           - Call self._normalize_heading() to keep heading in [0, 2π]
        
        3. Compute Jacobian of motion model:
           - Call self._motion_jacobian(self.mu, action[0], action[1], dt)
           - This gives you the linearized state transition matrix F
        
        4. Update covariance using the EKF covariance update equation:
           - Use self.Sigma for P and self.Q for process noise
        
        5. Apply covariance constraints:
           - Call self._constrain_covariance()
           - This ensures numerical stability
        
        6. Store state and covariance history:
           - Append copies of current state and covariance to respective history lists
        
        Parameters:
            action (np.ndarray): [forward_velocity, angular_velocity]
            dt (float): Time step duration
        """
        
        
        # raise NotImplementedError("Not implemented")
        self.last_forward_vel = forward_vel
        self.last_angular_vel = angular_vel
        self.last_dt = dt
        # Update state using nonlinear motion model
        self.mu = self._motion_model(self.mu, forward_vel, angular_vel, dt)
        self._normalize_heading()
        # Compute Jacobian of motion model
        F = self._motion_jacobian(self.mu, forward_vel, angular_vel, dt)
        # Update covariance using the EKF covariance update equation
        self.Sigma = F @ self.Sigma @ F.T + self.Q
        # Apply covariance constraints
        self._constrain_covariance()
        # Store state and covariance history
        self.state_history.append(self.mu.copy())
        self.covariance_history.append(self.Sigma.copy())
        

    def update(self, measurements: np.ndarray):
        """
        TODO: Implement the Extended Kalman Filter update step.
        
        This method should implement the update step of the EKF using nonlinear range measurements
        from multiple beams. The measurement model accounts for hallway geometry and obstacles.
        
        Implementation steps:
        1. Get expected measurements:
           - Call self._measurement_model(self.state)
           - This simulates what measurements we expect to see
        
        2. Compute measurement Jacobian:
           - Call self._measurement_jacobian(self.state)
           - This linearizes the measurement model around current state
        
        3. Setup adaptive measurement noise:
           - Start with base noise matrix self.R
           - Scale noise up for beams that don't hit anything:
             * If expected_z[i] >= self.env.max_beam_length: scale by 10.0
             * Otherwise scale by 2.0
           - Create diagonal matrix R with these scaled noises
        
        4. Compute Kalman gain:
           - Use self.covariance for P
        
        5. Update state:
           - Limit update magnitude if needed
           - Apply update: self.state = self.state + dx
           - Call self._normalize_heading()
        
        6. Update covariance:
           - Use Joseph form for better numerical stability:
             P = (I - KH)P(I - KH).T + KRK.T
           - Call self._constrain_covariance()
        
        7. Store updated state and covariance:
           - Append to respective history lists
        
        Parameters:
            measurements (np.ndarray): Array of beam distance measurements
        """
        # raise NotImplementedError("TODO: Implement the Extended Kalman Filter update step")
        # Get expected measurements
        expected_z = self._measurement_model(self.mu)
        # Compute measurement Jacobian
        H = self._measurement_jacobian(self.mu)
        # Setup adaptive measurement noise
        # using self.R
        for i in range(self.n_measurements):
            if expected_z[i] >= self.env.max_beam_length:
                self.R[i, i] *= 10.0
            else:
                self.R[i, i] *= 2.0
        # R = np.diag(np.diag(self.R)) 
        # Compute Kalman gain
        S = H @ self.Sigma @ H.T + self.R
        # Compute Kalman gain
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        # Update state
        # Limit update magnitude if needed
        # self.state = self.state + K @ (measurements - expected_z)
        innovation = measurements - expected_z
        # innovation[expected_z >= self.env.max_beam_length] = 0.0
        innovation = np.clip(innovation, -self.innovation_threshold, self.innovation_threshold)
        
        dx = K @ innovation
        self.mu = self.mu + dx
        self._normalize_heading()
        # Update covariance using Joseph form
        I = np.eye(self.n_states)
        # P = (I - K @ H) @ self.Sigma
        # P = (I - KH)P(I - KH).T + KRK.T
        # print("P shape:", self.Sigma.shape)  # (3, 3)
        # print("K shape:", K.shape)  # (3, 8)
        # print("H shape:", H.shape)  # (8, 3)
        # print("R shape:", R.shape)  # (8, 8)
        t1 = I - K @ H
        P = t1 @ self.Sigma @ t1.T + (K @ self.R @ K.T)
        self.Sigma = P
        # Apply covariance constraints
        self._constrain_covariance()
        
        # Store updated state and covariance
        self.state_history.append(self.mu.copy())
        self.covariance_history.append(self.Sigma.copy())
        

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current state estimate and covariance."""
        return self.mu.copy(), self.Sigma.copy()

    def get_position(self) -> np.ndarray:
        """Return current position estimate."""
        return self.mu[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Return current velocity estimate based on control inputs."""
        heading = self.mu[2]
        vel_scale = self.env._get_velocity_scaling(self.mu[:2])
        effective_vel = self.last_forward_vel * vel_scale
        return np.array([
            effective_vel * np.cos(heading),
            effective_vel * np.sin(heading)
        ])

    def get_state_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the history of states and covariances."""
        return np.array(self.state_history), np.array(self.covariance_history)

    def reset(self, initial_state=None):
        """Reset the filter state."""
        if initial_state is not None:
            self.mu = initial_state.copy()
        else:
            self.mu = np.array([5.0, 2.0, 0.0])
        
        # Reset covariance to initial uncertainty
        self.Sigma = np.diag([1.0, 1.0, np.pi/2])
        
        # Reset control inputs
        self.last_forward_vel = 0.0
        self.last_angular_vel = 0.0
        
        # Clear history
        self.state_history = [self.mu.copy()]
        self.covariance_history = [self.Sigma.copy()]
