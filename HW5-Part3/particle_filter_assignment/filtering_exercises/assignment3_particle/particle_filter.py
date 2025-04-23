from typing import Tuple

import numpy as np

from filtering_exercises.environments.multimodal_world import MultiModalWorld


class ParticleFilter:
    """
    Particle Filter implementation for robot localization.
    Handles non-Gaussian distributions and multimodal beliefs.

    State: [x, y, heading]
    Control: [forward_velocity, angular_velocity]
    """

    def __init__(self, env: MultiModalWorld, num_particles: int = 100):
        """Initialize particle filter with environment and number of particles."""
        self.env = env
        self.n_particles = num_particles
        
        
        self._initialize_particles(num_particles)
        # Initialize uniform weights
        self.weights = np.ones(num_particles) / num_particles
        
        # Motion model noise parameters
        self.alpha1 = 0.1  # Noise in forward motion
        self.alpha2 = 0.1  # Noise in rotation
        
        # Measurement model parameters
        self.sigma_range = env.sensor_noise_std
        self.sigma_bearing = env.sensor_noise_std
        
        # Resampling parameters
        self.resampling_noise = 0.02  # Small noise to add after resampling
        
        # Store history for debugging
        self.particle_history = [self.particles.copy()]
        self.weight_history = [self.weights.copy()]

    def _initialize_particles(self, num_particles: int = None):
        # Initialize particles uniformly in free space
        self.particles = np.zeros((num_particles, 3))  # x, y, heading
        for i in range(num_particles):
            while True:
                x = np.random.uniform(0, self.env.size[0])
                y = np.random.uniform(0, self.env.size[1])
                if not self.env._check_collision(np.array([x, y])):
                    break
            self.particles[i] = [x, y, np.random.uniform(-np.pi, np.pi)]

    
    #adding the function from nonlinear world as unable to call velocity_scaling being private
    def _get_velocity_scaling(self, position: np.ndarray) -> float:
        """
        Get velocity scaling based on position (simulates terrain effects).
        Creates a sinusoidal pattern of fast/slow regions.
        """
        x, y = position
        # Create a 2D sinusoidal pattern
        terrain = np.sin(self.terrain_frequency * x) * np.cos(
            self.terrain_frequency * y
        )
        # Map to range [0.5, 1.5] to scale velocity
        scaling = 1.0 + 0.5 * terrain
        return scaling
    
    def predict(self, action: np.ndarray):
        """
        TODO: Implement the Particle Filter prediction step.
        
        This method should implement the prediction step by propagating each particle
        through the motion model with noise. The motion model should account for the
        robot's unicycle dynamics.
        
        Implementation steps:
        1. Extract control inputs:
           - forward_vel = action[0]
           - angular_vel = action[1]
        
        2. For each particle:
           - Get velocity scaling based on terrain
           - Update particle state using noisy unicycle model:
             * dx = noisy_vel * cos(theta) * dt
             * dy = noisy_vel * sin(theta) * dt
             * dtheta = noisy_angular_vel * dt
             * add guassian noise to dx, dy, dtheta, with alpha1 and alpha2
           - check if the particle is in collision, if it is, resample it
             * use the function self.env._check_collision() to check if the particle is in collision
           - Normalize heading to [-π, π]
        
        3. Store updated particles in history:
           - Append self.particles.copy() to self.particle_history
        
        Parameters:
            action (np.ndarray): [forward_velocity, angular_velocity]
        """
        forward_vel = action[0]
        angular_vel = action[1]
        self.env.dt = 1.0  # Time step
        
        for i in range(self.n_particles):
            x ,y, theta = self.particles[i]
            # Get velocity scaling based on terrain
            vel_scaling = self.env._get_velocity_scaling(self.particles[i, :2])
            
            noisy_angular_vel = angular_vel * vel_scaling +np.random.normal(0, self.alpha2)
            noisy_vel = forward_vel * vel_scaling + np.random.normal(0, self.alpha1)
            # noisy_vel *= 0.5

            # Add noise to the motion model
            dx = noisy_vel * np.cos(theta) * self.env.dt
            dy = noisy_vel * np.sin(theta) * self.env.dt
            dtheta = noisy_angular_vel * self.env.dt
            
            x += dx
            y += dy
            theta += dtheta
            x = np.clip(x, 0, self.env.size[0])
            y = np.clip(y, 0, self.env.size[1])
            
            theta = np.arctan2(np.sin(self.particles[i, 2]), np.cos(self.particles[i, 2]))

            
            # noisy_angular_vel = angular_vel + np.random.normal(0, self.alpha2)
            # noisy_vel = forward_vel + np.random.normal(0, self.alpha1)
            
            # Check for collision
            if self.env._check_collision(np.array([x, y])):
                # Resample particle
                self.particles[i] = self.particles[np.random.choice(self.n_particles)]
                # self.particles[i] = [x, y, np.random.uniform(-np.pi, np.pi)]
            else:
                self.particles[i] = [x, y, theta]
        self.particle_history.append(self.particles.copy())
        
        
        # raise NotImplementedError("Not implemented")

    def update(self, measurements: np.ndarray):
        """
        TODO: Implement the Particle Filter update step.
        
        This method should implement the update step by computing importance weights
        for each particle based on how well the actual measurements match the expected
        measurements from that particle's state.
        
        Implementation steps:
        1. For each particle:
           - Get expected measurements using self._get_expected_measurements()
           - Compute measurement likelihood:
             * For each beam:
               - Calculate range error
               - Compute Gaussian probability
               - Multiply probabilities for final likelihood
           - Update particle weight: weight *= likelihood
        
        2. Normalize weights:
           - Sum of weights should be 1.0
           - Handle numerical underflow
        
        3. Store updated weights in history:
           - Append self.weights.copy() to self.weight_history
        
        Parameters:
            measurements (np.ndarray): Array of beam distances
        """
        # raise NotImplementedError("Not implemented")
        # print(measurements)
        measure = np.asarray(measurements)
        # print(measure)
        # print(measure.shape)
        if measure.ndim == 2 and measure.shape[1] == 2:
            measure = measure[:, 0]
        n_beams = self.env.n_beams
        if measure.ndim == 1 and measure.shape[0] < n_beams:
            measure = np.tile(measure, int(np.ceil(n_beams / measure.shape[0])))[:n_beams]

        sigma = self.env.sensor_noise_std 
        log_weights = np.log(self.weights + 1e-100)  # Avoid log(0)
        log_likelihoods = np.empty(self.n_particles)
        # Compute likelihood for each particle
        for i in range(self.n_particles):
            expected = self._get_expected_measurements(self.particles[i])  # [x, y, heading]
            range_error = measure - expected
            log_probs = -0.5 * (range_error / sigma) ** 2 - np.log(np.sqrt(2 * np.pi) * sigma)
            log_likelihoods[i] = log_probs.sum()

        # Update and normalize weights
        log_weights += log_likelihoods
        log_weights -= log_weights.max()  # Prevent overflow
        new_weights = np.exp(log_weights)
        
        # Handle numerical issues
        weight_sum = new_weights.sum()
        self.weights = new_weights / weight_sum
        # for i in range(self.n_particles):   
        #     expected_measurements = self._get_expected_measurements(self.particles[i,:2])
        #     # expected_measurements_i = expected_measurements[i]
        #     # measurements_i = measurements[i]
        #     # Compute measurement likelihood
        #     likelihood = 1.0
        #     for j in range(len(expected_measurements)):
        #         # Calculate range error
        #         range_error = expected_measurements[j] - measurements[j]
        #         print(range_error)
        #         # Compute Gaussian probability
        #         prob = np.exp(-0.5 * (range_error ** 2) / (self.sigma_range ** 2)) / (self.sigma_range * np.sqrt(2 * np.pi))
        #         # Multiply probabilities for final likelihood
        #         likelihood *= prob
        #     # Update particle weight
        #     self.weights[i] *= likelihood
        # # Normalize weights
        # self.weights /= np.sum(self.weights)
        # # Handle numerical underflow
        # if np.sum(self.weights) == 0:
        #     self.weights = np.ones(self.n_particles) / self.n_particles 
        
        # Store updated weights in history
        self.weight_history.append(self.weights.copy())

    def resample(self):
        """
        TODO: Implement the Particle Filter resampling step.
        
        This method should implement the resampling step. 
        After resampling, add small noise to particles to
        prevent particle depletion.
        
        Implementation steps:
        1. Implementresampling:
            - resample all particles according to their weights
            - there are many different resampling techniques, this is just one of them
            - Try using np.random.choice() to resample particles
        
        2. Create new particle array:
           - Copy selected particles
           - Add small random noise (self.resampling_noise)
           - Normalize headings to [-π, π]
        
        3. Reset weights to uniform:
           - All weights should be 1/N
        
        4. Store new particles and weights in history
        """
        # raise NotImplementedError("Not implemented")
        # Resample particles according to their weights
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
        self.particles = self.particles[indices]
        
        # Add small noise to particles
        noise = np.random.normal(0, self.resampling_noise, self.particles.shape)
        self.particles[:, :2] += noise[: , :2]
        self.particles[:, 2] = (self.particles[:, 2] + noise[:, 2]) % (2 * np.pi)
        # Reset weights to uniform
        self.weights = np.ones(self.n_particles) / self.n_particles
        # Store new particles and weights in history
        self.particle_history.append(self.particles.copy())
        self.weight_history.append(self.weights.copy())


    def _get_expected_measurements(self, state: np.ndarray) -> np.ndarray:
        """Get expected measurements for a particle state."""
        # Create a fake particle state for the environment
        self.env.agent_pos = state[:2]
        self.env.agent_heading = state[2]
        
        # Get sensor readings from environment
        return self.env._get_sensor_reading()

    def estimate_state(self) -> np.ndarray:
        """
        TODO: Implement state estimation from particles.
        
        This method should compute the weighted mean state from the particle set.
        Special care must be taken to handle the circular nature of the heading angle.
        
        Implementation steps:
        1. Compute weighted mean for x and y:
           - Simple weighted average using particle positions and weights
        
        2. Compute weighted mean for heading:
           - Convert angles to unit vectors
           - Take weighted average of vectors
           - Convert back to angle
        
        Returns:
            np.ndarray: Estimated state [x, y, heading]
        """
        # raise NotImplementedError("Not implemented")
        x_mean = np.average(self.particles[:, 0], weights=self.weights)
        y_mean = np.average(self.particles[:, 1], weights=self.weights)
        heading_mean = np.arctan2(np.sum(np.sin(self.particles[:, 2]) * self.weights),
                             np.sum(np.cos(self.particles[:, 2]) * self.weights))
        return np.array([x_mean, y_mean, heading_mean])

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current state estimate and particles."""
        return self.estimate_state(), self.particles.copy()

    def get_particles(self) -> np.ndarray:
        """Return current particles."""
        return self.particles.copy()

    def get_weights(self) -> np.ndarray:
        """Return current weights."""
        return self.weights.copy()

    def get_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the history of particles and weights."""
        return np.array(self.particle_history), np.array(self.weight_history)

    def reset(self, num_particles: int = None):
        """Reset the filter state."""
        if num_particles is not None:
            self.n_particles = num_particles
        
        # Reinitialize particles
        self.particles = np.zeros((self.n_particles, 3))
        for i in range(self.n_particles):
            while True:
                x = np.random.uniform(0, self.env.size[0])
                y = np.random.uniform(0, self.env.size[1])
                if not self.env._check_collision(np.array([x, y])):
                    break
            self.particles[i] = [x, y, np.random.uniform(-np.pi, np.pi)]
        
        # Reset weights
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Clear history
        self.particle_history = [self.particles.copy()]
        self.weight_history = [self.weights.copy()] 