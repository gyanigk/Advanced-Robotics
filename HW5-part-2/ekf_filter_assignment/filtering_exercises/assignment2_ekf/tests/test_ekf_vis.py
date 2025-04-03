#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time

from filtering_exercises.environments import NonlinearWorld
from filtering_exercises.assignment2_ekf.extended_kalman_filter import ExtendedKalmanFilter
from filtering_exercises.assignment2_ekf.visualizer import EKFVisualizer

def test_ekf_visualization():
    """Run a test visualization of the EKF."""
    # Create environment and filter
    env = NonlinearWorld()
    print(f"\nEnvironment Configuration:")
    print(f"Number of beams: {env.n_beams}")
    print(f"Environment size: {env.size}")
    print(f"Max beam length: {env.max_beam_length}")
    
    # Set initial state away from obstacles and boundaries
    initial_state = np.array([7.0, 7.0, -np.pi/4])
    env.reset()
    env.agent_pos = initial_state[:2]
    env.agent_heading = initial_state[2]
    
    # Initialize EKF with some uncertainty about initial state
    initial_state_ekf = initial_state + np.random.normal(0, [0.2, 0.2, 0.1])
    ekf = ExtendedKalmanFilter(env, initial_state_ekf)
    
    # Create visualizer
    vis = EKFVisualizer(env, ekf)
    
    # Initialize error tracking with timestamps
    position_errors = []
    heading_errors = []
    times = []
    measurement_innovations = []
    velocity_scales = []
    
    # Initialize beam artists with different styles for true and expected measurements
    beam_angles = np.linspace(0, 2*np.pi, env.n_beams, endpoint=False)
    print(f"\nBeam Configuration:")
    print(f"Beam angles (degrees): {np.degrees(beam_angles)}")
    
    # Create lists to store beam artists
    true_beam_lines = []
    est_beam_lines = []
    true_beam_ends = []
    est_beam_ends = []
    
    # Initialize beam artists for each beam
    for i in range(env.n_beams):
        # True measurement beams (red)
        line, = vis.ax.plot([], [], 'r-', alpha=0.3, linewidth=1.5, label='True Measurement' if i == 0 else "")
        true_beam_lines.append(line)
        end, = vis.ax.plot([], [], 'ro', markersize=3, alpha=0.3)
        true_beam_ends.append(end)
        
        # Expected measurement beams (blue)
        line, = vis.ax.plot([], [], 'b--', alpha=0.5, linewidth=1.0, label='Expected Measurement' if i == 0 else "")
        est_beam_lines.append(line)
        end, = vis.ax.plot([], [], 'bo', markersize=3, alpha=0.5)
        est_beam_ends.append(end)
    
    # Add legend
    vis.ax.legend()
    
    # Get initial measurements for debugging
    initial_measurements = env._get_sensor_reading()
    initial_expected = ekf._measurement_model(ekf.mu)
    print(f"\nInitial Measurements:")
    print(f"True measurements: {initial_measurements}")
    print(f"Expected measurements: {initial_expected}")
    
    # Simulation loop
    try:
        for i in range(200):
            t = i * env.dt
            times.append(t)
            
            vel_scale = env._get_velocity_scaling(ekf.mu[:2])
            velocity_scales.append(vel_scale)
            
            # Generate control input - smoother motion for better tracking
            if i < 50:  # Move diagonally down-left
                forward_vel = 0.3
                angular_vel = 0.0
                if (env.agent_pos[0] < 1.0 or env.agent_pos[0] > env.size[0] - 1.0 or
                    env.agent_pos[1] < 1.0 or env.agent_pos[1] > env.size[1] - 1.0):
                    angular_vel = 0.5
            elif i < 100:  # Turn right
                forward_vel = 0.2
                angular_vel = 0.3
            elif i < 150:  # Move diagonally up-right
                forward_vel = 0.3
                angular_vel = 0.0
                if (env.agent_pos[0] < 1.0 or env.agent_pos[0] > env.size[0] - 1.0 or
                    env.agent_pos[1] < 1.0 or env.agent_pos[1] > env.size[1] - 1.0):
                    angular_vel = -0.5
            else:  # Turn left
                forward_vel = 0.2
                angular_vel = -0.3
            
            # Move true robot state using environment dynamics
            true_state, measurements, collision = env.step(np.array([forward_vel, angular_vel]))
            
            if collision:
                forward_vel = -0.2
                angular_vel = 0.8
                true_state, measurements, _ = env.step(np.array([forward_vel, angular_vel]))
            
            # Update filter
            ekf.predict(forward_vel, angular_vel, env.dt)
            expected_measurements = ekf._measurement_model(ekf.mu)
            
            # Print measurement debug info occasionally
            if i % 50 == 0:
                print(f"\nStep {i} Debug Info:")
                print(f"True state: {true_state}")
                print(f"EKF state: {ekf.mu}")
                print(f"True measurements: {measurements}")
                print(f"Expected measurements: {expected_measurements}")
                print(f"Measurement differences: {np.abs(measurements - expected_measurements)}")
            
            innovations = measurements - expected_measurements
            measurement_innovations.append(np.mean(np.abs(innovations)))
            ekf.update(measurements)
            
            # Calculate errors
            pos_error = np.linalg.norm(true_state[:2] - ekf.mu[:2])
            heading_error = np.abs((true_state[2] - ekf.mu[2] + np.pi) % (2*np.pi) - np.pi)
            position_errors.append(pos_error)
            heading_errors.append(heading_error)
            
            # Update beam visualizations with both lines and endpoints
            for j in range(env.n_beams):
                # True measurements
                angle = beam_angles[j] + true_state[2]
                dist = measurements[j]
                end_x = true_state[0] + dist * np.cos(angle)
                end_y = true_state[1] + dist * np.sin(angle)
                true_beam_lines[j].set_data([true_state[0], end_x], [true_state[1], end_y])
                true_beam_ends[j].set_data([end_x], [end_y])

                # Expected measurements
                angle = beam_angles[j] + ekf.mu[2]
                dist = expected_measurements[j]
                end_x = ekf.mu[0] + dist * np.cos(angle)
                end_y = ekf.mu[1] + dist * np.sin(angle)
                est_beam_lines[j].set_data([ekf.mu[0], end_x], [ekf.mu[1], end_y])
                est_beam_ends[j].set_data([end_x], [end_y])
            
            # Update plots with current state
            vis.update_plots(true_state)
            
            # Update metrics in title
            window_size = 10
            avg_pos_error = np.mean(position_errors[-window_size:]) if len(position_errors) > window_size else pos_error
            avg_heading_error = np.mean(heading_errors[-window_size:]) if len(heading_errors) > window_size else heading_error
            avg_innovation = np.mean(measurement_innovations[-window_size:]) if len(measurement_innovations) > window_size else measurement_innovations[-1]
            
            vis.ax.set_title(
                f'Time: {t:.1f}s | Beams: {env.n_beams}\n' + 
                f'Position Error: {pos_error:.2f}m (Avg: {avg_pos_error:.2f}m)\n' +
                f'Heading Error: {np.degrees(heading_error):.1f}° (Avg: {np.degrees(avg_heading_error):.1f}°)\n' +
                f'Avg Innovation: {avg_innovation:.3f}m'
            )
            
            # Use a single draw and pause call per frame
            if i % 2 == 0:  # Update display every other frame for smoother performance
                plt.draw()
                plt.pause(0.001)  # Shorter pause for better responsiveness
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    finally:
        # Plot error analysis
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(times, position_errors)
        plt.title('Position Error Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(times, np.degrees(heading_errors))
        plt.title('Heading Error Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (degrees)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(times, measurement_innovations)
        plt.title('Average Measurement Innovation')
        plt.xlabel('Time (s)')
        plt.ylabel('Innovation (m)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(times, velocity_scales)
        plt.title('Velocity Scaling Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Scale Factor')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_ekf_visualization() 