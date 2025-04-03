import numpy as np
import pytest
from filtering_exercises.environments import NonlinearWorld
from filtering_exercises.assignment2_ekf.extended_kalman_filter import ExtendedKalmanFilter

@pytest.fixture
def env():
    return NonlinearWorld()

@pytest.fixture
def filter(env):
    return ExtendedKalmanFilter(env)

def test_initial_state(filter):
    """Test that initial state and covariance are set correctly."""
    # Check mean
    assert filter.mu.shape == (3,)  # x, y, theta
    assert np.all(np.isfinite(filter.mu))
    
    # Check covariance
    assert filter.Sigma.shape == (3, 3)
    assert np.all(np.isfinite(filter.Sigma))
    assert np.all(np.linalg.eigvals(filter.Sigma) > 0)  # Positive definite

def test_motion_jacobian(filter):
    """Test computation of motion model Jacobian."""
    # Set known state
    filter.mu = np.array([1.0, 2.0, np.pi/4])
    
    # Compute Jacobian
    G = filter.compute_motion_jacobian()
    
    # Check properties
    assert G.shape == (3, 3)
    assert np.all(np.isfinite(G))
    assert np.trace(G) != 0  # Should have non-zero diagonal elements

def test_measurement_jacobian(filter):
    """Test computation of measurement model Jacobian."""
    # Set known state
    filter.mu = np.array([1.0, 2.0, np.pi/4])
    
    # Compute Jacobian
    H = filter.compute_measurement_jacobian()
    
    # Check properties
    assert H.shape[1] == 3  # 3 state variables
    assert np.all(np.isfinite(H))

def test_predict_forward(filter):
    """Test prediction step with forward motion."""
    # Initial state
    initial_mu = filter.mu.copy()
    initial_Sigma = filter.Sigma.copy()
    
    # Predict forward motion
    forward_vel = 1.0
    angular_vel = 0.0
    dt = 0.1
    filter.predict(forward_vel, angular_vel, dt)
    
    # Check that state changed
    assert not np.allclose(filter.mu, initial_mu)
    assert not np.allclose(filter.Sigma, initial_Sigma)
    assert np.all(np.linalg.eigvals(filter.Sigma) > 0)  # Still positive definite

def test_predict_turn(filter):
    """Test prediction step with turning motion."""
    # Initial state
    initial_theta = filter.mu[2]
    
    # Predict turn motion
    forward_vel = 0.0
    angular_vel = 0.5
    dt = 0.1
    filter.predict(forward_vel, angular_vel, dt)
    
    # Check that orientation changed
    assert filter.mu[2] != initial_theta
    assert np.all(np.linalg.eigvals(filter.Sigma) > 0)  # Still positive definite

def test_update(filter):
    """Test measurement update step."""
    # Initial state uncertainty
    initial_trace = np.trace(filter.Sigma)
    
    # Get real measurements from environment
    measurements = filter.env._get_sensor_reading()
    
    # Update state
    filter.update(measurements)
    
    # Uncertainty should decrease
    assert np.trace(filter.Sigma) < initial_trace
    assert np.all(np.linalg.eigvals(filter.Sigma) > 0)  # Still positive definite

def test_full_cycle(filter):
    """Test a full prediction-update cycle."""
    # Initial state
    initial_mu = filter.mu.copy()
    initial_Sigma = filter.Sigma.copy()
    
    # Predict
    forward_vel = 1.0
    angular_vel = 0.2
    dt = filter.env.dt  # Use environment's dt
    filter.predict(forward_vel, angular_vel, dt)
    
    # Update with real measurements
    measurements = filter.env._get_sensor_reading()
    filter.update(measurements)
    
    # State should have changed
    assert not np.allclose(filter.mu, initial_mu)
    assert not np.allclose(filter.Sigma, initial_Sigma)
    assert np.all(np.linalg.eigvals(filter.Sigma) > 0)  # Still positive definite 