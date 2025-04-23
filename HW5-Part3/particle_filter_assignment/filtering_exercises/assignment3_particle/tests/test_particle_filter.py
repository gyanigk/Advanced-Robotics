import numpy as np
import pytest

from filtering_exercises.assignment3_particle.particle_filter import ParticleFilter
from filtering_exercises.environments import MultiModalWorld


@pytest.fixture
def env():
    return MultiModalWorld()


@pytest.fixture
def filter(env):
    return ParticleFilter(env, num_particles=100)


def test_initial_particles(filter):
    """Test that particles are initialized correctly."""
    # Check particle array shapes
    assert filter.particles.shape == (100, 3)  # 100 particles, 3 state variables
    assert filter.weights.shape == (100,)

    # Check weight normalization
    assert np.sum(filter.weights) == pytest.approx(1.0)
    assert np.all(filter.weights > 0)

    # Check particles are within bounds
    assert np.all(filter.particles[:, 0] >= 0)
    assert np.all(filter.particles[:, 0] <= filter.env.size[0])
    assert np.all(filter.particles[:, 1] >= 0)
    assert np.all(filter.particles[:, 1] <= filter.env.size[1])
    assert np.all(np.abs(filter.particles[:, 2]) <= np.pi)


def test_predict_forward(filter):
    """Test prediction step with forward motion."""
    # Store initial particles
    initial_particles = filter.particles.copy()

    # Predict forward motion
    action = np.array([1.0, 0.0])  # Forward velocity, no angular velocity
    filter.predict(action)

    # Check that particles moved
    assert not np.allclose(filter.particles, initial_particles)
    assert np.mean(filter.particles[:, 0]) != np.mean(initial_particles[:, 0])

    # Check bounds
    assert np.all(filter.particles[:, 0] >= 0)
    assert np.all(filter.particles[:, 0] <= filter.env.size[0])


def test_predict_turn(filter):
    """Test prediction step with turning motion."""
    # Store initial particles
    initial_particles = filter.particles.copy()

    # Predict turn motion
    action = np.array([0.0, 0.5])  # No forward velocity, positive angular velocity
    filter.predict(action)

    # Check that orientations changed
    assert not np.allclose(filter.particles[:, 2], initial_particles[:, 2])
    assert np.all(np.abs(filter.particles[:, 2]) <= np.pi)


def test_update(filter, env):
    """Test measurement update step."""
    # Initial weights
    initial_weights = filter.weights.copy()

    z = env._get_sensor_reading()

    # Update weights
    filter.update(z)

    # Check weight properties
    assert not np.allclose(filter.weights, initial_weights)
    assert np.sum(filter.weights) == pytest.approx(1.0)
    assert np.all(filter.weights >= 0)


def test_resample(filter):
    """Test resampling step."""
    # Set known weights
    filter.weights = np.zeros(100)
    filter.weights[0] = 0.5  # One dominant particle
    filter.weights[1:] = 0.5 / 99  # Rest equally weighted

    # Store particles before resampling
    old_particles = filter.particles.copy()

    # Resample
    filter.resample()

    # Check resampling properties
    assert len(filter.particles) == 100
    assert np.sum(filter.weights) == pytest.approx(1.0)
    assert np.all(
        np.abs(filter.weights - 1 / 100) < 1e-10
    )  # All weights should be equal


def test_estimate_state(filter):
    """Test state estimation from particles."""
    # Set known particle distribution
    filter.particles = np.tile([1.0, 2.0, np.pi / 4], (100, 1))
    filter.weights = np.ones(100) / 100

    # Estimate state
    state = filter.estimate_state()

    # Check estimate
    assert np.allclose(state, [1.0, 2.0, np.pi / 4])


def test_full_cycle(filter):
    """Test a full prediction-update-resample cycle."""
    # Initial state
    initial_particles = filter.particles.copy()

    # Predict
    action = np.array([1.0, 0.2])  # Forward and angular velocity
    filter.predict(action)

    # Update
    z = np.array([[2.0, np.pi / 4], [3.0, -np.pi / 6]])
    filter.update(z)

    # Resample
    filter.resample()

    # Check cycle results
    assert not np.allclose(filter.particles, initial_particles)
    assert np.sum(filter.weights) == pytest.approx(1.0)
    assert len(filter.particles) == 100
