import numpy as np
import pytest

from filtering_exercises.assignment1_bayes.bayes_filter import BayesFilter
from filtering_exercises.environments import GridWorld


@pytest.fixture
def env():
    return GridWorld(size=5)


@pytest.fixture
def filter(env):
    return BayesFilter(env)


def test_initial_belief(filter):
    """Test that initial belief is uniform."""
    belief = filter.belief
    expected = np.ones((5, 5)) / 25
    np.testing.assert_array_almost_equal(belief, expected)


def test_predict_forward(filter):
    """Test prediction step with forward action."""
    # Set initial belief to known state
    filter.belief = np.zeros((5, 5))
    filter.belief[2, 2] = 1.0

    # Predict forward motion
    filter.predict(0)  # 0 = forward

    # Check that belief has spread forward
    assert filter.belief[3, 2] > 0.1  # Most probability mass moved forward
    assert filter.belief[2, 2] < 0.3  # Some probability remained
    assert np.sum(filter.belief) == pytest.approx(1.0)  # Normalized


def test_predict_turn(filter):
    """Test prediction step with turn action."""
    # Set initial belief to known state
    filter.belief = np.zeros((5, 5))
    filter.belief[2, 2] = 1.0

    # Predict turn motion
    filter.predict(1)  # 1 = turn right

    # Check that belief stayed in same cell but spread
    assert filter.belief[2, 2] > 0.7  # Most probability mass stayed
    assert np.sum(filter.belief) == pytest.approx(1.0)  # Normalized


def test_update(filter, env):
    """Test measurement update step."""
    # Set initial belief to uniform
    filter.belief = np.ones((5, 5)) / 25

    # Generate fake measurement
    readings = [2, 3, 1, 2]  # Example beam readings

    # Update belief
    filter.update(readings)

    # Check that belief is still valid
    assert np.sum(filter.belief) == pytest.approx(1.0)  # Normalized
    assert np.all(filter.belief >= 0)  # All probabilities non-negative


def test_full_cycle(filter, env):
    """Test a full prediction-update cycle."""
    # Initial uniform belief
    assert np.sum(filter.belief) == pytest.approx(1.0)

    # Predict forward motion
    filter.predict(0)  # 0 = forward
    assert np.sum(filter.belief) == pytest.approx(1.0)

    # Update with measurement
    readings = [2, 3, 1, 2]  # Example beam readings
    filter.update(readings)
    assert np.sum(filter.belief) == pytest.approx(1.0)

    # Belief should be more concentrated after update
    assert np.max(filter.belief) > 1 / 25  # More concentrated than uniform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
