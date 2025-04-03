# Assignment 2: Extended Kalman Filter

This assignment focuses on implementing an Extended Kalman Filter (EKF) for robot localization in a continuous world with nonlinear dynamics.

## Learning Objectives

- Understand nonlinear state estimation
- Implement linearization through Jacobian matrices
- Handle continuous state spaces
- Work with Gaussian distributions and uncertainty

## Installation

```bash
pip install -e ".[assignment2]"
```

## Implementation Tasks

You need to implement several methods in `extended_kalman_filter.py`:

1. `predict(action)`: Implement the EKF prediction step
   - Update state mean using nonlinear motion model
   - Compute motion model Jacobian
   - Update covariance matrix
   - Handle different actions (forward, turn)

2. `update(readings)`: Implement the EKF measurement update step
   - Compute measurement Jacobian
   - Calculate Kalman gain
   - Update state estimate and covariance
   - Handle range-bearing measurements


## Testing Your Implementation

Run the provided test suite:
```bash
pytest tests/test_extended_kalman_filter.py -v
```

## Visualization

```python
from filtering_exercises.environments import NonlinearWorld
from filtering_exercises.assignment2_ekf import ExtendedKalmanFilter
from filtering_exercises.utils import FilterVisualizer

# Create environment and filter
env = NonlinearWorld()
ekf = ExtendedKalmanFilter(env)

# Visualize performance
vis = FilterVisualizer(env, ekf)
vis.visualize_episode()
```

