# Assignment 1: Discrete Bayes Filter

This assignment focuses on implementing a discrete Bayes filter for robot localization in a grid world environment.

## Learning Objectives

- Understand discrete probability distributions
- Implement Bayes' rule for state estimation
- Handle motion and measurement uncertainty
- Work with discrete state spaces

## Installation

```bash
pip install -e ".[assignment1]"
```

## Implementation Tasks

You need to implement two main methods in `bayes_filter.py`:

1. `predict(action)`: Implement the prediction step
   - Update belief state based on robot's action
   - Account for motion uncertainty
   - Handle different actions (forward, turn)
   - Maintain proper probability normalization

2. `update(readings)`: Implement the measurement update step
   - Update belief based on sensor readings
   - Account for measurement noise
   - Handle beam measurements
   - Normalize posterior distribution

## Testing Your Implementation

Run the provided test suite:
```bash
pytest tests/test_bayes_filter.py -v
```

## Visualization

```python
from filtering_exercises.environments import GridWorld
from filtering_exercises.assignment1_bayes import BayesFilter
from filtering_exercises.utils import FilterVisualizer

# Create environment and filter
env = GridWorld()
bf = BayesFilter(env)

# Visualize performance
vis = FilterVisualizer(env, bf)
vis.visualize_episode()
```

## Submission

Submit your implementation of `bayes_filter.py` along with:
1. Visualization screenshots showing your filter in action
2. Brief explanation of your implementation approach
3. Analysis of filter performance in different scenarios 