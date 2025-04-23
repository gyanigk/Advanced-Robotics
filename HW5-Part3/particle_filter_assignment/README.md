# Assignment 3: Particle Filter

This assignment focuses on implementing a Particle Filter for robot localization in a multimodal world with complex dynamics and non-Gaussian noise.

## Learning Objectives

- Understand particle-based state estimation
- Implement importance sampling and resampling
- Handle multimodal distributions
- Work with non-Gaussian noise models

## Installation

```bash
pip install -e .
```

## Implementation Tasks

You need to implement several methods in `particle_filter.py`:

1. `predict(action)`: Implement the particle prediction step
   - Propagate each particle through motion model
   - Add motion noise to particles
   - Handle different actions (forward, turn)
   - Account for motion uncertainty

2. `update(readings)`: Implement the measurement update step
   - Calculate particle weights based on sensor readings
   - Normalize weights
   - Handle range-bearing measurements
   - Account for measurement noise

3. `resample()`: Implement particle resampling
   - Use low variance resampling algorithm
   - Maintain particle diversity
   - Handle edge cases (degenerate particles)
   - Reset weights after resampling

4. `estimate_state()`: Implement state estimation
   - Calculate weighted mean of particles
   - Handle circular quantities (angles)
   - Return best state estimate

## Testing Your Implementation

Run the provided test suite:
```bash
pytest tests/test_particle_filter.py -v
```

## Visualization

```python
from filtering_exercises.environments import MultiModalWorld
from filtering_exercises.assignment3_particle import ParticleFilter
from filtering_exercises.utils import FilterVisualizer

# Create environment and filter
env = MultiModalWorld()
pf = ParticleFilter(env, num_particles=100)

# Visualize performance
vis = FilterVisualizer(env, pf)
vis.visualize_episode()
```

## Submission

Submit your implementation of `particle_filter.py` along with:
1. Visualization screenshots showing particle distribution evolution
2. Analysis of how particle count affects performance
3. Discussion of resampling strategy effectiveness
4. Comparison with EKF in multimodal scenarios
5. Investigation of particle depletion problems and solutions 