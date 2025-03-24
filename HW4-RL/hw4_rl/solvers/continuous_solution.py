#!/usr/bin/env python
"""
CSCI 5302 HW2: Mountain Car Continuous State Space Solution
Last updated 9/23, 5:00pm

This module implements various reinforcement learning algorithms for solving the Mountain Car
problem with continuous state spaces. It includes value iteration, deterministic policy iteration,
and stochastic policy iteration approaches, all using state space discretization.

The main components are:
- TabularPolicy: A class representing discrete state-action policies
- DiscretizedSolver: A class that handles continuous state space discretization and policy computation
- Various utility functions for policy evaluation and visualization
"""

version = "v2025.02.26.1400"

import copy
import itertools
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Callable

import gymnasium as gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.special import softmax

from hw4_rl.envs import MountainCarEnv

student_name = "My Name"  # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy:
    """
    A tabular policy for discrete state and action spaces.
    
    This class implements a tabular policy that maintains state values, action probabilities,
    and transition/reward functions for a discretized state-action space.
    
    Attributes:
        num_states (int): Total number of discrete states
        num_actions (int): Total number of possible actions
        _transition_function (np.ndarray): State transition probabilities of shape (num_states, num_actions, num_states)
        _reward_function (np.ndarray): Reward values of shape (num_states, num_actions, num_states)
        _value_function (np.ndarray): State values of shape (num_states,)
        _policy (np.ndarray): Action probabilities of shape (num_states, num_actions)
    """

    def __init__(self, num_bins_per_dim: int, num_dims: int, num_actions: int) -> None:
        """
        Initialize the tabular policy.

        Args:
            num_bins_per_dim: Number of bins per state dimension
            num_dims: Number of state dimensions
            num_actions: Number of possible actions
        """
        self.num_states = num_bins_per_dim**num_dims
        self.num_actions = num_actions

        # Initialize transition and reward functions
        self._transition_function = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )
        self._reward_function = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )

        # Initialize value function and policy
        self._value_function = np.zeros(self.num_states)
        self._policy = np.random.uniform(0, 1, size=(self.num_states, self.num_actions))

    def get_action(self, state: int) -> int:
        """
        Sample an action from the policy's distribution at the given state.

        Args:
            state: The current state index

        Returns:
            The sampled action index
        """
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state: int, value: float) -> None:
        """
        Set the value of a given state.

        Args:
            state: The state index
            value: The value to set
        """
        self._value_function[state] = value

    def get_state_value(self, state: int) -> float:
        """
        Get the value of a given state.

        Args:
            state: The state index

        Returns:
            The value of the state
        """
        return self._value_function[state]

    def get_value_function(self) -> np.ndarray:
        """
        Get a copy of the entire value function.

        Returns:
            A copy of the value function array
        """
        return copy.deepcopy(self._value_function)

    def set_value_function(self, values: np.ndarray) -> None:
        """
        Set the entire value function.

        Args:
            values: Array of values to set
        """
        self._value_function = copy.copy(values)

    def set_policy(self, state: int, action_probs: np.ndarray) -> None:
        """
        Set action probabilities for a given state.

        Args:
            state: The state index
            action_probs: Array of action probabilities
        """
        self._policy[state] = copy.copy(action_probs)

    def get_policy(self, state: int) -> np.ndarray:
        """
        Get action probabilities for a given state.

        Args:
            state: The state index

        Returns:
            Array of action probabilities
        """
        return self._policy[state]

    def get_policy_function(self) -> np.ndarray:
        """
        Get a copy of the entire policy function.

        Returns:
            A copy of the policy array
        """
        return copy.deepcopy(self._policy)

    def set_policy_function(self, policy: np.ndarray) -> None:
        """
        Set the entire policy function.

        Args:
            policy: Array of policy values to set
        """
        self._policy = copy.deepcopy(policy)

    def set_transition(self, state: int, action: int, next_state: int, prob: float) -> None:
        """
        Set transition probability for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index
            prob: Transition probability
        """
        self._transition_function[state, action, next_state] = prob

    def get_transition(self, state: int, action: int, next_state: int) -> float:
        """
        Get transition probability for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index

        Returns:
            The transition probability
        """
        return self._transition_function[state, action, next_state]

    def set_reward(self, state: int, action: int, next_state: int, reward: float) -> None:
        """
        Set reward for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index
            reward: Reward value
        """
        self._reward_function[state, action, next_state] = reward

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index

        Returns:
            The reward value
        """
        return self._reward_function[state, action, next_state]


class DiscretizedSolver:
    """
    Solver for continuous state spaces using discretization.
    
    This class implements various reinforcement learning algorithms for solving continuous state
    space problems by discretizing the state space into bins. It supports different policy
    computation methods including value iteration, deterministic policy iteration, and
    stochastic policy iteration.
    
    Attributes:
        _mode (str): Discretization mode ('nn' for nearest neighbor or 'linear' for interpolation)
        _policy_type (str): Type of policy computation method
        temperature (float): Temperature parameter for softmax in stochastic policies
        eps (float): Small value for numerical stability
        _num_bins (int): Number of bins per dimension for discretization
        gamma (float): Discount factor for future rewards
        env (gym.Env): The Gymnasium environment instance
        env_name (str): Name of the environment
        goal_position (float): Target position for the mountain car
        state_lower_bound (np.ndarray): Lower bounds of the state space
        state_upper_bound (np.ndarray): Upper bounds of the state space
        bin_sizes (np.ndarray): Size of bins in each dimension
        num_dims (int): Number of state dimensions
        solver (TabularPolicy): The underlying tabular policy
        performance_history (List[float]): History of performance metrics during training
    """

    def __init__(
        self,
        mode: str,
        num_bins: int = 21,
        temperature: float = 1.0,
        policy_type: str = "deterministic_vi",
    ) -> None:
        """
        Initialize the discretized solver.

        Args:
            mode: Discretization mode ('nn' or 'linear')
            num_bins: Number of bins per dimension for discretization
            temperature: Temperature parameter for softmax in stochastic policies
            policy_type: Type of policy computation ('deterministic_vi', 'stochastic_pi', or 'deterministic_pi')
        
        Raises:
            AssertionError: If mode or policy_type are not valid values
        """
        # Validate inputs
        assert mode in ["nn", "linear"], "Mode must be 'nn' or 'linear'"
        assert policy_type in ["deterministic_vi", "stochastic_pi", "deterministic_pi"]

        # Store parameters
        self._mode = mode
        self._policy_type = policy_type
        self.temperature = temperature
        self.eps = 1e-6
        self._num_bins = num_bins
        self.gamma = 0.95  # Discount factor

        # Initialize environment
        self.env = gym.make("mountaincar5302-v0")
        self.env_name = "MountainCar"
        self.goal_position = 0.5

        # Get state space information
        self.state_lower_bound = self.env.observation_space.low
        self.state_upper_bound = self.env.observation_space.high
        self.bin_sizes = (
            self.state_upper_bound - self.state_lower_bound
        ) / self._num_bins
        self.num_dims = self.state_lower_bound.shape[0]

        # Initialize policy and tracking variables
        self.solver = TabularPolicy(
            self._num_bins, self.num_dims, self.env.action_space.n
        )
        self.performance_history = []

        # Build transition and reward functions
        self.populate_transition_and_reward_funcs()

    def populate_transition_and_reward_funcs(self) -> None:
        """
        Initialize transition and reward functions for all state-action pairs.
        
        This method iterates through all possible state-action pairs and computes
        their transition probabilities and expected rewards through sampling.
        """
        num_states = self._num_bins**self.num_dims
        for state in range(num_states):
            for action in range(self.env.action_space.n):
                self.add_transition(state, action)
                
        # remove later 
        
        state_idx = self.get_state_index_from_coordinates(np.array([0.5, 0.0]))
        for a in range(3):
            for s_next in range(self.solver.num_states):
                r = self.solver.get_reward(state_idx, a, s_next)
                p = self.solver.get_transition(state_idx, a, s_next)
                if p > 0 and r != -1.0:
                    print(f"Goal state {state_idx}, Action {a}, Next {s_next}: Reward {r}")

    def add_transition(self, state_idx: int, action_idx: int) -> None:
        """
        Compute and store transition and reward information for a state-action pair.
        
        This method samples multiple transitions from a given state-action pair to
        estimate transition probabilities and expected rewards.

        Students need to:
        1. Take action_idx in the environment from the current state
        2. Record the next_state and reward from the environment
        3. Append these to next_states_list and rewards_list respectively
        4. Update the transition probabilities and rewards in the solver using weighted averaging

        Args:
            state_idx: Index of the current state
            action_idx: Index of the action to take
        """
        state_vector = self.get_coordinates_from_state_index(state_idx)

        # Sample multiple transitions for better estimates
        n_samples = 5
        next_states_list = []
        rewards_list = []

        for _ in range(n_samples):
            # Reset environment and set state
            self.env.reset()
            self.env.unwrapped.state = np.array(state_vector, dtype=np.float32)

            # Student code here
            # Sample random actions and observe next states and rewards
            # Append to lists
            reward_sum = 0
            for t in range(n_samples):
                next_state, reward, terminated, truncated, _ = self.env.step(action_idx)
                # next_states_list.append(next_state)
                # rewards_list.append(reward)
                reward_sum +=  reward
                if terminated:
                    break
                
            # next_state, reward, terminated, truncated, _ = self.env.step(action_idx)
            next_states_list.append(next_state)
            rewards_list.append(reward_sum/(t+1))

        # Average the transitions and rewards
        for i in range(n_samples):
            next_states = self.get_discrete_state_probabilities(next_states_list[i])
            reward = rewards_list[i]

            for next_state, prob in next_states:
                # Update transition probability
                current_prob = self.solver.get_transition(
                    state_idx, action_idx, next_state
                )
                new_prob = (current_prob * (i) + prob) / (i + 1) if i > 0 else prob
                self.solver.set_transition(state_idx, action_idx, next_state, new_prob)
                # Update the reward and transition probability
                current_reward = self.solver.get_reward(state_idx, action_idx, next_state)
                new_reward = (current_reward * (i) + reward) / (i + 1) if i > 0 else reward
                self.solver.set_reward(state_idx, action_idx, next_state, new_reward)
        

    def get_discrete_state_probabilities(self, continuous_state: np.ndarray) -> List[Tuple[int, float]]:
        """
        Convert continuous state to discrete state probabilities.
        
        For 'nn' mode, returns the nearest discrete state with probability 1.
        For 'linear' mode, returns interpolation weights for neighboring states.

        Students need to:
        For 'nn' mode:
        1. Calculate the nearest discrete state index
        2. Return a list with single tuple of (state_index, 1.0)

        For 'linear' mode:
        1. Find valid neighboring grid points within state bounds
        2. Convert valid neighbors to state indices
        3. Calculate interpolation weights based on distance to neighbors
        4. Return list of (state_index, weight) tuples

        Args:
            continuous_state: The continuous state vector

        Returns:
            List of tuples (state_index, probability) for the discrete states
        """
        if self._mode == "nn":
            # Student code here
            # Find the nearest neighbor
            # pass  # Placeholder for student implementation
            # return [(0, 1.0)]  # Placeholder return value
            state_idx = self.get_state_index_from_coordinates(continuous_state)
            return [(state_idx, 1.0)]
        
        else:  # linear interpolation
            # Get neighboring grid points
            offsets = np.array(
                list(itertools.product([-0.5, 0, 0.5], repeat=self.num_dims))
            )
            neighbor_coords = continuous_state + offsets * self.bin_sizes

            # Find valid neighbors
            # Student code here
            # pass  # Placeholder for student implementation
            
            
        
            # remove later 
            # offsets = np.array(list(itertools.product([-0.5, 0, 0.5], repeat=self.num_dims)))
            # neighbor_coords = continuous_state + offsets * self.bin_sizes
            # # Convert to state indices
            # # Student code here
            # # pass  # Placeholder for student implementation

            # # Calculate interpolation weights
            # # Student code here
            # # pass  # Placeholder for student implementation
            # # Find valid neighbors within bounds
            # valid_mask = np.all((neighbor_coords >= self.state_lower_bound) & 
            #                 (neighbor_coords < self.state_upper_bound), axis=1)
            # valid_coords = neighbor_coords[valid_mask]
            
            # # Convert to state indices
            # state_indices = [self.get_state_index_from_coordinates(coord) 
            #                 for coord in valid_coords]
            
            # # Calculate interpolation weights based on distance
            # distances = np.abs(valid_coords - continuous_state) / self.bin_sizes
            # weights = np.prod(1 - distances, axis=1)
            # weights = weights / np.sum(weights)  # Normalize weights
            # return list(zip(state_indices, weights))
            # return [(0, 1.0)]  # Placeholder return value
            
            # rm
            # valid_neighbors = []
            # for neighbor in neighbor_coords:
            #     if np.all(neighbor >= self.state_lower_bound) and np.all(neighbor < self.state_upper_bound):
            #         valid_neighbors.append(neighbor)
            
            # state_indices = [self.get_state_index_from_coordinates(neighbor) for neighbor in valid_neighbors]
            # # distances = np.abs(np.array(valid_neighbors) - continuous_state) / self.bin_sizes  # Normalized distance per dimension
            # # weights = np.prod(1 - distances, axis=1)  # Linear interpolation: multiply (1 - distance) across dimensions
            # # weights = weights / np.sum(weights)  # Normalize weights to sum to 1
            # # return list(zip(state_indices, weights))
            # distances= np.linalg.norm(np.array(valid_neighbors) - continuous_state, axis=1)
            # inv_distances = np.exp(-distances)
            # weights = inv_distances / np.sum(inv_distances)
            # return list(zip(state_indices, weights))
            # Find valid neighbors within bounds
            valid_mask = np.all((neighbor_coords >= self.state_lower_bound) & 
                            (neighbor_coords < self.state_upper_bound), axis=1)
            valid_coords = neighbor_coords[valid_mask]
            
            if len(valid_coords) == 0:  # Fallback if no valid neighbors
                state_idx = self.get_state_index_from_coordinates(continuous_state)
                return [(state_idx, 1.0)]
            
            # Convert to state indices
            state_indices = [self.get_state_index_from_coordinates(coord) 
                            for coord in valid_coords]
            
            # Calculate distances and convert to probabilities using softmax
            distances = np.linalg.norm(valid_coords - continuous_state, axis=1)
            # Use inverse distance with temperature parameter for better distribution
            temperature = np.mean(self.bin_sizes) / 2  # Adaptive temperature
            inv_distances = -distances / temperature
            weights = softmax(inv_distances)
            
            # Remove duplicates and combine weights
            unique_indices = []
            unique_weights = []
            for idx, weight in zip(state_indices, weights):
                if idx not in unique_indices:
                    unique_indices.append(idx)
                    unique_weights.append(weight)
                else:
                    unique_weights[unique_indices.index(idx)] += weight
            
            # Renormalize weights
            total_weight = sum(unique_weights)
            if total_weight > 0:
                unique_weights = [w / total_weight for w in unique_weights]
            
            return list(zip(unique_indices, unique_weights))

    def compute_policy(self, max_iterations: int = 100, min_iter: int = 5, eval_sample_size: int = 15) -> None:
        """
        Compute optimal policy using the specified algorithm.
        
        This method selects and runs the appropriate policy computation algorithm based on
        the policy_type specified during initialization.

        Args:
            max_iterations: Maximum number of iterations to run
            min_iter: Minimum number of iterations before checking convergence
            eval_sample_size: Number of episodes to use for policy evaluation
        """
        if self._policy_type == "deterministic_vi":
            self._value_iteration(max_iterations, min_iter, eval_sample_size)
        elif self._policy_type == "stochastic_pi":
            self._stochastic_policy_iteration(
                max_iterations, min_iter, eval_sample_size
            )
        else:  # deterministic_pi
            self._deterministic_policy_iteration(
                max_iterations, min_iter, eval_sample_size
            )

    def _value_iteration(self, max_iterations: int, min_iter: int, eval_sample_size: int) -> None:
        """
        Implement value iteration algorithm.

        Students need to:
        1. Compute Q-values for all state-action pairs using:
           - Current value function
           - Transition probabilities
           - Reward function
           - Discount factor (self.gamma)
        2. Update value function with maximum Q-value for each state
        3. Update policy to be deterministic, choosing action with highest Q-value
        4. Check for convergence by comparing old and new value functions

        Args:
            max_iterations: Maximum number of iterations to run
            min_iter: Minimum number of iterations before checking convergence
            eval_sample_size: Number of episodes to use for policy evaluation
        """
        eps_value = 1e-5
        value_function = np.zeros(self.solver.num_states)
        policy = np.zeros((self.solver.num_states, self.solver.num_actions))

        for i in range(max_iterations):
            iter_start_time = time.time()
            
            Q = np.sum(self.solver._transition_function * (self.solver._reward_function + self.gamma * value_function), axis=2)     
            # new_values[state] = np.max(q_values)  # Update value function with max Q-value
            new_values = np.max(Q, axis=1)
            # # Check convergence
            value_diff = np.max(np.abs(new_values - value_function))
            value_function = new_values
            best_action = np.argmax(Q, axis = 1)
            policy = np.zeros((self.solver.num_states, self.solver.num_actions))
            for state in range(self.solver.num_states):
                # action_values = np.sum(self.solver._transition_function[s] * (self.solver._reward_function[s] + self.gamma * value_function[np.newaxis, np.newaxis, :]), axis=1)
                # policy[s] = 0.0
                policy[state, best_action[state]] = 1.0
                
            # Remove later    
            # old_values = value_function.copy()
            # # Compute Q-values
            # # Student code here
            # Q = np.zeros((self.solver.num_states, self.solver.num_actions))
            # for s in range(self.solver.num_states):
            #     for a in range(self.solver.num_actions):
            #         Q[s, a] = np.sum(self.solver._transition_function[s, a] * 
            #                     (self.solver._reward_function[s, a] + 
            #                         self.gamma * value_function))
            # new_values = np.max(Q, axis=1)
            # policy = np.zeros_like(Q)
            # policy[np.arange(self.solver.num_states), np.argmax(Q, axis=1)] = 1.0
            # # Update value function and policy
            # # Check convergence
            # value_diff = np.max(np.abs(new_values - value_function))
            # value_function = new_values
            
            # Evaluate current policy
            self.solver.set_policy_function(policy)
            self.solver.set_value_function(value_function)
            reward, steps = self.solve(max_steps=200, sample_size=eval_sample_size)
            self.performance_history.append(reward)

            print(
                f"VI Iteration {i}, diff {value_diff:.6f}, "
                f"elapsed {time.time() - iter_start_time:.3f}, "
                f"performance {reward:.2f}"
            )
            if i >= min_iter and value_diff < eps_value:
                break

    def _stochastic_policy_iteration(self, max_iterations, min_iter, eval_sample_size):
        """
        Implement stochastic policy iteration with softmax policy.

        Students need to:
        1. Policy Evaluation:
           - Update value function using current policy's transition and reward functions
           - Iterate until convergence or horizon reached
        
        2. Policy Improvement:
           - Compute Q-values using current value function
           - Update policy using softmax over Q-values with temperature parameter
           - Check convergence by comparing old and new policies using KL divergence

        Args:
            max_iterations: Maximum number of iterations to run
            min_iter: Minimum number of iterations before checking convergence
            eval_sample_size: Number of episodes to use for policy evaluation
        """
        eps_value = 1e-5
        eps_policy = 1e-5
        horizon = 100

        print(f"Solver num_states: {self.solver.num_states}, num_actions: {self.solver.num_actions}")
        print(f"Expected 2D states (num_bins^2): {self._num_bins * self._num_bins}")
    
        # Initialize uniform random policy
        policy = (
            np.ones((self.solver.num_states, self.solver.num_actions))
            / self.solver.num_actions
        )
        value_function = np.zeros(self.solver.num_states)

        for k in range(max_iterations):
            iter_start_time = time.time()
            # old_policy = policy.copy()
            # Policy evaluation
            policy_T = np.sum(
                self.solver._transition_function * policy[:, :, np.newaxis], axis=1
            )
            policy_R = np.sum(
                self.solver._reward_function * policy[:, :, np.newaxis], axis=1
            )

            for _ in range(horizon):
                # Student code here 
                # Update value function
                # pass  # Placeholder for student implementation
                delta = 0
                for state in range(self.solver.num_states):
                    new_values = np.sum(
                        policy[state] * np.sum(
                            self.solver._transition_function[state] * (
                                self.solver._reward_function[state] + self.gamma * value_function
                            ),
                            axis = 1
                        )
                    )
                    delta = max(delta, np.abs(new_values - value_function[state]))
                    value_function[state] = new_values
                
                if delta < eps_value:
                    break
                # value_function = policy_R + self.gamma * policy_T.dot(value_function)

            # Policy improvement
            # Student code here
            # pass  # Placeholder for student implementation
            new_policy = np.zeros((self.solver.num_states, self.solver.num_actions))
            for s in range(self.solver.num_states):
                action_values = np.sum(self.solver._transition_function[s] * (
                                self.solver._reward_function[s] + self.gamma * value_function
                            ),
                            axis = 1
                        )
                new_policy[s] = softmax(action_values/self.temperature)

            policy_diff = kl_divergence(policy, new_policy)
            if k > min_iter and policy_diff < eps_policy:
                break

            policy = new_policy
            
            # Q = np.zeros((self.solver.num_states, self.solver.num_actions))
            # for s in range(self.solver.num_states):
            #     for a in range(self.solver.num_actions):
            #         Q[s, a] = np.sum(self.solver._transition_function[s, a] * 
            #                     (self.solver._reward_function[s, a] + 
            #                         self.gamma * value_function))
            
            # policy = softmax(Q / self.temperature, axis=1)
            # Check convergence
            # pass  # Placeholder for student implementation
            # policy_diff = np.mean(np.sum(policy * (np.log(policy + eps_policy) - 
            #                 np.log(old_policy + eps_policy)), axis=1))
            
            # Update and evaluate
            self.solver.set_policy_function(policy)
            self.solver.set_value_function(value_function)
            reward, steps = self.solve(max_steps=200, sample_size=eval_sample_size)
            self.performance_history.append(reward)

            print(
                f"Stochastic PI Iteration {k}, policy_diff {policy_diff:.6f}, "
                f"elapsed {time.time() - iter_start_time:.3f}, "
                f"performance {reward:.2f}"
            )
            if k >= min_iter and policy_diff < eps_policy:
                break
            
    def _deterministic_policy_iteration(
        self, max_iterations, min_iter, eval_sample_size
    ):
        """
        Implement deterministic policy iteration.

        Students need to:
        1. Policy Evaluation:
           - Update value function using current policy's transition and reward functions
           - Iterate until convergence or horizon reached
        
        2. Policy Improvement:
           - Compute Q-values using current value function
           - Update policy to be deterministic, choosing action with highest Q-value
           - Check if policy is stable (unchanged from previous iteration)
           - Set policy_stable flag based on whether policy changed

        Args:
            max_iterations: Maximum number of iterations to run
            min_iter: Minimum number of iterations before checking convergence
            eval_sample_size: Number of episodes to use for policy evaluation
        """
        eps_value = 1e-5
        horizon = 100

        
        # print(f"Solver num_states: {self.solver.num_states}, num_actions: {self.solver.num_actions}")
        # print(f"Expected 2D states (num_bins^2): {self._num_bins * self._num_bins}")
    
        # Initialize random deterministic policy
        policy = np.zeros((self.solver.num_states, self.solver.num_actions))
        policy[
            np.arange(self.solver.num_states),
            np.random.randint(0, self.solver.num_actions, size=self.solver.num_states),
        ] = 1.0
        value_function = np.zeros(self.solver.num_states)

        for k in range(max_iterations):
            iter_start_time = time.time()
            # old_policy = policy.copy()
            
            policy_stable = False
            # Policy evaluation
            policy_T = np.sum(
                self.solver._transition_function * policy[:, :, np.newaxis], axis=1
            )
            policy_R = np.sum(
                self.solver._reward_function * policy[:, :, np.newaxis], axis=1
            )

            # Policy Evaluation
            for _ in range(horizon):
                new_value_function = np.sum(
                                    policy_T * (policy_R + self.gamma * value_function), axis=1
                                )                
                if (np.max(np.abs(new_value_function - value_function)) < eps_value):
                    break
                value_function = new_value_function

            # Policy Improvement
            new_policy = np.zeros((self.solver.num_states, self.solver.num_actions))
            # for state in range(self.solver.num_states):
            #     q_values = np.sum(self.solver._transition_function[state] * (
            #                     self.solver._reward_function[state] + self.gamma * value_function), 
            #                     axis=1)
            for s in range(self.solver.num_states):
                action_values = np.sum(self.solver._transition_function[s] * (
                                self.solver._reward_function[s] + self.gamma * value_function
                            ),
                            axis = 1
                        )
                new_policy[s, np.argmax(action_values)] = 1.0

            # Check if policy is stable
            if np.array_equal(new_policy, policy):
                policy_stable = True
            policy = new_policy
            
            # policy_stable = np.array_equal(new_policy, policy)
            # policy = new_policy.copy()
            
            
            # for _ in range(horizon):
            #     # Student code here
            #     # Update value function
            #     # pass  # Placeholder for student implementation
            #     value_function2 = policy_R + self.gamma * policy_T.dot(value_function)
            
            # if (np.max(np.abs(value_function2 - value_function)) < eps_value):
            #         break    
            # Q = np.zeros((self.solver.num_states, self.solver.num_actions))
            # for s in range(self.solver.num_states):
            #     # for a in range(self.solver.num_actions):
            #     Q = np.sum(self.solver._transition_function[s] * (self.solver._reward_function[s] + self.gamma * value_function))
            
            # policy = np.zeros_like(Q)
            # policy[np.arange(self.solver.num_states), np.argmax(Q, axis=1)] = 1.0      
            # # Policy improvement
            # # Student code here
            # # pass  # Placeholder for student implementation
            # policy_stable = np.all(policy == old_policy)
            # if np.array_equal(policy_new, policy):
            #     policy_stable = True
            # Student code here
            # pass  # Placeholder for student implementation

            # Check convergence
            # pass  # Placeholder for student implementation

            # Update and evaluate
            self.solver.set_policy_function(policy)
            self.solver.set_value_function(value_function)
            reward, steps = self.solve(max_steps=200, sample_size=eval_sample_size)
            self.performance_history.append(reward)

            print(
                f"Deterministic PI Iteration {k}, "
                f"elapsed {time.time() - iter_start_time:.3f}, "
                f"performance {reward:.2f}"
            )

            if policy_stable and k >= min_iter:
                break
            if k > min_iter and reward > self.expected_reward():
                break

        # Final evaluation
        print(f"Value function size: {value_function.size}, Expected: {self._num_bins * self._num_bins}")
        reward, steps = self.solve(max_steps=200)
        print(f"Final policy performance: Reward of {reward:.2f} after {steps} steps.")

    def solve(self, visualize: bool = False, max_steps: float = float("inf"), sample_size: int = 1) -> Tuple[float, float]:
        """
        Execute the current policy in the environment.
        
        This method runs the current policy for one or more episodes and returns the
        average reward and number of steps.

        Args:
            visualize: Whether to render the environment
            max_steps: Maximum number of steps per episode
            sample_size: Number of episodes to run

        Returns:
            Tuple of (average_reward, average_steps)
        """
        rewards = []
        steps = []

        for _ in range(sample_size):
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            if visualize:
                self.env.render()

            episode_reward = 0
            num_steps = 0
            done = False

            while not done and num_steps < max_steps:
                # Get action using current policy
                if self._mode == "nn":
                    discrete_state = self.get_discrete_state_probabilities(state)[0][0]
                    action = self.solver.get_action(discrete_state)
                else:
                    discrete_states = self.get_discrete_state_probabilities(state)
                    states, probs = np.array(discrete_states).T
                    action = self.solver.get_action(int(states[np.argmax(probs)]))

                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = np.array(next_state, dtype=np.float32)
                done = terminated or truncated

                episode_reward += reward
                num_steps += 1

                if visualize:
                    self.env.render()

            rewards.append(episode_reward)
            steps.append(num_steps)

        return np.mean(rewards), np.mean(steps)

    def get_state_index_from_coordinates(self, continuous_state: np.ndarray) -> int:
        """
        Convert continuous state to discrete state index.
        
        This method maps a continuous state vector to its corresponding discrete state index
        based on the discretization scheme.

        Args:
            continuous_state: The continuous state vector

        Returns:
            Index of the corresponding discrete state
        """
        continuous_state = np.clip(
            continuous_state, self.state_lower_bound, self.state_upper_bound - 1e-6
        )

        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        pos_bin = int((continuous_state[0] - self.state_lower_bound[0]) / bin_sizes[0])
        vel_bin = int((continuous_state[1] - self.state_lower_bound[1]) / bin_sizes[1])

        pos_bin = np.clip(pos_bin, 0, self._num_bins - 1)
        vel_bin = np.clip(vel_bin, 0, self._num_bins - 1)

        return pos_bin * self._num_bins + vel_bin

    def get_coordinates_from_state_index(self, state_idx: int) -> np.ndarray:
        """
        Convert discrete state index to continuous state coordinates.
        
        This method maps a discrete state index back to the center of its corresponding
        continuous state region.

        Args:
            state_idx: The discrete state index

        Returns:
            The continuous state vector at the center of the bin
        """
        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        pos_idx = state_idx // self._num_bins
        vel_idx = state_idx % self._num_bins

        position = (pos_idx + 0.5) * bin_sizes[0] + self.state_lower_bound[0]
        velocity = (vel_idx + 0.5) * bin_sizes[1] + self.state_lower_bound[1]

        return np.array([position, velocity], dtype=np.float32)

    def expected_reward(self) -> float:
        """
        Compute expected reward threshold based on discretization.
        
        This method calculates a baseline expected reward threshold that depends on
        the granularity of the discretization.

        Returns:
            The expected reward threshold
        """
        return -110 - (30 * (((40**2) / self.solver.num_states)) ** 0.75)

    def plot_value_function(self, value_function: np.ndarray, filename: Optional[str] = None) -> Tuple[np.ndarray, Figure]:
        """
        Plot the value function as a heatmap.
        
        This method creates a visualization of the value function across the state space
        and optionally saves it to a file.

        Args:
            value_function: Array of state values to plot
            filename: Optional path to save the plot

        Returns:
            Tuple of (image_array, matplotlib_figure)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)

        # Normalize and reshape values
        V = (value_function - value_function.min()) / (
            value_function.max() - value_function.min() + 1e-6
        )
        V = V.reshape(self._num_bins, self._num_bins).T

        # Create heatmap
        image = (plt.cm.coolwarm(V)[::-1, :, :-1] * 255.0).astype(np.uint8)
        ax.set_title(f"Env: {self.env_name}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.imshow(image)

        # Save plot
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir, f"{self.env_name}_{self._mode}_{self._num_bins}.png"
            )

        plt.savefig(filename)
        plt.close()

        # Convert to image array
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).reshape(
            int(fig.get_size_inches()[1] * fig.get_dpi()),
            int(fig.get_size_inches()[0] * fig.get_dpi()),
            4,
        )[:, :, :3]

        return image, fig

    def plot_policy(self, filename: Optional[str] = None) -> Tuple[np.ndarray, Figure]:
        """
        Plot the policy as a heatmap.
        
        This method creates a visualization of the policy across the state space
        and optionally saves it to a file.

        Args:
            filename: Optional path to save the plot

        Returns:
            Tuple of (image_array, matplotlib_figure)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)

        # Create policy grid
        policy_grid = np.zeros((self._num_bins, self._num_bins))
        for i in range(self._num_bins):
            for j in range(self._num_bins):
                state_idx = i * self._num_bins + j
                policy = self.solver.get_policy(state_idx)
                policy_grid[j, i] = np.argmax(policy)

        # Create custom colormap
        colors = ["red", "gray", "blue"]
        cmap = plt.cm.colors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Plot policy
        im = ax.imshow(policy_grid[::-1], cmap=cmap, norm=norm)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=c, label=l)
            for c, l in zip(colors, ["Left", "No Action", "Right"])
        ]
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

        ax.set_title(f"Env: {self.env_name} Policy")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")

        # Save plot
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir,
                f"{self.env_name}_{self._policy_type}_policy_{self._num_bins}.png",
            )

        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")

        # Convert to image array
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).reshape(
            int(fig.get_size_inches()[1] * fig.get_dpi()),
            int(fig.get_size_inches()[0] * fig.get_dpi()),
            4,
        )[:, :, :3]
        plt.close()

        return image, fig


def kl_divergence(policy_old: np.ndarray, policy_new: np.ndarray) -> float:
    """
    Compute KL divergence between two policies.
    
    This function calculates the Kullback-Leibler divergence between two policy
    distributions, which measures how much they differ.

    Args:
        policy_old: The original policy distribution
        policy_new: The new policy distribution

    Returns:
        The KL divergence value
    """
    eps = 1e-10
    policy_old = np.clip(policy_old, eps, 1.0)
    policy_new = np.clip(policy_new, eps, 1.0)
    return np.sum(policy_old * np.log(policy_old / policy_new))


def plot_policy_curves(
    reward_histories: List[List[float]], 
    labels: List[str], 
    filename: Optional[str] = None
) -> None:
    """
    Plot learning curves for different policies.
    
    This function creates a plot comparing the learning progress of different
    policy computation methods over iterations.

    Args:
        reward_histories: List of reward histories for each policy
        labels: List of labels for each policy
        filename: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    plt.clf()
    styles = ["-", "--", ":", "-."]
    colors = ["b", "r", "g", "m", "c", "y", "k"]

    for idx, (history, label) in enumerate(zip(reward_histories, labels)):
        style = styles[idx // len(colors)]
        color = colors[idx % len(colors)]
        plt.plot(range(len(history)), history, style, color=color, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Return")
    plt.title("Policy Iteration Performance")
    plt.legend()
    plt.grid(True)

    if filename is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, "mountaincar_learning_curves.png")

    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    print(
        "Testing Mountain Car with different policy computation methods and bin sizes..."
    )

    #remove later
    # print("Testing Mountain Car with fixed environment...")
    # solver = DiscretizedSolver(mode="nn", num_bins=21, policy_type="deterministic_vi")
    # solver.compute_policy(max_iterations=100, min_iter=5, eval_sample_size=15)
    # reward, steps = solver.solve(max_steps=200, sample_size=10)
    # print(f"Final Results - Reward: {reward:.2f}, Steps: {steps:.2f}")
    
    
    # Configuration
    bin_sizes = [21,51,101]
    temperatures = [0.1]
    mode = ['nn', 'linear']
    curr_mode = mode[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Store results
    all_results = {}

    for n_bins in bin_sizes:
        print(f"\n=== Testing with {n_bins} bins ===")
        bin_histories = []
        bin_labels = []

        # Test each algorithm
        algorithms = [
            ("deterministic_vi", "Value Iteration", None),
            ("deterministic_pi", "Policy Iteration", None),
            ("stochastic_pi", "Stochastic PI", temperatures[0]),
        ]

        for policy_type, label, temp in algorithms:
            print(f"\n--- Testing {label} ---")
            solver = DiscretizedSolver(
                mode=curr_mode,
                num_bins=n_bins,
                policy_type=policy_type,
                temperature=temp if temp is not None else 1.0,
            )

            # Train and evaluate
            start_time = time.time()
            solver.compute_policy()
            elapsed_time = time.time() - start_time
            print(f"Computed {label} Policy in {elapsed_time:.2f} seconds")

            # Plot results
            solver.plot_value_function(
                solver.solver.get_value_function(),
                os.path.join(
                    figures_dir, f"mountaincar_{curr_mode}_{policy_type}_value_{n_bins}.png"
                ),
            )
            solver.plot_policy(
                os.path.join(
                    figures_dir, f"mountaincar_{curr_mode}_{policy_type}_policy_{n_bins}.png"
                )
            )

            # Store results with readable names
            if temp is not None:
                result_key = (label, str(temp), str(n_bins))
            else:
                result_key = (label, "N/A", str(n_bins))
            all_results[result_key] = np.mean(solver.performance_history[-10:])
            bin_histories.append(solver.performance_history)
            bin_labels.append(label)

        # Plot learning curves
        plot_policy_curves(
            bin_histories,
            bin_labels,
            os.path.join(figures_dir, f"mountaincar_learning_curves_{n_bins}_bins.png"),
        )

    # Print final comparison
    print("\n=== Final Performance Comparison ===")
    print("\nBin Size | Algorithm | Temperature | Performance")
    print("-" * 50)
    for (algo, temp, bins), value in sorted(all_results.items()):
        print(f"{bins:8} | {algo:15} | {temp:10} | {value:11.2f}")
