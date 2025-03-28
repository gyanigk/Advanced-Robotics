#!/usr/bin/env python
"""
CSCI 5302 HW2: Tabular Solution Implementation

This module implements various reinforcement learning algorithms for solving gridworld
problems using tabular methods. It includes value iteration, deterministic policy iteration,
and stochastic policy iteration approaches.

The main components are:
- TabularPolicy: A class representing discrete state-action policies
- GridworldSolver: A class that handles policy computation and visualization
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from hw4_rl.envs import GridworldEnv

student_name = "Gyanig Kumar"  # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy:
    """
    A tabular policy implementation for discrete state/action spaces.

    This class implements a tabular policy and value function for reinforcement learning
    in discrete state/action spaces. It maintains mappings from states to values and
    from states to action probability distributions.

    Attributes:
        num_states (int): Total number of discrete states
        num_actions (int): Total number of possible actions
        state_ranges (np.ndarray): Array of [min, max) ranges for each state dimension
        _value_function (np.ndarray): Array mapping states to their values
        _policy (np.ndarray): Array mapping states to action probability distributions
    """

    def __init__(self, n_states: int, state_ranges: np.ndarray, n_actions: int) -> None:
        """
        Initialize the tabular policy.

        Args:
            n_states: Number of discrete states
            state_ranges: Array of [min, max) ranges for each state dimension
            n_actions: Number of possible actions
        """
        self.num_states = n_states
        self.num_actions = n_actions
        self.state_ranges = state_ranges
        
        # Create data structure to store mapping from state to value
        self._value_function = np.zeros(shape=n_states)

        # Create data structure to store array with probability of each action for each state
        self._policy = np.random.uniform(0, 1, size=(n_states, self.num_actions))

    def get_action(self, state: Union[int, np.ndarray]) -> int:
        """
        Sample an action from the policy's action distribution for the given state.

        Args:
            state: The current state index or state coordinate vector

        Returns:
            The sampled action index
        """
        # Convert state to integer if it's a numpy array
        if isinstance(state, np.ndarray):
            state = self.get_state_index_from_coordinates(state)
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1

        # Sample from policy distribution for state
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state: int, value: float) -> None:
        """
        Set the value for a given state.

        Args:
            state: The state index
            value: The value to set for this state
        """
        self._value_function[state] = value

    def get_state_value(self, state: Union[int, np.ndarray]) -> float:
        """
        Get the value for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            The value for the given state
        """
        if isinstance(state, int):
            return self._value_function[state]
        else:
            # Map state vector to state index
            return self._value_function[self.get_state_index_from_coordinates(state)]

    def get_state_index_from_coordinates(self, state: np.ndarray) -> int:
        """
        Convert a state coordinate vector to its corresponding state index.

        Args:
            state: A numpy array containing the (x,y) coordinates of the state

        Returns:
            The integer index corresponding to the state coordinates
        """
        # Convert numpy array to tuple of integers
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        return state[0] * (self.state_ranges[0][1] - self.state_ranges[0][0]) + state[1]

    def get_coordinates_from_state_index(self, state_idx: int) -> np.ndarray:
        """
        Convert a state index to its corresponding coordinate vector.

        Args:
            state_idx: The integer index of the state

        Returns:
            A numpy array containing the (x,y) coordinates corresponding to the state index
        """
        return np.array(
            [
                state_idx // (self.state_ranges[0][1] - self.state_ranges[0][0]),
                state_idx % (self.state_ranges[0][1] - self.state_ranges[0][0]),
            ]
        )

    def get_value_function(self) -> np.ndarray:
        """
        Get a deep copy of the current value function.

        Returns:
            A numpy array representing the value function table, where each entry
            maps a state index to its value
        """
        return copy.deepcopy(self._value_function)

    def set_value_function(self, v: np.ndarray) -> None:
        """
        Set the value function to a new array.

        Args:
            v: A numpy array containing the new value function table
        """
        self._value_function = copy.copy(v)

    def set_policy(self, state: int, action_prob_array: np.ndarray) -> None:
        """
        Set the action probability distribution for a given state.

        Args:
            state: The state index
            action_prob_array: A numpy array containing probabilities for each action
        """
        self._policy[state] = copy.copy(action_prob_array)

    def get_policy(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """
        Get the action probability distribution for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            A numpy array containing probabilities for each action in the given state
        """
        if isinstance(state, int):
            return self._policy[state]
        else:
            # Map state vector to state index
            return self._policy[self.get_state_index_from_coordinates(state)]

    def get_policy_function(self) -> np.ndarray:
        """
        Get a deep copy of the current policy function.

        Returns:
            A numpy array representing the policy table, where each entry maps
            a state to a probability distribution over actions
        """
        return copy.deepcopy(self._policy)

    def set_policy_function(self, p: np.ndarray) -> None:
        """
        Set the policy function to a new array.

        Args:
            p: A numpy array containing the new policy table
        """
        self._policy = copy.copy(p)


class GridworldSolver:
    """
    A solver for gridworld reinforcement learning problems.

    This class implements various policy computation methods for gridworld environments,
    including deterministic value iteration, stochastic policy iteration, and
    deterministic policy iteration.

    Attributes:
        _policy_type (str): Type of policy computation method to use
        env (gym.Env): The gridworld environment
        env_name (str): Name of the environment
        temperature (float): Temperature parameter for stochastic policies
        eps (float): Small constant for numerical stability
        gamma (float): Discount factor for future rewards
        solver (TabularPolicy): The policy object that stores computed policies and values
        performance_history (List[float]): History of cumulative rewards from policy evaluations
    """

    def __init__(
        self,
        policy_type: str = "deterministic_vi",
        gridworld_map_number: int = 0,
        noisy_transitions: bool = False,
        max_ent_temperature: float = 1.0,
    ) -> None:
        """
        Initialize the GridworldSolver.

        Args:
            policy_type: The type of policy computation to use. Must be one of:
                ["deterministic_vi", "stochastic_pi", "deterministic_pi"]
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions
            max_ent_temperature: Temperature parameter for stochastic policies

        Raises:
            AssertionError: If policy_type is not one of the allowed values
        """
        self._policy_type = policy_type
        assert policy_type in ["deterministic_vi", "stochastic_pi", "deterministic_pi"]
        self.env: Optional[gym.Env] = None
        self.env_name = ""
        self.init_environment(gridworld_map_number, noisy_transitions)
        self.temperature = max_ent_temperature
        self.eps = 1e-6
        self.gamma = 0.99

        # Get the unwrapped environment to access its attributes
        assert self.env is not None
        unwrapped_env = self.env.unwrapped
        self.solver = TabularPolicy(
            unwrapped_env.num_states,
            unwrapped_env.get_state_ranges(),
            unwrapped_env.num_actions,
        )
        self.performance_history: List[float] = []

    def init_environment(
        self, gridworld_map_number: int = 0, noisy_transitions: bool = False
    ) -> None:
        """
        Initialize the gridworld environment.

        Args:
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions

        Raises:
            AssertionError: If gridworld_map_number is not 0 or 1
        """
        assert gridworld_map_number in [0, 1]
        if noisy_transitions:
            self.env_name = f"gridworldnoisy-v{gridworld_map_number}"
        else:
            self.env_name = f"gridworld-v{gridworld_map_number}"

        self.env = gym.make(self.env_name)
        self.env.reset()

    def compute_policy(self) -> None:
        """
        Compute optimal policy using the specified algorithm.
        
        This method selects and runs the appropriate policy computation algorithm based on
        the policy_type specified during initialization.
        """
        if self._policy_type == "deterministic_vi":
            self._value_iteration()
        elif self._policy_type == "stochastic_pi":
            self._stochastic_policy_iteration()
        else:  # deterministic_pi
            self._deterministic_policy_iteration()

    def solve(
        self,
        start_state: Optional[np.ndarray] = None,
        visualize: bool = False,
        max_steps: float = float("inf"),
    ) -> Tuple[float, int]:
        """
        Execute the current policy in the environment.
        
        This method runs the current policy from a given start state (or default start state)
        and returns the cumulative reward and number of steps taken.

        Args:
            start_state: Optional starting state coordinates
            visualize: Whether to render the environment
            max_steps: Maximum number of steps to take

        Returns:
            Tuple of (cumulative_reward, num_steps)
        """
        assert self.env is not None
        state, _ = self.env.reset()
        if start_state is not None:
            self.env.unwrapped.change_start_state(start_state)
            state = start_state

        if visualize:
            self.env.render()

        episode_reward = 0
        num_steps = 0
        done = False

        while not done and num_steps < max_steps:
            # Get action using current policy
            action = self.solver.get_action(state)

            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            done = terminated or truncated

            episode_reward += reward
            num_steps += 1

            if visualize:
                self.env.render()

        return episode_reward, num_steps

    def plot_policy_curve(
        self, reward_history: List[float], filename: Optional[str] = None
    ) -> None:
        """
        Plot the learning curve showing policy performance over iterations.

        Args:
            reward_history: List of rewards from each policy evaluation
            filename: Optional path to save the plot
        """
        plt.figure()
        plt.plot(range(len(reward_history)), reward_history)
        plt.xlabel("Iteration")
        plt.ylabel("Return")
        plt.title("Policy Iteration Performance")

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(figures_dir, "gridworld_learning_curve.png")

        plt.savefig(filename)
        plt.close()

    def plot_rollouts(self, filename: Optional[str] = None) -> None:
        """
        Plot state visitation frequencies from policy rollouts.

        Args:
            filename: Optional path to save the plot
        """
        plt.figure()
        plt.plot(range(len(self.performance_history)), self.performance_history)
        plt.xlabel("Iteration")
        plt.ylabel("Return")
        plt.title("Policy Iteration Performance")

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(figures_dir, "gridworld_rollouts.png")

        plt.savefig(filename)
        plt.close()

    def plot_entropy(self, filename: Optional[str] = None) -> None:
        """
        Plot the entropy of the policy over states.

        Args:
            filename: Optional path to save the plot
        """
        policy = self.solver.get_policy_function()
        entropy = -np.sum(policy * np.log(policy + self.eps), axis=1)
        plt.figure()
        plt.plot(range(len(entropy)), entropy)
        plt.xlabel("State")
        plt.ylabel("Entropy")
        plt.title("Policy Entropy")

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(figures_dir, "gridworld_entropy.png")

        plt.savefig(filename)
        plt.close()

    def plot_value_function(
        self, value_function: np.ndarray, filename: Optional[str] = None
    ) -> Tuple[np.ndarray, Figure]:
        """
        Plot the value function as a heatmap.

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
            value_function.max() - value_function.min() + self.eps
        )
        V = V.reshape(
            self.solver.state_ranges[0][1] - self.solver.state_ranges[0][0],
            self.solver.state_ranges[1][1] - self.solver.state_ranges[1][0],
        ).T

        # Create heatmap
        image = (plt.cm.coolwarm(V)[::-1, :, :-1] * 255.0).astype(np.uint8)
        ax.set_title(f"Env: {self.env_name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.imshow(image)

        # Save plot
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir, f"{self.env_name}_{self._policy_type}_value.png"
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

    def _value_iteration(self) -> None:
        """
        Implement value iteration algorithm.
        
        This method iteratively updates state values based on the Bellman optimality
        equation until convergence. The student needs to implement:
        
        1. Value function update using the Bellman optimality equation:
           V(s) = max_a [ sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')] ]
           
        2. Policy update to be deterministic (probability 1 for best action):
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           
        3. Check for convergence by comparing old and new value functions
        
        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        v_i = np.zeros(unwrapped_env.num_states)
        p_i = np.zeros((unwrapped_env.num_states, unwrapped_env.num_actions))

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for _ in range(horizon):
            # Student code here     
            # pass  # Placeholder for student implementation
            delta = 0.0
            v_new = np.zeros_like(v_i) 
            for s in range(unwrapped_env.num_states):
                Q_sa = np.zeros(unwrapped_env.num_actions)
                for a in range(unwrapped_env.num_actions):
                    Q_sa[a] = np.sum(
                        T[s, a, :] * (R[s, a, :] + self.gamma * v_i)
                    )
                    
                # Update value function
                v_new[s] = np.max(Q_sa)
                # Update policy
                best_action = np.argmax(Q_sa)
                p_i[s, :] = 0
                p_i[s, best_action] = 1.0
                # delta = np.abs(v_new[s] - v_i[s])
                
                # delta = 0
                delta = max(delta, np.abs(v_new[s] - v_i[s]))
            v_i = v_new
            # Check convergence 
            if delta<self.eps:
                break
            
            # pass  # Placeholder for student implementation

            
            # pass  # Placeholder for student implementation

        self.solver.set_policy_function(p_i)
        self.solver.set_value_function(v_i)

    def _stochastic_policy_iteration(self) -> None:
        """
        Implement stochastic policy iteration with softmax policy.
        
        This method alternates between policy evaluation and policy improvement steps,
        using a softmax distribution over Q-values for the policy. The student needs to implement:
        
        1. Policy Evaluation: Update value function using current policy:
           V(s) = sum_a pi(s,a) * sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           
        2. Policy Improvement: Update policy using softmax over Q-values:
           pi(s,a) = exp(Q(s,a)/temperature) / sum_a' exp(Q(s,a')/temperature)
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           
        The temperature parameter controls exploration vs exploitation:
        - High temperature -> More uniform/exploratory policy
        - Low temperature -> More deterministic/exploitative policy
        
        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        # Initialize with a uniform random policy
        p_i = (
            np.ones((unwrapped_env.num_states, unwrapped_env.num_actions))
            / unwrapped_env.num_actions
        )
        temp_reciprocal = 1.0 / self.temperature

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for k in range(50):
            print("Stochastic Policy Iteration %d" % k)

            # Policy Evaluation
            elapsed = time.time()
            v_i = np.zeros(unwrapped_env.num_states)

            for _ in range(horizon):
                # Student code here
                # Update value function
                # pass  # Placeholder for student implementation
                new_v = np.zeros(unwrapped_env.num_states)
                for s in range(unwrapped_env.num_states):
                    state_val = 0.0
                    for a in unwrapped_env.actions:
                        Q_sa = np.sum(T[s,a, :]*(R[s,a, :] + self.gamma * v_i))
                        state_val += p_i[s,a] * Q_sa
                    new_v[s] = state_val
                    print(state_val,"checking updates")
                v_i = new_v
                
                
            elapsed = time.time() - elapsed
            print(".....Evaluate done in %g" % elapsed)
            elapsed = time.time()
            
            # Policy Improvement with Softmax
            # Student code here
            # pass  # Placeholder for student implementation
            for s in range(unwrapped_env.num_states):
                Q_sa = np.zeros(unwrapped_env.num_actions)
                for a in range(unwrapped_env.num_actions):
                    Q_sa[a] = np.sum(
                        T[s, a, :] * (R[s, a, :] + self.gamma * v_i)
                    )
                    # print("Q_sa : ",Q_sa,"checking Q values")
                max_Q = np.max(Q_sa)
                exp_Q = np.exp((Q_sa - max_Q) * temp_reciprocal)
                p_i[s, :] = exp_Q / np.sum(exp_Q) # Softmax policy
                # print("p_i : ",p_i[s,:],"checking policy updates")                
                
            
            
            # Compute softmax policy with temperature
            # Student code here
            # pass  # Placeholder for student implementation

            elapsed = time.time() - elapsed
            print(".....Improve done in %g" % elapsed)

            self.solver.set_policy_function(p_i)
            self.performance_history.append(self.solve(max_steps=20)[0])

        self.solver.set_value_function(v_i)

    def _deterministic_policy_iteration(self) -> None:
        """
        Implement deterministic policy iteration.
        
        This method alternates between policy evaluation and policy improvement steps,
        selecting the best action in each state deterministically. The student needs to implement:
        
        1. Policy Evaluation: Update value function using current policy:
           V(s) = sum_a pi(s,a) * sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           Note: Since policy is deterministic, this simplifies to:
           V(s) = sum_s' T(s,a*,s')[R(s,a*,s') + gamma * V(s')]
           where a* is the action with probability 1 in state s
           
        2. Policy Improvement: Update policy to be deterministic for best action:
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
        
        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        p_i = self.solver.get_policy_function()

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for k in range(50):
            print("Policy Iteration %d" % k)

            # Policy Evaluation
            elapsed = time.time()
            v_i = np.zeros(unwrapped_env.num_states)

            for _ in range(horizon):
                # Get expected value for current policy
                # Student code here
                # pass  # Placeholder for student implementation
                
                #running the iterations for the value function
                # print("v_i : ",v_i,"checking value function updates")
                new_v = np.zeros(unwrapped_env.num_states)
                for s in range(unwrapped_env.num_states):
                    a_star = np.argmax(p_i[s])
                    new_v[s] = np.sum(T[s, a_star, :] * (R[s, a_star, :] + self.gamma * v_i))
                v_i = new_v
                
            elapsed = time.time() - elapsed
            print(".....Evaluate done in %g" % elapsed)
            elapsed = time.time()

            # Policy Improvement
            # Student code here
            # pass  # Placeholder for student implementation
            for s in range(unwrapped_env.num_states):
                Q_sa = np.zeros(unwrapped_env.num_actions)
                for a in range(unwrapped_env.num_actions):
                    Q_sa[a] = np.sum(
                        T[s, a, :] * (R[s, a, :] + self.gamma * v_i)
                    )
                best_action = np.argmax(Q_sa)
                p_i[s, :] = 0
                p_i[s, best_action] = 1.0
            
            elapsed = time.time() - elapsed
            print(".....Improve done in %g" % elapsed)

            self.solver.set_policy_function(p_i)
            self.performance_history.append(self.solve(max_steps=20)[0])

        self.solver.set_value_function(v_i)


if __name__ == "__main__":

    ############ Q1.1 ############
    gw0_det_solver = GridworldSolver(
        policy_type="stochastic_pi", gridworld_map_number=0
    )
    gw1_det_solver = GridworldSolver(
        policy_type="stochastic_pi", gridworld_map_number=1
    )
    for solver in [gw0_det_solver, gw1_det_solver]:
        print("Starting!")
        start_time = time.time()
        solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q1.a VI Policy in %g seconds" % elapsed_time)
        solver.plot_value_function(solver.solver.get_value_function())


    ############ Q1.2 ############
    gw0_solver = GridworldSolver(policy_type="stochastic_pi", gridworld_map_number=0)

    for solver in [gw0_solver]:
        start_time = time.time()
        solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q2 PI Policy in %g seconds" % elapsed_time)
        solver.plot_policy_curve(solver.performance_history)
