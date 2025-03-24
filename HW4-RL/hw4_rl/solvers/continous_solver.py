# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Assuming the previous code is in a file called discretized_solver.py
# from hw4_rl.solvers import DiscretizedSolver
# from hw4_rl.envs import MountainCarEnv

# from hw4_rl.solvers import DiscretizedSolver
# version = "v2025.02.26.1400"

# import copy
# import itertools
# import os
# import time
# from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Callable

# import gymnasium as gym
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# from scipy.special import softmax

from hw4_rl.envs import MountainCarEnv
from hw4_rl.solvers import DiscretizedSolver, TabularPolicy


from hw4_rl.solvers import DiscretizedSolver

# Create a solver for mountain car
solver = DiscretizedSolver(
    num_bins=51,  # Number of bins for discretization
    mode="nn",  # or "linear"
    policy_type= "stochastic_pi"
)

# Compute the optimal policy
solver.compute_policy()

# Visualize the value function
solver.plot_value_function(solver.solver.get_value_function())

# student_name = "Gyanig Kumar"  # Set to your name
# GRAD = True  # Set to True if graduate student

# temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
# bin_sizes = [21, 31, 51, 101]
# algorithms = [
#     ("deterministic_vi", "Value Iteration", None),
#     ("deterministic_pi", "Policy Iteration", None),
#     ("stochastic_pi", "Stochastic PI", temperatures[0]),
# ]
# policy_typef = algorithms[0][0]
# n_bins = bin_sizes[0]
# print(n_bins)
# script_dir = os.path.dirname(os.path.abspath(__file__))
# figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
# print(script_dir,"\n",figures_dir)
# # Create a solver for mountain car
# solver = DiscretizedSolver(
#     num_bins=31,  # Number of bins for discretization
#     mode="linear",  # or "linear"
#     policy_type = policy_typef
# )

# label = "Value Iteration"
# # Compute the optimal policy
# start_time = time.time()
# solver.compute_policy()
# elapsed_time = time.time() - start_time
# print(f"Computed {label} Policy in {elapsed_time:.2f} seconds")
# solver.plot_value_function(solver.solver.get_value_function())
# solver.plot_policy()
# reward, steps = solver.solve()
# print(f"Final Results - Reward: {reward:.2f}, Steps: {steps:.2f}")

# # Visualize the value function
# solver.plot_value_function(solver.solver.get_value_function())

# solver.plot_value_function(solver.solver.get_value_function(),os.path.join(figures_dir, f"mountaincar_{policy_type}_value_{n_bins}.png"))
# solver.plot_policy(os.path.join(figures_dir, f"mountaincar_{policy_type}_policy_{n_bins}.png"))
# print(
#         "Testing Mountain Car with different policy computation methods and bin sizes..."
#     )

#remove later
# print("Testing Mountain Car with fixed environment...")
# solver = DiscretizedSolver(mode="nn", num_bins=21, policy_type="deterministic_vi")
# solver.compute_policy(max_iterations=100, min_iter=5, eval_sample_size=15)
# reward, steps = solver.solve(max_steps=200, sample_size=10)
# print(f"Final Results - Reward: {reward:.2f}, Steps: {steps:.2f}")


# Configuration
# bin_sizes = [31,51,101]
# temperatures = [1.0]
# script_dir = os.path.dirname(os.path.abspath(__file__))
# figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
# os.makedirs(figures_dir, exist_ok=True)

# # Store results
# all_results = {}

# for n_bins in bin_sizes:
#     print(f"\n=== Testing with {n_bins} bins ===")
#     bin_histories = []
#     bin_labels = []

#     # Test each algorithm
#     algorithms = [
#         ("deterministic_vi", "Value Iteration", None),
#         ("deterministic_pi", "Policy Iteration", None),
#         ("stochastic_pi", "Stochastic PI", temperatures[0]),
#     ]

#     for policy_type, label, temp in algorithms:
#         print(f"\n--- Testing {label} ---")
#         solver = DiscretizedSolver(
#             mode="nn",
#             num_bins=n_bins,
#             policy_type=policy_type,
#             temperature=temp if temp is not None else 1.0,
#         )

#         # Train and evaluate
#         start_time = time.time()
#         solver.compute_policy()
#         elapsed_time = time.time() - start_time
#         print(f"Computed {label} Policy in {elapsed_time:.2f} seconds")

#         # Plot results
#         solver.plot_value_function(
#             solver.solver.get_value_function(),
#             os.path.join(
#                 figures_dir, f"mountaincar_{policy_type}_value_{n_bins}.png"
#             ),
#         )
#         solver.plot_policy(
#             os.path.join(
#                 figures_dir, f"mountaincar_{policy_type}_policy_{n_bins}.png"
#             )
#         )

#         # Store results with readable names
#         if temp is not None:
#             result_key = (label, str(temp), str(n_bins))
#         else:
#             result_key = (label, "N/A", str(n_bins))
#         all_results[result_key] = np.mean(solver.performance_history[-10:])
#         bin_histories.append(solver.performance_history)
#         bin_labels.append(label)

#     # Plot learning curves
#     plot_policy_curves(
#         bin_histories,
#         bin_labels,
#         os.path.join(figures_dir, f"mountaincar_learning_curves_{n_bins}_bins.png"),
#     )

# # Print final comparison
# print("\n=== Final Performance Comparison ===")
# print("\nBin Size | Algorithm | Temperature | Performance")
# print("-" * 50)
# for (algo, temp, bins), value in sorted(all_results.items()):
#     print(f"{bins:8} | {algo:15} | {temp:10} | {value:11.2f}")

# env = MountainCarEnv()
# state, _ = env.reset()
# total_reward = 0
# for _ in range(200):
#     state, reward, done, truncated, _ = env.step(2)  # Always push right
#     total_reward += reward
#     if done:
#         print(f"Reached goal! Steps: {_ + 1}, Reward: {total_reward}")
#         break
# else:
#     print(f"Failed to reach goal. Reward: {total_reward}")


# def run_solver(mode='nn', num_bins=21, policy_type='deterministic_vi', temperature=1.0):
#     """
#     Run the solver with specified parameters and return results.

#     Args:
#         mode: Discretization mode ('nn' or 'linear')
#         num_bins: Number of bins per dimension
#         policy_type: Type of policy computation
#         temperature: Temperature for stochastic policies

#     Returns:
#         solver: Trained DiscretizedSolver instance
#         reward: Final reward
#         steps: Final steps
#     """
#     print(f"\nRunning {policy_type} with {mode} discretization ({num_bins} bins)")

#     solver = DiscretizedSolver(
#         mode=mode,
#         num_bins=num_bins,
#         temperature=temperature,
#         policy_type=policy_type
#     )

#     solver.compute_policy(
#         max_iterations=100,
#         min_iter=5,
#         eval_sample_size=15
#     )

#     reward, steps = solver.solve(
#         visualize=False,
#         max_steps=200,
#         sample_size=10
#     )

#     print(f"Final Results - Reward: {reward:.2f}, Steps: {steps:.2f}")
#     return solver, reward, steps

# def plot_performance(solvers, labels):
#     """
#     Plot performance history for multiple solvers.

#     Args:
#         solvers: List of DiscretizedSolver instances
#         labels: List of labels for each solver
#     """
#     plt.figure(figsize=(10, 6))
#     for solver, label in zip(solvers, labels):
#         plt.plot(solver.performance_history, label=label)

#     plt.xlabel('Iteration')
#     plt.ylabel('Average Reward')
#     plt.title(f'Performance History')
#     plt.legend()
#     plt.grid(True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f'performance_history_{timestamp}.png'
#     plt.savefig(filename)
#     print(f"Performance plot saved as {filename}")
#     plt.close()

# def print_summary(configurations, results):
#     """
#     Print a formatted summary table of results.

#     Args:
#         configurations: List of configuration dictionaries
#         results: List of tuples (solver, reward, steps)
#     """
#     print("\nSummary of Results:")
#     print("-" * 70)
#     print("| {:<15} | {:<10} | {:<12} | {:<12} |".format(
#         "Policy Type", "Mode", "Reward", "Steps"))
#     print("-" * 70)

#     for config, (_, reward, steps) in zip(configurations, results):
#         print("| {:<15} | {:<10} | {:<12.2f} | {:<12.2f} |".format(
#             config['policy_type'],
#             config['mode'],
#             reward,
#             steps
#         ))
#     print("-" * 70)

# def main():
#     # Configuration for different runs
#     configurations = [
#         {
#             'mode': 'nn',
#             'num_bins': 21,
#             'policy_type': 'deterministic_vi',
#             'temperature': 1.0
#         },
#         {
#             'mode': 'nn',
#             'num_bins': 21,
#             'policy_type': 'stochastic_pi',
#             'temperature': 1.0
#         },
#         {
#             'mode': 'nn',
#             'num_bins': 21,
#             'policy_type': 'deterministic_pi',
#             'temperature': 1.0
#         }
#     ]

#     if True:
#         configurations.extend([
#             {
#                 'mode': 'linear',
#                 'num_bins': 21,
#                 'policy_type': 'deterministic_vi',
#                 'temperature': 1.0
#             },
#             {
#                 'mode': 'linear',
#                 'num_bins': 21,
#                 'policy_type': 'stochastic_pi',
#                 'temperature': 1.0
#             }
#         ])

#     # Run all configurations
#     solvers = []
#     labels = []
#     results = []

#     for config in configurations:
#         solver, reward, steps = run_solver(
#             mode=config['mode'],
#             num_bins=config['num_bins'],
#             policy_type=config['policy_type'],
#             temperature=config['temperature']
#         )
#         solvers.append(solver)
#         labels.append(f"{config['policy_type']} ({config['mode']})")
#         results.append((solver, reward, steps))
    
#         # Plot value function
#         image, _ = solver.plot_value_function(solver.solver.get_value_function())

#     # Plot performance comparison
#     plot_performance(solvers, labels)

#     # Print formatted summary
#     print_summary(configurations, results)

# if __name__ == "__main__":
#     main()