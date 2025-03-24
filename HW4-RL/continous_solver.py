from hw4_rl.solvers import DiscretizedSolver

# Create a solver for mountain car
solver = DiscretizedSolver(
    # 51,  # Number of bins for discretization
    # _mode='nn'  # or "linear"
)

# Compute the optimal policy
solver.compute_policy()

# Visualize the value function
solver.plot_value_function(solver.solver.get_value_function())