from hw4_rl.solvers import GridworldSolver

# Create a solver for gridworld
solver = GridworldSolver(
    policy_type="deterministic_vi",  # or "stochastic_pi" or "deterministic_pi"
    gridworld_map_number=1,  # or 1
    noisy_transitions=False
)

# Compute the optimal policy
solver.compute_policy()

# Visualize the value function
solver.plot_value_function(solver.solver.get_value_function())