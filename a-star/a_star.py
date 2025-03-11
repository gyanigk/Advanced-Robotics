"""
A* Grid Path Planning Implementation

This module implements the A* path planning algorithm on a 2D grid with obstacles.
The implementation allows for diagonal movements and uses a Euclidean distance
heuristic to guide the search.

COEN 5830 HW2: Path Planning

Instructions:
------------
1. Add your name below
2. Complete all sections marked with "Student Task:" comments/docstrings
3. The main algorithm steps are in the `planning` method
4. Test your implementation with different obstacle configurations in main()

Full Name: GYANIG KUMAR


Full Name: GYANIG KUMAR
Answers for Part 1:
1. Implementation done
2. Created more obstacles in the environment to make the a* star to take time to go through the maze
3. Yes, Dijkstra’s algorithm can outperform A* under certain conditions. For example, if A* uses a non-optimal or expensive cost function heuristic 
(contribute like a zero heuristic), it behaves like Dijkstra’s but with added overhead for calculating the heuristic at every node. 
Consider a large, obstacle-free environment—say, a 100×100 grid with same cost function. In this scenario, the heuristic provides no useful guidance, 
so A* expands nodes in the same manner as Dijkstra’s algorithm. However, the extra time spent computing the heuristic for each node can make A* slower overall, 
thereby giving Dijkstra’s algorithm a performance advantage.

Answers for Part 2:
1. Implementation done
2. For Holonomic, we do not consider the expand_dis does not contribute. But, for nonholonomic, the expand_dis if set to 0.1, is not able to create curved paths enough
to be able avoid obstacles. Expand_Dis contributes to extention of the tree at each iterations. If it is small, then it would be able to create the essential extention.
This slows down the contruction of the path when exploring the state space. There are two computationally expensive ways to overcome this, i.e 
    a) Increasing the max_interations 
    b) Decreasing the path_resolution
    to able to connect the start and end states. 

Answers for Part 3:
1. Implementation done. 
"""

import math
from typing import Dict, List, Optional, Tuple
import time
import matplotlib.pyplot as plt

show_animation = True


class AStarPlanner:
    """A* path planning algorithm implementation on a 2D grid."""

    def __init__(
        self,
        obstacle_x: List[float],
        obstacle_y: List[float],
        resolution: float,
        robot_radius: float,
    ):
        """
        Initialize the A* path planner.

        Args:
            obstacle_x: List of x coordinates of obstacles
            obstacle_y: List of y coordinates of obstacles
            resolution: Grid resolution (cell size)
            robot_radius: Radius of the robot for collision checking

        Class Attributes:
            self.resolution: Size of each grid cell (determines grid granularity)
            self.robot_radius: Robot's radius used for collision checking with obstacles
            self.min_x: Minimum x-coordinate of the grid world
            self.min_y: Minimum y-coordinate of the grid world
            self.max_x: Maximum x-coordinate of the grid world
            self.max_y: Maximum y-coordinate of the grid world
            self.obstacle_map: 2D boolean array where True indicates presence of obstacle
            self.x_width: Number of grid cells in x direction
            self.y_width: Number of grid cells in y direction
            self.motion: List of possible movement directions and their costs
        """
        # Grid resolution - determines size of each grid cell
        self.resolution = resolution
        # Robot's physical size for collision checking
        self.robot_radius = robot_radius
        # Grid world boundaries
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        # 2D array marking obstacle locations
        self.obstacle_map = None
        # Grid dimensions in cells
        self.x_width, self.y_width = 0, 0
        # Create obstacle map from given coordinates
        self.calc_obstacle_map(obstacle_x, obstacle_y)
        # Get possible movement directions
        self.motion = self.get_motion_model()

    class Node:
        """A node in the A* search grid."""

        def __init__(self, x: int, y: int, cost: float, parent_index: int):
            """
            Initialize a search node.

            Args:
                x: Grid x coordinate
                y: Grid y coordinate
                cost: Cost to reach this node from start
                parent_index: Index of parent node in the closed set
            """
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

    def planning(
        self, start_x: float, start_y: float, goal_x: float, goal_y: float
    ) -> Tuple[List[float], List[float]]:
        """
        Plan a path from start to goal position using A* algorithm.

        Args:
            start_x: Start x coordinate
            start_y: Start y coordinate
            goal_x: Goal x coordinate
            goal_y: Goal y coordinate

        Returns:
            rx, ry: Lists of x and y coordinates of the path from goal to start

        -------------------------------------------------------
        Student Task: Complete the A* algorithm implementation:
        1. Initialize start and goal nodes
        2. Maintain open_set for nodes to be expanded
        3. Maintain closed_set for already expanded nodes
        4. While open_set is not empty:
           a. Find node with minimum f_cost (g_cost + heuristic)
           b. If node is goal, backtrack to find path
           c. Expand node's neighbors and update costs


        Please refer to the A* algorithm pseudocode in the course notes
        for details on how the algorithm works.
        """
        start_node = self.Node(
            self.calc_xy_index(start_x, self.min_x),
            self.calc_xy_index(start_y, self.min_y),
            0.0,
            -1,
        )
        goal_node = self.Node(
            self.calc_xy_index(goal_x, self.min_x),
            self.calc_xy_index(goal_y, self.min_y),
            0.0,
            -1,
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            current = None
            # YOUR CODE GOES HERE
            # Check if open_set is empty. If so, break out of the while loop
            # Find the node in open_set with least cost to come (g) + cost to go (heuristic) and assign it to current
            if not open_set:
                break

            current_index = min(open_set, key=lambda idx: open_set[idx].cost + self.calc_heuristic(open_set[idx],goal_node))
            current = open_set[current_index]

            # DO NOT ALTER THE FOLLOWING LINES
            if show_animation:
                plt.plot(
                    self.calc_grid_position(current.x, self.min_x),
                    self.calc_grid_position(current.y, self.min_y),
                    "xc",
                )
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)
            # YOUR CODE GOES HERE
            # Check if current Node is equal to the goal. If so, assign goal_node.parent_index and goal_node.cost to the appropriate values and break out of the while loop
            # If current node isnt the goal, remove the current node from the open set and add it to the closed set
            # Use the motion model to expand the search to other neighbors in the grid.
            # Check if the neighboring cell is a part of closed set. If so, move on.
            # If the neighboring cell is not within bounds of the state space, move on.
            # If the neighboring cell is neither in open_set or closed_set, add it to the open set.
            # If the neighboring cell is a part of the open cell, will expanding from the current node reduce the total cost to reach the neighbor? If so, replace it with the current node. (Essentially changing its parent and cost).
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                goal_node = current
                break

            # 2. Move current node from open_set to closed_set
            closed_set[current_index] = current
            del open_set[current_index]

            # 3. Expand neighbors using motion model:
            #    - Skip if in closed_set
            #    - Skip if out of bounds
            #    - Add to open_set if new
            #    - Update if better path found
            for move in self.motion:
                node_x = current.x + int(move[0])
                node_y = current.y + int(move[1])
                node_cost = current.cost + move[2]

                node = self.Node(node_x,node_y,node_cost,current_index)
                node_index = self.calc_grid_index(node)

                if node_index in closed_set:
                    continue

                if not self.verify_node(node):
                    continue
                
                if node_index not in open_set:
                    open_set[node_index] = node
                else:
                    if open_set[node_index].cost > node.cost:
                        open_set[node_index] =node
        
        
        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(
        self, goal_node: Node, closed_set: Dict[int, Node]
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate the final path from start to goal node.

        Args:
            goal_node: The goal node
            closed_set: Set of expanded nodes

        Returns:
            rx, ry: Lists of x and y coordinates of the path
        """
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)
        ]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1: Node, n2: Node) -> float:
        """
        Calculate heuristic distance between two nodes.

        Args:
            n1: First node
            n2: Second node

        Returns:
            Weighted Euclidean distance between nodes
        """
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index: int, min_position: float) -> float:
        """
        Calculate grid position from index.

        Args:
            index: Grid index
            min_position: Minimum position value

        Returns:
            Actual position coordinate
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position: float, min_pos: float) -> int:
        """
        Calculate grid index from position.

        Args:
            position: Actual position coordinate
            min_pos: Minimum position value

        Returns:
            Grid index
        """
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node: Node) -> int:
        """
        Calculate unique index for grid position.

        Args:
            node: Node to calculate index for

        Returns:
            Unique grid index
        """
        return node.y * self.x_width + node.x

    def verify_node(self, node: Node) -> bool:
        """
        Verify if a node is valid for path planning.

        This function checks two conditions:
        1. If the node's coordinates are within the grid boundaries
        2. If the node's position doesn't collide with any obstacles

        Args:
            node: The node to verify, containing x and y coordinates in grid units

        Returns:
            bool: True if the node is valid (within bounds and collision-free),
                 False otherwise

        Student Task:
        ------------
        1. Check if node coordinates (node.x, node.y) are within grid bounds:
           - x should be between 0 and self.x_width
           - y should be between 0 and self.y_width
        2. If within bounds, check self.obstacle_map to see if the node
           position collides with an obstacle
        3. Return False if either check fails, True if both pass
        """
        # YOUR CODE GOES HERE
        # Verify if the node is within bounds and isn't colliding with an obstacle
        # Return False if node is invalid. Otherwise, return True
        if node.x<0 or node.x>=self.x_width:
            return False
        if node.y<0 or node.y>=self.y_width:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True 
        pass

    def calc_obstacle_map(
        self, obstacle_x: List[float], obstacle_y: List[float]
    ) -> None:
        """
        Calculate a grid-based obstacle map from obstacle coordinates.

        This function converts continuous obstacle coordinates into a discrete grid
        representation where each cell is marked as either containing an obstacle or not.

        Args:
            obstacle_x: List of x-coordinates of obstacles in world units
            obstacle_y: List of y-coordinates of obstacles in world units

        Note:
            - The function updates several class attributes:
              * self.min_x, self.min_y: Minimum x,y coordinates of the grid
              * self.max_x, self.max_y: Maximum x,y coordinates of the grid
              * self.x_width, self.y_width: Width and height of the grid
              * self.obstacle_map: 2D grid marking obstacle positions

        Student Task:
        ------------
        1. Find grid boundaries:
           - Calculate min/max x,y from obstacle coordinates
           - Convert to grid coordinates using self.resolution
           - Store in self.min_x, self.min_y, self.max_x, self.max_y

        2. Calculate grid dimensions:
           - Use min/max values to compute grid width and height
           - Store in self.x_width and self.y_width

        3. After the obstacle_map is initialized (DO NOT MODIFY THE INITIALIZATION):
           - Convert each obstacle coordinate to grid coordinates
           - Mark corresponding cells in obstacle_map as True
        """
        # Find the minimum and maximum bounds for x and y
        # Use the bounds to obtain the width along x and y axes and store them in self.x_width and self.y_width respectively
        self.min_x = round(min(obstacle_x))
        self.min_y = round(min(obstacle_y))
        self.max_x = round(max(obstacle_x))
        self.max_y = round(max(obstacle_y))


        self.x_width = int((self.max_x - self.min_x) / self.resolution) + 1
        self.y_width = int((self.max_y - self.min_y) / self.resolution) + 1

        # DO NOT ALTER THE NEXT TWO LINES.
        self.obstacle_map = [
            [False for _ in range(self.y_width)] for _ in range(self.x_width)
        ]
        # pass
        # For each cell in self.obstacle_map, use the calculations above to assign it as
        # boolean True if the cell overlaps with an obstacle and boolean False if it doesn't
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for ox, oy in zip(obstacle_x, obstacle_y):
                    distance = math.hypot(ox - x, oy - y)
                    if distance <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model() -> List[List[float]]:
        """
        Define the possible movement directions and their costs.

        Returns:
            List of [dx, dy, cost] for each possible movement
        """
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]
        return motion


def main():
    print(__file__ + " start!!")

    #default implementation
    # start and goal position
    # start_x = 5.0  # [m]
    # start_y = 5.0  # [m]
    # goal_x = 50.0  # [m]
    # goal_y = 50.0  # [m]
    # cell_size = 2.0  # [m]
    # robot_radius = 1.0  # [m]

    # Feel free to change the obstacle positions and test the implementation on various scenarios
    # obstacle_x, obstacle_y = [], []
    # for i in range(0, 60):
    #     obstacle_x.append(i)
    #     obstacle_y.append(0.0)
    # for i in range(0, 60):
    #     obstacle_x.append(60.0)
    #     obstacle_y.append(i)
    # for i in range(0, 61):
    #     obstacle_x.append(i)
    #     obstacle_y.append(60.0)
    # for i in range(0, 61):
    #     obstacle_x.append(0.0)
    #     obstacle_y.append(i)
    # for i in range(0, 40):
    #     obstacle_x.append(20.0)
    #     obstacle_y.append(i)
    # for i in range(0, 40):
    #     obstacle_x.append(40.0)
    #     obstacle_y.append(60.0 - i)

    # testing both algorithms implementation
    # start and goal position
    start_x = 0.0  # [m]
    start_y = 0.0  # [m]
    goal_x = 40.0  # [m]
    goal_y = 40.0  # [m]
    cell_size = 2.0  # [m]
    robot_radius = 1.0  # [m]
    # Feel free to change the obstacle positions and test the implementation on various scenarios
    obstacle_x, obstacle_y = [], []
    for i in range(0, 50):
        obstacle_x.append(i)
        obstacle_y.append(0.0)
    for i in range(0, 50):
        obstacle_x.append(50.0)
        obstacle_y.append(i)
    for i in range(0, 51):
        obstacle_x.append(i)
        obstacle_y.append(50.0)
    for i in range(0, 51):
        obstacle_x.append(0.0)
        obstacle_y.append(i)
    for i in range(0, 45):
        obstacle_x.append(10.0)
        obstacle_y.append(i)
    for i in range(0, 45):
        obstacle_x.append(20.0)
        obstacle_y.append(50-i)
    for i in range(0, 45):
        obstacle_x.append(35.0)
        obstacle_y.append(i)

    if show_animation:  # pragma: no cover
        plt.plot(obstacle_x, obstacle_y, ".k")
        plt.plot(start_x, start_y, "og")
        plt.plot(goal_x, goal_y, "xb")
        plt.grid(True)
        plt.axis("equal")
    a_star = AStarPlanner(obstacle_x, obstacle_y, cell_size, robot_radius)
    start_time = time.time()
    rx, ry = a_star.planning(start_x, start_y, goal_x, goal_y)
    end_time = time.time()
    duration = end_time-start_time
    print(f"duration: {duration:.3f}")
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
