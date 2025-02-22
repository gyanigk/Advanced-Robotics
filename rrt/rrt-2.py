"""
Rapidly Exploring Random Tree (RRT) Path Planning Implementation

This module implements the RRT algorithm for path planning, which works by:
1. Incrementally building a tree by sampling random points
2. Connecting new points to the nearest node in the tree
3. Checking for collisions and maintaining feasible paths

COEN 5830 HW2: Path Planning

Instructions:
------------
1. Add your name below
2. Complete all sections marked with "Student Task:" comments/docstrings
3. The main algorithm steps are in the `planning` and `steer` methods
4. Test your implementation with different obstacle configurations

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
import random
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class RRT:
    """RRT path planning implementation."""

    class Node:
        """A node in the RRT tree."""

        def __init__(self, x: float, y: float, yaw: float = 0.0):
            """
            Initialize a node.

            Args:
                x: X coordinate
                y: Y coordinate
                yaw: Heading angle in radians (for non-holonomic planning)
            """
            self.x = x
            self.y = y
            self.yaw = yaw  # Heading angle for non-holonomic planning
            self.path_x: List[float] = []
            self.path_y: List[float] = []
            self.path_yaw: List[float] = []  # Heading angles along path
            self.parent: Optional["RRT.Node"] = None

    class AreaBounds:
        """Bounds for the planning area."""

        def __init__(self, area: List[float]):
            """
            Initialize area bounds.

            Args:
                area: List of [xmin, xmax, ymin, ymax]
            """
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(
        self,
        start: List[float],
        goal: List[float],
        obstacle_list: List[Tuple[float, float, float]],
        rand_area: List[float],
        expand_dis: float = 3.0,
        path_resolution: float = 0.5,
        goal_sample_rate: int = 5,
        max_iter: int = 500,
        play_area: Optional[List[float]] = None,
        robot_radius: float = 0.0,
        max_steer: float = 0.5,
    ):
        """
        Initialize the RRT planner.

        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacle_list: List of obstacles [(x, y, radius), ...]
            rand_area: Random sampling area [min, max]
            expand_dis: Maximum distance to expand tree
            path_resolution: Resolution for path checking
            goal_sample_rate: Rate at which goal is sampled instead of random point
            max_iter: Maximum number of iterations
            play_area: Optional bounds for valid area [xmin, xmax, ymin, ymax]
            robot_radius: Radius of robot for collision checking
            max_steer: Maximum steering angle (radians)

        Class Attributes:
            self.start: Node object representing start position
            self.end: Node object representing goal position
            self.min_rand: Minimum value for random sampling range
            self.max_rand: Maximum value for random sampling range
            self.play_area: AreaBounds object defining valid planning space (None if unbounded)
            self.expand_dis: Maximum distance to extend tree in each iteration
            self.path_resolution: Distance between intermediate points when extending tree
            self.goal_sample_rate: Probability (in %) of sampling goal instead of random point
            self.max_iter: Maximum number of iterations for tree expansion
            self.obstacle_list: List of obstacles, each defined by (x, y, radius)
            self.node_list: List of all nodes in the tree
            self.robot_radius: Robot's physical size for collision checking
            self.max_steer: Maximum steering angle (radians)
        """
        # Create Node objects for start and goal positions
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        # Random sampling bounds
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        # Create AreaBounds object if play_area is specified
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        # Tree expansion parameters
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        # Environment and planning data
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.max_steer = max_steer

    def get_random_node(self) -> "RRT.Node":
        """
        Sample a random node in the state space.

        The sampling should account for:
        - goal_sample_rate: Probability of sampling the goal node
        - min_rand/max_rand: Bounds of the sampling space
        - For non-holonomic planning, include random heading angle

        Student Task:
        ------------
        1. With probability goal_sample_rate/100, return the goal node
        2. Otherwise, sample random position within min_rand/max_rand bounds
        3. For non-holonomic planning, sample random heading in [-pi, pi]
        4. Return new Node with sampled values

        Returns:
            Node: Randomly sampled node
        """
        # YOUR CODE GOES HERE
        # Use goal_sample_rate to bias sampling towards goal
        # Sample random position within bounds
        # For non-holonomic planning, include random heading

        if random.randint(0,100)< self.goal_sample_rate:
            return self.end
        else:
            x = random.uniform(self.min_rand, self.max_rand)
            y = random.uniform(self.min_rand, self.max_rand)
            yaw = random.uniform(-math.pi, math.pi)
            return self.Node(x,y, yaw)
        
        
        # pass
    
    def calc_dist_to_goal(self, x, y):
        """Calculate distance from current position to the goal."""
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def generate_final_course(self, goal_ind, nonholonomic=False):
        """Generate the final path from the goal node to the start node.

        Args:
            goal_ind: Index of the goal node in node_list
            nonholonomic: Whether to generate non-holonomic path segments

        Returns:
            path: List of points along the path
        """
        path = []
        node = self.node_list[goal_ind]

        # For non-holonomic planning, we want to use the stored path points
        if nonholonomic:
            # Start from the goal node and work backwards
            while node is not None:
                # Add all points from this node's path (in reverse)
                for x, y in zip(reversed(node.path_x), reversed(node.path_y)):
                    path.append((x, y))
                node = node.parent
        else:
            # For standard RRT, just use node positions
            while node.parent is not None:
                path.append((node.x, node.y))
                node = node.parent
            path.append((node.x, node.y))

        return path[::-1]  # Reverse path to get start-to-goal order

    def planning(
        self, animation: bool = True, nonholonomic: bool = False
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal using RRT.

        Args:
            animation: Whether to show animation of tree growth
            nonholonomic: Whether to plan with non-holonomic constraints

        Returns:
            path: List of points [(x1,y1), (x2,y2),...] or None if no path found

        Student Task:
        ------------
        1. Initialize the tree with the start node (add to node_list)
        2. For max_iter iterations:
           a. Sample random point using get_random_node()
           b. Find nearest node in tree
           c. Attempt to steer towards sampled point
           d. If valid new node:
              - Add to tree
              - Check if goal is reachable
              - If goal reached, extract and return path
        3. Return None if no path found
        """
        self.node_list = [self.start]

        rnd_node = None
        for i in range(self.max_iter):
            # YOUR CODE GOES HERE
            # Use get_random_node() to sample a random configuration
            # Find nearest node in the tree
            # Steer towards the sampled node (use nonholonomic_steer if nonholonomic=True)
            # Check for collisions and validity
            rnd_node = self.get_random_node()
            nearest_node = self.find_nearest_node(rnd_node)
            if nonholonomic:
                node=self.nonholonomic_steer(nearest_node,rnd_node,self.max_steer,self.expand_dis)
            else:
                node = self.steer(nearest_node,rnd_node,self.expand_dis)
            if node is None:
                continue

            self.node_list.append(node)
            # DO NOT ALTER THE NEXT 2 LINES
            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)
            # YOUR CODE GOES HERE
            # Check if goal is reachable from new node
            if self.calc_dist_to_goal(node.x, node.y)<=self.expand_dis:
                best_node = self.steer(node,self.end, self.expand_dis)
                if best_node is not None and self.check_collision(best_node,self.obstacle_list):
                    self.node_list.append(best_node)
                    return self.generate_final_course(len(self.node_list)-1,nonholonomic)
        return None

    def steer(
        self,
        from_node: "RRT.Node",
        to_node: "RRT.Node",
        extend_length: float = float("inf"),
    ) -> "RRT.Node":
        """
        Steer from one node towards another within constraints.

        Args:
            from_node: Node to steer from
            to_node: Node to steer towards
            extend_length: Maximum distance to extend

        Returns:
            new_node: New node after steering

        Student Task:
        ------------
        1. Create new node at from_node location
        2. Calculate distance and angle to to_node
        3. If distance > extend_length:
           - Scale movement to respect extend_length
        4. Move towards to_node:
           - Update position incrementally using path_resolution
           - Store path points in new_node.path_x and new_node.path_y
        5. Set parent relationship
        6. Return new node
        """
        new_node = self.Node(from_node.x, from_node.y)
        distance, theta = self.calc_distance_and_angle(
            new_node, to_node
        )  # This function returns the distance d from new_node to to_node and the theta represents the angle the line joining them makes with the x axis

        new_node.path_x = [from_node.x]
        new_node.path_y = [from_node.y]
        # YOUR CODE GOES HERE
        # If extend_length is greater than the distance, then ensure the robot doesn't go beyong extend_length
        # Propagate the robot iteratively from new_node to to_node until extend_length is reached
        # Use check_collision to check if the robot is colliding with an obstacle
        # If the robot is colliding with an obstacle, then stop the propagation
        # Append the coordinates of the robot into new_node.path_x and new_node.path_y
        total_dist = 0.0
        while total_dist < min(extend_length,distance):
            next_x = new_node.path_x[-1] + self.path_resolution * math.cos(theta)
            next_y = new_node.path_y[-1] + self.path_resolution * math.sin(theta)
            temp_node = self.Node(next_x, next_y)
            # Check for collision along the path
            if not self.check_collision(temp_node, self.obstacle_list):
                return None
            new_node.path_x.append(next_x)
            new_node.path_y.append(next_y)
            total_dist += self.path_resolution

        
        d, _ = self.calc_distance_and_angle(
            new_node, to_node
        )  # We want to check if the robot has reached to_node
        # YOUR CODE GOES HERE
        # Check if the robot has reached to_node. If yes, add to_node coordinates to the path of new_node
        # Add from_node as parent of new_node
        # if self.calc_dist_to_goal(new_node.path_x[-1], new_node.path_y[-1]) <= self.path_resolution:
        #     new_node.path_x.append(to_node.x)
        #     new_node.path_y.append(to_node.y)
        if self.calc_distance_and_angle(new_node, to_node)[0] <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
        
        new_node.x = new_node.path_x[-1]
        new_node.y = new_node.path_y[-1]
        new_node.parent = from_node

        return new_node

    @staticmethod
    def calc_distance_and_angle(
        from_node: "RRT.Node", to_node: "RRT.Node"
    ) -> Tuple[float, float]:
        """
        Calculate distance and angle between nodes.

        Args:
            from_node: Starting node
            to_node: Target node

        Returns:
            distance: Euclidean distance between nodes
            theta: Angle in radians from from_node to to_node

        Student Task:
        ------------
        1. Calculate Euclidean distance between nodes
        2. Calculate angle of line from from_node to to_node
           relative to x-axis (use math.atan2)
        3. Return distance and angle
        """
        distance = None
        theta = None
        # YOUR CODE GOES HERE
        # Write code to find the distance between from_node and to_node
        # and the angle made by the line joining from_node and to_node with the x_axis
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance = math.hypot(dx,dy)
        theta = math.atan2(dy,dx)
        return distance, theta

    def find_nearest_node(self, node: "RRT.Node") -> "RRT.Node":
        """
        Find the nearest node in the tree to the given node.

        Args:
            node: Node to find nearest node to

        Returns:
            nearest_node: Nearest node in the tree

        Student Task:
        ------------
        1. Initialize variables for tracking nearest node and minimum distance
        2. Iterate through node_list
        3. Calculate distance to each node
        4. Update nearest node if current distance is smaller
        5. Return the nearest node found
        """
        nearest_node = None
        min_distance = float("inf")
        # YOUR CODE HERE
        for n in self.node_list:
            dx = n.x - node.x
            dy = n.y - node.y
            sq_dist = dx*dx +dy*dy
            # d = math.hypot(n.x-node.x, n.y-node.y)
            if sq_dist<min_distance:
                min_distance =sq_dist
                nearest_node = n
        return nearest_node

    def check_collision(
        self, node: "RRT.Node", obstacle_list: List[Tuple[float, float, float]]
    ) -> bool:
        """
        Check if the node collides with any obstacles.

        Args:
            node: Node to check for collision
            obstacle_list: List of obstacles [(x, y, radius), ...]

        Returns:
            collision_free: True if node is collision-free, False otherwise

        Student Task:
        ------------
        1. For each obstacle:
           - Calculate distance from node to obstacle center
           - Check if distance is less than obstacle radius + robot_radius
        2. Return True if no collisions, False otherwise
        """
        # YOUR CODE GOES HERE
        for (ox,oy,radius) in obstacle_list:
            d = math.hypot(node.x-ox,node.y-oy)
            if d<=(radius+self.robot_radius):
                return False
        return True

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, "-r")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for ox, oy, size in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot(
                [
                    self.play_area.xmin,
                    self.play_area.xmax,
                    self.play_area.xmax,
                    self.play_area.xmin,
                    self.play_area.xmin,
                ],
                [
                    self.play_area.ymin,
                    self.play_area.ymin,
                    self.play_area.ymax,
                    self.play_area.ymax,
                    self.play_area.ymin,
                ],
                "-k",
            )

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    def nonholonomic_steer(
        self,
        from_node: "RRT.Node",
        to_node: "RRT.Node",
        max_steer: float = 0.5,
        extend_length: float = float("inf"),
    ) -> "RRT.Node":
        """
        Steer from one node towards another respecting non-holonomic constraints.

        Args:
            from_node: Node to steer from
            to_node: Node to steer towards
            max_steer: Maximum steering angle (radians)
            extend_length: Maximum distance to extend

        Returns:
            new_node: New node after steering

        Student Task:
        ------------
        1. Create new node at from_node location with heading
        2. Calculate target angle to to_node
        3. Determine steering angle (limited by max_steer)
        4. Generate curved path points using interpolate_path:
           - Account for forward-only motion
           - Respect steering constraints
           - Stay within extend_length
        5. Check path for collisions
        6. Update node path and parent relationship
        7. Return new node
        """
        new_node = self.Node(from_node.x, from_node.y, from_node.yaw)

        # YOUR CODE GOES HERE
        # Calculate target angle and steering angle
        # Generate path points using interpolate_path
        # Check for collisions along path
        # Update node attributes and parent
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        target_angle = math.atan2(dy,dx)
        steering_angle = target_angle - from_node.yaw
        steering_angle = max(min(steering_angle, max_steer), -max_steer)
        path_x, path_y, path_yaw = self.interpolate_path(
            from_node.x,
            from_node.y,
            from_node.yaw,
            to_node.x,
            to_node.y,
            steering_angle,
            extend_length,
        )
        # Check for collision along the path
        for (x, y) in zip(path_x, path_y):
            temp_node = self.Node(x, y)
            if not self.check_collision(temp_node, self.obstacle_list):
                return None
        
        new_node.path_x = path_x
        new_node.path_y = path_y
        new_node.path_yaw = path_yaw
        new_node.x = path_x[-1]
        new_node.y = path_y[-1]
        new_node.yaw = path_yaw[-1]
        new_node.parent = from_node

        return new_node

    def interpolate_path(
        self,
        x: float,
        y: float,
        yaw: float,
        target_x: float,
        target_y: float,
        steering_angle: float,
        extend_length: float,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Generate points along a curved path for non-holonomic motion.
        This function is provided for you - you do not need to implement it.

        The function uses a simple bicycle model to generate points along a curved path:
        - The robot can only move forward
        - The robot has a minimum turning radius based on maximum steering angle
        - Points are generated at regular intervals defined by path_resolution

        Args:
            x: Start x position
            y: Start y position
            yaw: Start heading angle
            target_x: Target x position
            target_y: Target y position
            steering_angle: Applied steering angle (limited by max_steer)
            extend_length: Maximum path length

        Returns:
            path_x, path_y, path_yaw: Lists of points and angles along path
        """
        path_x = [x]
        path_y = [y]
        path_yaw = [yaw]

        # Clamp steering angle to max_steer
        steering_angle = max(min(steering_angle, self.max_steer), -self.max_steer)

        # Calculate vehicle parameters
        wheel_base = 1.0  # [m] - distance between front and rear axles

        # Initialize current position
        current_x = x
        current_y = y
        current_yaw = yaw

        # Calculate distance to target
        distance = math.hypot(target_x - current_x, target_y - current_y)

        # Generate points along path
        traveled = 0.0
        while traveled < min(distance, extend_length):
            # Update position using bicycle model
            current_x += self.path_resolution * math.cos(current_yaw)
            current_y += self.path_resolution * math.sin(current_yaw)
            current_yaw += self.path_resolution * math.tan(steering_angle) / wheel_base

            # Normalize yaw angle
            current_yaw = self.normalize_angle(current_yaw)

            # Store points
            path_x.append(current_x)
            path_y.append(current_y)
            path_yaw.append(current_yaw)

            # Update traveled distance
            traveled += self.path_resolution

        return path_x, path_y, path_yaw

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize an angle to [-pi, pi].
        This is a helper function for interpolate_path.

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle in radians
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def plot_curved_path(
        self,
        path_x: List[float],
        path_y: List[float],
        path_yaw: List[float],
        color: str = "-g",
    ):
        """
        Plot a curved path with heading indicators.

        Args:
            path_x: List of x coordinates
            path_y: List of y coordinates
            path_yaw: List of heading angles
            color: Path color
        """
        plt.plot(path_x, path_y, color)
        # Plot heading indicators at intervals
        for i in range(0, len(path_x), 5):
            plt.arrow(
                path_x[i],
                path_y[i],
                0.5 * math.cos(path_yaw[i]),
                0.5 * math.sin(path_yaw[i]),
                head_width=0.1,
                head_length=0.2,
                fc=color[1],
                ec=color[1],
            )


def main(goal_x=6.0, goal_y=10.0):
    print("start " + __file__)
    obstacleList = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
    ]  # [x, y, radius]

    # Initialize plot
    plt.ion()  # Enable interactive mode
    plt.figure(figsize=(10, 10))

    rrt = RRT(
        start=[0, 0],
        goal=[goal_x, goal_y],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        robot_radius=0.8,
    )

    # Run planning
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        print(path)
        rx, ry = zip(*path)
        # Draw the final path in red
        plt.plot(rx, ry, '-r', linewidth=2)

    plt.ioff()  # Disable interactive mode
    plt.show()  # This will block until the window is closed


def main_nonholonomic(goal_x: float = 6.0, goal_y: float = 10.0):
    """
    Test the non-holonomic RRT implementation.

    Args:
        goal_x: Goal x coordinate
        goal_y: Goal y coordinate
    """
    print("Testing Non-holonomic RRT")

    # Test scenario with obstacles
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
    ]

    # Initialize RRT with non-holonomic parameters
    rrt = RRT(
        start=[0, 0],
        goal=[goal_x, goal_y],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list,
        robot_radius=0.8,
        max_steer=0.8,  # Increased steering angle for better maneuverability
        path_resolution=0.1,  # Finer resolution for smoother paths
        expand_dis=3.0,  # Increased expansion distance
        max_iter=1000,  # More iterations since non-holonomic planning is harder
        goal_sample_rate=20,  # Increased goal sampling to guide search better
    )

    # Plan path with non-holonomic constraints
    path = rrt.planning(animation=show_animation, nonholonomic=True)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!")
        print(path)
        rx, ry = zip(*path)
        # Draw the final path in red
        plt.plot(rx, ry, '-r', linewidth=2)
        # print("Number of iterations")
        if show_animation:
            plt.plot(path[0][0], path[0][1], "xr", label="Start")
            plt.plot(path[-1][0], path[-1][1], "xb", label="Goal")
            plt.grid(True)
            plt.pause(0.01)
            plt.show()


if __name__ == "__main__":
    main()  # Run standard RRT
    main_nonholonomic()  # Run non-holonomic RRT. Comment this out to run standard RRT
