import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """

        # TODO: Implement adding a neighbor in an undirected manner
        self.neighbors.append(node)
        node.neighbors.append(self)

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    # TODO: Implement the logic to build nodes and link neighbors

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 0:
                temp = Node((i,j))
                nodes_dict[(i,j)] = temp
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 0:
                if i+1 <= rows-1 and maze[i+1][j] == 0:
                    Node.add_neighbor(nodes_dict[(i,j)], nodes_dict[(i+1,j)])
                if i-1 >= 0 and maze[i-1][j] == 0:
                    Node.add_neighbor(nodes_dict[(i,j)], nodes_dict[(i-1,j)])
                if j+1 <= cols-1 and maze[i][j+1] == 0:
                    Node.add_neighbor(nodes_dict[(i,j)], nodes_dict[(i,j+1)])
                if j-1 >= 0 and maze[i][j-1] == 0:
                    Node.add_neighbor(nodes_dict[(i,j)], nodes_dict[(i,j-1)])
    start_node = None
    goal_node = None
    if maze[0][0] == 0:
        start_node = nodes_dict[(0,0)]
    if maze[rows-1][cols-1] == 0:
        goal_node = nodes_dict[(rows-1,cols-1)]
    z = 0
    # for x in nodes_dict.values():
    #         for y in x.neighbors:
    #             z+=1
    # print(z)
    # print(len(nodes_dict))

    # TODO: Assign start_node and goal_node if they exist in nodes_dict

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node: Node, goal_node: Node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    visited = set()
    parent_map = {}
    path = []
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        visited.add(node)
        if node == goal_node:
            break
        for neighbor in sorted(node.neighbors, key=lambda x: x.value):
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
                parent_map[neighbor] = node
    if node != goal_node:
        return None
    temp = parent_map[goal_node]
    path = deque([goal_node.value])
    path.appendleft(temp.value)
    while temp:
        path.appendleft(parent_map[temp].value)
        temp = parent_map[temp]
        if temp == start_node:
            break
    
    # TODO: Implement BFS
    path = list(path)
    return path


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    visited = set()
    parent_map = {}
    path = []
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        visited.add(node)
        if node == goal_node:
            break
        for neighbor in node.neighbors:
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
                parent_map[neighbor] = node
    if node != goal_node:
        return None
    temp = parent_map[goal_node]
    path = deque([goal_node.value])
    path.appendleft(temp.value)
    while temp:
        path.appendleft(parent_map[temp].value)
        temp = parent_map[temp]
        if temp == start_node:
            break
    
    path = list(path)
    return path


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*
    queue = [(0, start_node)]
    f_score = {}
    g_score = {}
    g_score[start_node] = 0
    parent_map = {}
    while queue:
        node = queue.pop(0)[1]
        f_score[node] = g_score[node] + manhattan_distance(node, goal_node)
        if node == goal_node:
            break
        for neighbor in node.neighbors:
            maybe_g_score = g_score[node] + 1
            if neighbor not in g_score or maybe_g_score < g_score[neighbor]:
                g_score[neighbor] = maybe_g_score
                f_cost = maybe_g_score + manhattan_distance(neighbor, goal_node)
                heapq.heappush(queue, (f_cost, neighbor))
                parent_map[neighbor] = node
    if node != goal_node:
        return None
    temp = parent_map[goal_node]
    path = deque([goal_node.value])
    path.appendleft(temp.value)
    while temp:
        path.appendleft(parent_map[temp].value)
        temp = parent_map[temp]
        if temp == start_node:
            break
    
    path = list(path)
    return path

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    r1,c1 = node_a.value
    r2,c2 = node_b.value
    # TODO: Return |r1 - r2| + |c1 - c2|
    return abs(r1-r2) + abs(c1-c2)


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    # TODO: Implement bidirectional search
    visited1 = set()
    visited2 = set()
    parent_map1 = {}
    parent_map2 = {}
    path1 = []
    path2 = []
    queue1 = deque([start_node])
    queue2 = deque([goal_node])
    midNode = None
    while queue1:
        node1 = queue1.popleft()
        visited1.add(node1)
        if node1 in visited2:
            midNode = node1
            break
        for neighbor in node1.neighbors:
            if neighbor not in visited1 and neighbor not in queue1:
                queue1.append(neighbor)
                parent_map1[neighbor] = node1
        if queue2:
            node2 = queue2.popleft()
            visited2.add(node2)
            if node2 in visited1:
                midNode = node2
                break
            for neighbor in node2.neighbors:
                if neighbor not in visited2 and neighbor not in queue2:
                    queue2.append(neighbor)
                    parent_map2[neighbor] = node2

    if midNode == None:
        return None
    temp = parent_map1[midNode]
    path1 = deque([midNode.value])
    path1.appendleft(temp.value)
    while temp:
        path1.appendleft(parent_map1[temp].value)
        temp = parent_map1[temp]
        if temp == start_node:
            break
    temp = parent_map2[midNode]
    path2 = deque([midNode.value])
    path2.appendleft(temp.value)
    x = 0
    while temp:
        path2.appendleft(parent_map2[temp].value)
        temp = parent_map2[temp]
        if temp == goal_node:
            break

    path1 = list(path1)
    path2 = list(path2)
    full_path = path1 + path2
    return full_path


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing
    current = start_node
    curr_cost = 0
    parent_map = {}
    visited = set()
    counter = 0
    while goal_node not in parent_map and counter <= 625:
        curr_cost = manhattan_distance(current, goal_node)
        if not current.neighbors:
            return None
        next = random.choice(current.neighbors)
        next_cost = manhattan_distance(next, goal_node)
        visited.add(current)
        if next_cost < curr_cost:
            if next not in visited:
                parent_map[next] = current
            current = next
        elif random.random() < math.e**(-1*((curr_cost - next_cost)/temperature)):
            if next not in visited:
                parent_map[next] = current
            current = next
            if temperature >= min_temperature:
                temperature *= cooling_rate
        counter += 1
    # print(parent_map)
    # print(goal_node)
    if counter > 625:
        return None
    temp = parent_map[goal_node]
    path = deque([goal_node.value])
    path.appendleft(temp.value)
    while temp:
        path.appendleft(parent_map[temp].value)
        temp = parent_map[temp]
        if temp == start_node:
            break
    
    path = list(path)
    return path

###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction
    return None


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
