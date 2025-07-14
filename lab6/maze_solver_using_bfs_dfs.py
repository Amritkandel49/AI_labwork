import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def to_numeric_grid(maze):
    # maze = np.array(maze)
    # print(maze)
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 'S':
                maze[i][j] = 2
            elif maze[i][j] == 'G':
                maze[i][j] = 3
            else:
                maze[i][j] = maze[i][j]
    return maze.astype(int)


def find_pos(maze, value):
    maze = np.array(maze)
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == value:
                return (i, j)



def get_neighbours(r, c, numeric_maze):
    grid = np.array(numeric_maze)
    ROWS = grid.shape[0] - 1
    COLS = grid.shape[1] - 1
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r <= ROWS and 0 <= new_c <= COLS and grid[new_r][new_c] != 1:  # Exclude walls
            neighbors.append((new_r, new_c))
    return neighbors


def bfs(maze, start, goal):
    queue = deque([start])
    visited = set()
    visited.add(start)
    parent = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]
        r, c = current
        for neighbor in get_neighbours(r, c, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    return None

def dfs(maze, start, goal):
    stack = [start]
    visited = set()
    visited.add(start)
    parent = {start: None}

    while stack:
        current = stack.pop()
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for neighbor in get_neighbours(*current, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
    return None


def visualize_path(maze, path, start_pos, goal_pos):
    grid = np.array(maze)
    plt.imshow(grid, cmap='viridis')
    
    for y, x in path:
        plt.scatter(x, y, c='b', s=50, marker='o')

    # start = np.argwhere(grid == 2)
    # goal = np.argwhere(grid == 3)

    for y, x in [start_pos]:
        plt.scatter(x, y, c='g', s=100, marker='*', label='Start')

    for y, x in [goal_pos]:
        plt.scatter(x, y, c='r', s=100, marker='*', label='Goal')

    plt.title("Path Visualization")
    plt.legend()
    plt.show()


def pipeline(maze, search_alg):
    numeric_maze = to_numeric_grid(maze)
    start_pos = find_pos(numeric_maze, 2)
    goal_pos = find_pos(numeric_maze, 3)

    if search_alg == 'bfs':
        path = bfs(numeric_maze, start_pos, goal_pos)
    elif search_alg == 'dfs':
        path = dfs(numeric_maze, start_pos, goal_pos)
    else:
        raise ValueError("Invalid search algorithm. Use 'bfs' or 'dfs'.")
    
    print(f"Path from Start to Goal using {search_alg.upper()}: {path}")
    visualize_path(numeric_maze, path, start_pos, goal_pos)


if __name__ == "__main__":
    maze = np.array([
    [0, 'S', 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 0, 'G']])

    search_alg = input("Enter search algorithm (bfs/dfs): ").strip().lower()
    if search_alg not in ['bfs', 'dfs']:
        print("Invalid input. Please enter 'bfs' or 'dfs'.")
    else:
        pipeline(maze, search_alg)      