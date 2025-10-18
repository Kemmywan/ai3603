import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3-map/map.npy')


### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

def Distance_to_Wall(world_map, position):
    x, y = position
    d = 1000.0
    dist = 1
    while True:
        for dx in [-dist, 0, dist]:
            for dy in [-dist, 0, dist]:
                if dx * dx + dy * dy <= dist * dist:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 120 and 0 <= ny < 120:
                        if world_map[nx][ny] == 1:
                            d = min(d, (dx * dx + dy * dy) ** 0.5)
        if d != 1000:
            return d
        dist += 1

def heuristic(world_map, pos, goal_pos, if_turn=0):
    return abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1]) - 3 * Distance_to_Wall(world_map, pos) + 10 * if_turn

###  END CODE HERE  ###


def Improved_A_star(world_map, start_pos, goal_pos):
    """
    Given map of the world, start position of the robot and the position of the goal, 
    plan a path from start position to the goal using A* algorithm.

    Arguments:
    world_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    start_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###

    fringe = PriorityQueue()

    fringe.push([start_pos,[start_pos], 0], 0 + heuristic(world_map, start_pos, goal_pos))

    best_g = dict()

    best_g[tuple(start_pos)] = 0

    while not fringe.isEmpty():
        state, path, cost = fringe.pop()
        if state == goal_pos:
            return path
        dx_prev = 2
        dy_prev = 2
        if len(path) >= 2:
            dx_prev = state[0] - path[-2][0]
            dy_prev = state[1] - path[-2][1]
        # Get successors
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            successor = [state[0]+dx, state[1]+dy]
            if (dx == dx_prev and dy == dy_prev):
                if_turn = 0
            else:
                if_turn = 1
            if 0 <= successor[0] < 120 and 0 <= successor[1] < 120 and world_map[successor[0]][successor[1]] == 0:
                stepCost = 1.4 if abs(dx) + abs(dy) == 2 else 1
                new_cost = cost + stepCost
                if tuple(successor) not in best_g or new_cost < best_g[tuple(successor)]:
                    best_g[tuple(successor)] = new_cost
                    f_cost = new_cost + heuristic(world_map, successor, goal_pos, if_turn)
                    fringe.push([successor, path + [successor], new_cost], f_cost)
    
    raise Exception("No path found")


    ###  END CODE HERE  ###





if __name__ == '__main__':

    # Get the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    map = np.load(MAP_PATH)

    # Define goal position of the exploration
    goal_pos = [100, 100]

    # Define start position of the robot.
    start_pos = [10, 10]

    # Plan a path based on map from start position of the robot to the goal.
    path = Improved_A_star(map, start_pos, goal_pos)

    # Visualize the map and path.
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(start_pos[0], start_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
