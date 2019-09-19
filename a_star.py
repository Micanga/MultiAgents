from heapq import heappush, heappop # for priority queue
import math


class node:

    xPos = 0
    yPos = 0

    distance = 0
    priority = 0 # smaller: higher priority

    def __init__(self, xPos, yPos, distance, priority):
        self.xPos = xPos
        self.yPos = yPos
        self.distance = distance
        self.priority = priority

    def __lt__(self, other): # for priority queue
        return self.priority < other.priority

    def updatePriority(self, xDest, yDest):
        self.priority = self.distance + self.estimate(xDest, yDest) * 10 # A*

    def next_distance(self, i): # i: direction
        if i % 2 == 0:
            self.distance += 10
        else:
            self.distance += 14
    
    # Estimation function for the remaining distance to the goal.
    def estimate(self, xDest, yDest):
        xd = xDest - self.xPos
        yd = yDest - self.yPos
        # Euclidian Distance
        d = math.sqrt(xd * xd + yd * yd)
        return(d)

    # A-star algorithm.
    # Path returned will be a string of digits of directions.

class a_star:

    def __init__(self, simulator ,a_agent):

        self.n = simulator.dim_w
        self.m = simulator.dim_h
        self.the_map = simulator.the_map
        self.directions = 4
        self.dx = [-1, 0, 1,  0]
        self.dy = [0,  1, 0, -1]
        self.obstacles = self.create_abstacles_list(simulator,a_agent)

    @staticmethod
    def create_abstacles_list(sim, a_agent):
        obstacles = list()

        for item in sim.items:
            if a_agent.get_memory() != item.get_position():
                obstacles.append(item.get_position())

        for agent in sim.agents:
            if a_agent.get_position() != agent.get_position():
                obstacles.append(agent.get_position())

        for obstacle in sim.obstacles:
            obstacles.append(obstacle.get_position())

        if (sim.main_agent is not None):
            obstacles.append(sim.main_agent.get_position())

        return obstacles

    def position_is_obstacle(self,x,y):
        for obstacle in self.obstacles:
            if obstacle[0] == x and obstacle[1] == y:
                return True
        return False

    def pathFind(self, xStart, yStart, xFinish, yFinish):
        closed_nodes_map = []
        open_nodes_map = []
        dir_map = []
        row = [0] * self.n
        for i in range(self.m):# create 2d arrays
            closed_nodes_map.append(list(row))
            open_nodes_map.append(list(row))
            dir_map.append(list(row))

        pq = [[], []] # priority queues of open (not-yet-tried) nodes
        pqi = 0 # priority queue index

        n0 = node(xStart, yStart, 0, 0)
        n0.updatePriority(xFinish, yFinish)
        heappush(pq[pqi], n0)
        open_nodes_map[xStart][yStart] = n0.priority # mark it on the open nodes map

        # A* search
        while len(pq[pqi]) > 0:
            # get the current node w/ the highest priority
            # from the list of open nodes
            n1 = pq[pqi][0] # top node

            n0 = node(n1.xPos, n1.yPos, n1.distance, n1.priority)
            x = n0.xPos
            y = n0.yPos
            heappop(pq[pqi]) # remove the node from the open list
            open_nodes_map[x][y] = 0
            # mark it on the closed nodes map
            closed_nodes_map[x][y] = 1

            # quit searching when the goal state is reached
            # if n0.estimate(xFinish, yFinish) == 0:
            if x == xFinish and y == yFinish:
                # generate the path from finish to start
                # by following the directions
                path = ''
                while not (x == xStart and y == yStart):
                    j = dir_map[x][y]

                    c = str((j + self.directions / 2) % self.directions)
                    path = c + path
                    x += self.dx[j]
                    y += self.dy[j]

                return path

            for i in range(self.directions):
                xdx = x + self.dx[i]
                ydy = y + self.dy[i]

                if not (xdx < 0 or xdx > self.n-1 or ydy < 0 or ydy > self.m - 1):
                    if not(self.position_is_obstacle(xdx, ydy) or closed_nodes_map[xdx][ydy] == 1):

                        m0 = node(xdx, ydy, n0.distance, n0.priority)
                        m0.next_distance(i)
                        m0.updatePriority(xFinish, yFinish)

                        if open_nodes_map[xdx][ydy] == 0:
                            open_nodes_map[xdx][ydy] = m0.priority
                            heappush(pq[pqi], m0)

                            dir_map[xdx][ydy] = (i + self.directions / 2) % self.directions
                        elif open_nodes_map[xdx][ydy] > m0.priority:

                            open_nodes_map[xdx][ydy] = m0.priority

                            dir_map[xdx][ydy] = (i + self.directions / 2) % self.directions

                            while not (pq[pqi][0].xPos == xdx and pq[pqi][0].yPos == ydy):
                                heappush(pq[1 - pqi], pq[pqi][0])
                                heappop(pq[pqi])
                            heappop(pq[pqi])

                            if len(pq[pqi]) > len(pq[1 - pqi]):
                                pqi = 1 - pqi
                            while len(pq[pqi]) > 0:
                                heappush(pq[1-pqi], pq[pqi][0])
                                heappop(pq[pqi])
                            pqi = 1 - pqi
                            heappush(pq[pqi], m0) # add the better node instead
        return '' # no route found
