import numpy as np
import random
from collections import defaultdict
from itertools import chain



seed = 123
random.seed(seed)
np.random.seed(seed)


def txt_generator(agent_count, item_count, grid_width, grid_height):
    """
    Agent: meta description, x-coordinate, y-coordinate, level, radius, angle
    Item: meta description, x-coordinate, y-coordinate, level
    :param agent_count: integer - number of agents
    :param item_count: integer - number of items
    :param grid_width: Grid width in number of cells
    :param grid_height: Grid height in number of cells
    :return: Print statements
    """
    # Print Grid line
    print('grid, {}, {}'.format(grid_width, grid_height))
    angles = [0, np.round(np.pi/2, 3), np.round(np.pi, 3), np.round(np.pi*2, 3)]
    i = 0
    while i < agent_count:
        if i == 0:
            prefix = 'main'
        else:
            prefix = 'agent'
        print('{}, {}, {}, {}, {}, {}, {}'.format(prefix,
                                                  i+1,  # meta-name
                                                  np.round(np.random.uniform(0, grid_width)).astype(int),  # x-coord
                                                  np.round(np.random.uniform(0, grid_height)).astype(int),  # y-coord
                                                  np.round(np.random.uniform(), 3),  # level
                                                  np.round(np.random.uniform(0.1, 2*np.pi-0.01), 1).astype(int),  # radius
                                                  random.choice(angles)))  # direction
        i += 1
    i = 0
    while i < item_count:
        print('item{}, {}, {}, {}'.format(i+1,
                                          np.round(np.random.uniform(0, grid_width)).astype(int),
                                          np.round(np.random.uniform(0, grid_height)).astype(int),
                                          np.round(np.random.uniform(), 3).astype(int)))
        i += 1

def loader(path):
    info = defaultdict(list)
    with open(path) as info_read:
        for line in info_read:
            data = line.strip().split(', ')
            key, val = data[0], data[1:]
            info[key].append(val)
    return info

if __name__ == '__main__':
    txt_generator(2, 10, 10, 10)
    t = loader('/home/tpin3694/Documents/python/MultiAgents/simulation.csv')