from numpy.random import choice
import position
import numpy as np
from math import sqrt
import parameter_estimation
from copy import copy

class Agent:
    def __init__(self, x, y, direction, agent_type='l1', index='0'):
        self.position = (int(x), int(y))
        self.index = index
        self.level = None
        self.radius = None
        self.angle = None

        self.co_radius = None
        self.co_angle = None

        self.last_loaded_item_pos = None
        if isinstance(direction, basestring):
            self.direction = self.convert_direction(direction)
        else:
            self.direction = float(direction)

        self.agent_type = agent_type

        self.memory = position.position(-1, -1)
        self.route_actions = None

        self.item_to_load = -1

        self.actions_probability = {'L': 0.20, 'N': 0.20, 'E': 0.20, 'S': 0.20, 'W': 0.20}
        self.visible_agents = []
        self.visible_items = []
        self.next_action = None
        self.state_dim = []


    ####################################################################################################################
    def set_parameters(self, sim, level, radius, angle):

        self.state_dim.append(sim.dim_w)
        self.state_dim.append(sim.dim_h)

        self.level = float(level)
        self.radius = float(radius)
        self.angle = float(angle)

        self.co_radius = sqrt(sim.dim_w ** 2 + sim.dim_h ** 2)
        self.co_angle = 2 * np.pi

    ####################################################################################################################
    def set_direction(self, direction):
        self.direction = direction

    ####################################################################################################################
    def reset_memory(self):
        self.memory = position.position(-1, -1)

    ####################################################################################################################
    def get_position(self):
        return self.position[0], self.position[1]

    ####################################################################################################################
    def equals(self, other_agent):
        (x, y) = self.position
         
        (other_x, other_y) = other_agent.get_position()

        (other_memory_x, other_memory_y) = other_agent.get_memory()
        
        (memory_x, memory_y) = self.get_memory()

        return x == other_x and y == other_y and \
               memory_x == other_memory_x and memory_y == other_memory_y and \
               self.agent_type == other_agent.agent_type and \
               self.index == other_agent.index and \
               self.direction == other_agent.direction

    ####################################################################################################################
    def copy(self):

        (x, y) = self.position

        copy_agent = Agent(x, y, self.direction, self.agent_type, self.index)

        (memory_x, memory_y) = self.memory.get_position()

        copy_agent.memory = position.position(memory_x, memory_y)

        copy_agent.direction = self.direction
        copy_agent.level = self.level
        copy_agent.radius = self.radius
        copy_agent.angle = self.angle

        copy_agent.co_radius = self.co_radius
        copy_agent.co_angle = self.co_angle
        copy_agent.next_action = self.next_action

        return copy_agent

    ####################################################################################################################
    def is_agent_face_to_item(self, sim):

        dx = [-1, 0, 1,  0]  # 0:W ,  1:N , 2:E  3:S
        dy = [ 0, 1, 0, -1]

        x_diff = 0
        y_diff = 0

        x, y = self.get_position()

        if self.direction == 2 * np.pi / 2:
            # Agent face to West
            x_diff = dx[0]
            y_diff = dy[0]

        if self.direction == np.pi / 2:
            # Agent face to North
            x_diff = dx[1]
            y_diff = dy[1]

        if self.direction == 0 * np.pi / 2:
            # Agent face to East
            x_diff = dx[2]
            y_diff = dy[2]

        if self.direction == 3 * np.pi / 2:
            # Agent face to South
            x_diff = dx[3]
            y_diff = dy[3]

        if 0 <= x + x_diff < sim.dim_w and 0 <= y + y_diff < sim.dim_h and \
                sim.is_there_item_in_position(x + x_diff, y + y_diff) != -1:

            return True, (x + x_diff, y + y_diff)

        return False,(-1,-1)

    ################################################################################################################

    def get_memory(self):
        (memory_x, memory_y) = self.memory.get_position()
         
        return memory_x, memory_y
    
    ################################################################################################################

    def is_item_nearby(self, items):

        pos = self.position

        for i in range(0, len(items)):
            if not items[i].loaded:
                item = items[i]
                (xI, yI) = item.position.get_position()
                if (yI == pos[1] and abs(pos[0] - xI) == 1) or (xI == pos[0] and abs(pos[1] - yI) == 1):
                    return i
        return -1

    ################################################################################################################
    def find_nearest_item(self, items):

        minimum_distance = 10000
        nearest_item_index = -1

        for i in range(0, len(items)):
            if not items[i].loaded:
                item = items[i]
                (xI, yI) = item.position.get_position()
                if self.distance(item) < minimum_distance:
                    minimum_distance = self.distance(item)
                    nearest_item_index = i

        return nearest_item_index

    ################################################################################################################
    # The agent is "near" if it is next to the destination, and the heading is correct
    def is_agent_near_destination(self, item_x, item_y):
        dx = [-1, 0, 1,  0]  # 0:W ,  1:N , 2:E  3:S
        dy = [ 0, 1, 0, -1]

        x_diff = 0
        y_diff = 0

        pos = self.position

        if self.direction == 2 * np.pi / 2:
            # Agent face to West
            x_diff = dx[0]
            y_diff = dy[0]

        if self.direction == np.pi / 2:
            # Agent face to North
            x_diff = dx[1]
            y_diff = dy[1]

        if self.direction == 0 * np.pi / 2:
            # Agent face to East
            x_diff = dx[2]
            y_diff = dy[2]

        if self.direction == 3 * np.pi / 2:
            # Agent face to South
            x_diff = dx[3]
            y_diff = dy[3]
        
        (xI, yI) = (item_x, item_y)
        if (yI == pos[1] and abs(pos[0] - xI) == 1) or (xI == pos[0] and abs(pos[1] - yI) == 1):
            if (pos[0] + x_diff == xI) and (pos[1] + y_diff == yI):
                return True
            else:
                return False
        else:
            return False

    ################################################################################################################
    def set_probability_main_action(self):
        if self.next_action == 'L':
            self.actions_probability['L'] = 0.96
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.01
            return

        if self.next_action == 'N':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.96
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.01
            return

        if self.next_action == 'W':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.96
            return

        if self.next_action == 'S':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.96
            self.actions_probability['W'] = 0.01
            return

        if self.next_action == 'E':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.96
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.01
            return

    ################################################################################################################
    def set_actions_probabilities(self,action):

        if action == 'L':
            self.actions_probability['L'] = 0.96
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.01
            return

        if action == 'N':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.96
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.01
            return

        if action == 'W':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.96
            return

        if action == 'S':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.01
            self.actions_probability['S'] = 0.96
            self.actions_probability['W'] = 0.01
            return

        if action == 'E':
            self.actions_probability['L'] = 0.01
            self.actions_probability['N'] = 0.01
            self.actions_probability['E'] = 0.96
            self.actions_probability['S'] = 0.01
            self.actions_probability['W'] = 0.01
            return

    ################################################################################################################
    def get_action_probability(self, action):

        if action == 'W':
            return self.actions_probability['W']

        if action == 'L':
            return self.actions_probability['L']

        if action == 'N':
            return self.actions_probability['N']

        if action == 'E':
            return self.actions_probability['E']

        if action == 'S':
            return self.actions_probability['S']

    ################################################################################################################
    def get_actions_probabilities(self):

        actions_probabilities = list()
        actions_probabilities.append(self.actions_probability['L'])
        actions_probabilities.append(self.actions_probability['N'])
        actions_probabilities.append(self.actions_probability['E'])
        actions_probabilities.append(self.actions_probability['S'])
        actions_probabilities.append(self.actions_probability['W'])
        return actions_probabilities

    ################################################################################################################
    def change_direction(self, dx, dy):

        if dx == 1 and dy == 0:  # 'E':
            self.direction = 0 * np.pi / 2

        if dx == 0 and dy == -1:  # 'N':
            self.direction = np.pi / 2

        if dx == -1 and dy == 0:  # 'W':
            self.direction = 2 * np.pi / 2

        if dx == 0 and dy == -1:  # 'S':
            self.direction = 3 * np.pi / 2

    ################################################################################################################
    def change_direction_with_action(self, action):

        if action == 'W':  # 'W':
            self.direction = 2 * np.pi / 2

        if action == 'N':  # 'N':
            self.direction = np.pi / 2

        if action == 'E':  # 'E':
            self.direction = 0 * np.pi / 2

        if action == 'S':  # 'S':
            self.direction = 3 * np.pi / 2

    ################################################################################################################
    @staticmethod
    def convert_direction(direction):

        if direction == 'N':
            return np.pi / 2

        if direction == 'W':
            return np.pi

        if direction == 'E':
            return 0

        if direction == 'S':
            return 3*np.pi/2

    ################################################################################################################
    def get_agent_direction(self):

        if self.direction == np.pi / 2:
            return 'N'

        if self.direction == np.pi:
            return 'W'

        if self.direction == 0:
            return 'E'

        if self.direction == 3 * np.pi / 2:
            return 'S'

    ################################################################################################################
    def change_position_direction(self, dim_w, dim_h):
        dx = [-1, 0, 1,  0]  # 0:W ,  6AGA_O_2:N , 2:E  3:S
        dy = [0, 1, 0, -1]

        x_diff = 0
        y_diff = 0

        if self.next_action == 'W':
            x_diff = dx[0]
            y_diff = dy[0]
            self.direction = 2 * np.pi / 2

        if self.next_action == 'N':
            x_diff = dx[1]
            y_diff = dy[1]
            self.direction = np.pi / 2

        if self.next_action == 'E':
            x_diff = dx[2]
            y_diff = dy[2]
            self.direction = 0 * np.pi / 2

        if self.next_action == 'S':
            x_diff = dx[3]
            y_diff = dy[3]
            self.direction = 3 * np.pi / 2

        x, y = self.get_position()

        if 0 <= x + x_diff < dim_w and 0 <= y + y_diff < dim_h:
            self.position = (x + x_diff, y + y_diff)

        return self.position

    ################################################################################################################
    def new_position_with_given_action(self, dim_w, dim_h, action):

        dx = [-1, 0, 1,  0]  # 0:W ,  1:N , 2:E  3:S
        dy = [0, 1, 0, -1]

        x_diff = 0
        y_diff = 0

        new_position = self.position
        if action == 'W':
            x_diff = dx[0]
            y_diff = dy[0]
            self.direction = 2 * np.pi / 2

        if action == 'N':
            x_diff = dx[1]
            y_diff = dy[1]
            self.direction = np.pi / 2

        if action == 'E':
            x_diff = dx[2]
            y_diff = dy[2]
            self.direction = 0 * np.pi / 2

        if action == 'S':
            x_diff = dx[3]
            y_diff = dy[3]
            self.direction = 3 * np.pi / 2

        x, y = self.get_position()

        if 0 <= x + x_diff < dim_w and 0 <= y + y_diff < dim_h:
            new_position = (x + x_diff, y + y_diff)

        return new_position

    ################################################################################################################
    def set_actions_probability(self, l, n, e, s, w):
        self.actions_probability['L'] = l
        self.actions_probability['N'] = n
        self.actions_probability['E'] = e
        self.actions_probability['S'] = s
        self.actions_probability['W'] = w

    ####################################################################################################################
    def angle_of_gradient(self, point, direction):

        point_position = point.get_position()
        my_position = self.get_position()

        # We need to rotate and translate the coordinate system first!
        xt = point_position[0] - my_position[0]
        yt = point_position[1] - my_position[1]

        x = np.cos(direction)*xt + np.sin(direction)*yt
        y = -np.sin(direction)*xt + np.cos(direction)*yt

        return np.arctan2(y, x)

    ################################################################################################################
    def distance(self, point):
        point_position = point.get_position()
        my_position = self.get_position()
        return sqrt((point_position[0] - my_position[0]) ** 2 + (point_position[1] - my_position[1]) ** 2)

    ################################################################################################################
    def number_visible_items(self):

        return len(self.visible_items)

    ####################################################################################################################
    def set_random_action(self):

        actions = ['N', 'E', 'S', 'W', 'L']
        self.next_action = choice(actions)
        return

    ####################################################################################################################
    def visible_agents_items(self, items, agents):

        self.visible_agents = list()
        self.visible_items = list()

        radius = self.co_radius * self.radius
        angle = self.angle * self.co_angle

        for item in items:
                
            if not item.loaded:
                if self.distance(item) < radius:
                    if -angle / 2 <= self.angle_of_gradient(item, self.direction) <= angle / 2:
                        self.visible_items.append(item)

        for i in range(0, len(agents)):
            if self.index != i:
                if self.distance(agents[i]) < radius:
                    if -angle / 2 <= self.angle_of_gradient(agents[i], self.direction) <= angle / 2:
                        self.visible_agents.append(agents[i])
    
    ####################################################################################################################
    def choose_target_f2(self,items,agents):
        # 1. Initialising the support variables
        max_level, max_distance = -1, -1
        high_level_agent, furthest_agent = None, None

        # 2. Searching for the furthest agent and for
        # the highest level agent above own level
        for visible_agent in self.visible_agents:
            if self.distance(visible_agent) > max_distance:
                max_distance = self.distance(visible_agent)
                furthest_agent = visible_agent

            if visible_agent.level > max_level:
                if visible_agent.level > self.level:
                    max_level = visible_agent.level
                    high_level_agent = visible_agent

        # 3. if agents visible but no items visible, return agent with highest level above own level,
        # or furthest agent if none are above own level;
        if len(self.visible_items) == 0 and len(self.visible_agents) > 0:
            if high_level_agent is None:
                return furthest_agent.copy()
            else:
                return high_level_agent.copy()
        # 5. if agents and items visible
        elif len(self.visible_items) > 0 and len(self.visible_agents) > 0:
            # a. Selecting the leader
            if high_level_agent is None:
                leader_agent = furthest_agent.copy()
            else:
                leader_agent = high_level_agent.copy()

            # b. and return item that this agent would choose if it had type L2;
            if leader_agent is not None:

                leader_agent.agent_type = 'l2'
                leader_agent.level = self.level
                leader_agent.radius = self.radius
                leader_agent.angle = self.angle
                leader_agent.visible_agents_items(items, agents)

                return leader_agent.choose_target(items, agents)
            else:
                return position.position(-1, -1)
        else:
            return position.position(-1, -1)

    ####################################################################################################################
    def choose_target_f1(self,items,agents):
        # 1. Initialising the support variables
        max_distance = -1
        furthest_agent = None

        # 2. Searching for the furthest agent
        for visible_agent in self.visible_agents:
            if self.distance(visible_agent) > max_distance:
                max_distance = self.distance(visible_agent)
                furthest_agent = visible_agent

        # 3. if agents visible but no items visible, return furthest agent;
        if len(self.visible_items) == 0 and len(self.visible_agents) > 0:
            return copy(furthest_agent)

        # 4. if agents and items visible, return item that furthest agent would choose if it had type L1;
        elif len(self.visible_items) > 0 and len(self.visible_agents) > 0:
            if furthest_agent is not None:

                furthest_agent = copy(furthest_agent)
                furthest_agent.agent_type = 'l1'
                furthest_agent.level = self.level
                furthest_agent.radius = self.radius
                furthest_agent.angle = self.angle
                furthest_agent.visible_agents_items(items, agents)

                return furthest_agent.choose_target(items, agents)
            else:
                return position.position(-1, -1)
        else:
            return position.position(-1, -1)

    ####################################################################################################################
    def choose_target_l2(self,items,agents):
        # 1. Initialising the support variables
        max_index, max_level = -1, -1

        # 2. Searching for highest level item below own level
        for i in range(0, len(self.visible_items)):
            if self.visible_items[i].level > max_level:
                if self.visible_items[i].level < self.level:
                    max_level = self.visible_items[i].level
                    max_index = i

        # - return the result if the item was found
        if max_index > -1:
            return self.visible_items[max_index].position
        # 3. Else search for highest level item
        else:
            for i in range(0, len(self.visible_items)):
                if self.visible_items[i].level > max_level:
                    max_level = self.visible_items[i].level
                    max_index = i

            # - return the result if the item was found 
            if max_index > -1:
                return self.visible_items[max_index].position
            # 4. Else return no item
            else:
                return position.position(-1, -1)

    ####################################################################################################################
    def choose_target_l1(self,items,agents):
        # 1. Initialising the support variables
        max_index, max_distance = -1, -1

        # 2. Searching for max distance item
        for i in range(0, len(self.visible_items)):
            if self.distance(self.visible_items[i]) > max_distance:
                max_distance = self.distance(self.visible_items[i])
                max_index = i

        # 3. Returning the result
        if max_index > -1:
            return self.visible_items[max_index]
        else:
            return position.position(-1, -1)

    ####################################################################################################################
    def choose_target(self, items, agents):
        # 1. Choosing target (item) if type L1
        if self.agent_type == "l1":
            return self.choose_target_l1(items,agents)

        # 2. Choosing target (item) if type L2
        elif self.agent_type == "l2":
            return self.choose_target_l2(items,agents)

        # 3. Choosing target (agent) if type F1
        elif self.agent_type == "f1":
            return self.choose_target_f1(items,agents)

        # 4. Choosing target (agent) if type F2
        elif self.agent_type == "f2":
            return self.choose_target_f2(items,agents)

        else:
            return position.position(-1, -1)



