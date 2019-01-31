
import position
import numpy as np
from math import sqrt


class Agent:
    def __init__(self, x, y, direction, index='0'):
        self.position = (int(x), int(y))

        if isinstance(direction, basestring):
            self.direction = self.convert_direction(direction)
        else:
            self.direction = float(direction)

        self.agents_parameter_estimation = None
        self.index = index
        self.last_loaded_item_pos = None
        self.item_to_load = -1
        self.next_action = None
        self.agent_type = None
        self.previous_agent_status = None
        self.choose_target_pos = None
        self.choose_target_direction = None
        self.choose_target_state = None
        self.choose_target_history = []

    ####################################################################################################################

    def get_position(self):
        return self.position[0], self.position[1]

    ####################################################################################################################
    def get_estimated_type(self):
        return self.agents_parameter_estimation.get_highest_type_probability()

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
            return 3 * np.pi / 2

    ####################################################################################################################
    def set_direction(self, direction):
        self.direction = direction

    ####################################################################################################################
    def equals(self, other_agent):
        (x, y) = self.position
         
        (other_x, other_y) = other_agent.get_position()

        return x == other_x and y == other_y and \
               self.direction == other_agent.direction

    ####################################################################################################################
    def copy(self):

        (x, y) = self.position
        copy_agent = Agent(x, y, self.direction)
        copy_agent.direction = self.direction
        return copy_agent

    ################################################################################################################

    def is_item_nearby(self, items):

        pos = self.position

        for i in range(0, len(items)):
            item = items[i]
            (xI, yI) = item.position.get_position()
            if (yI == pos[1] and abs(pos[0] - xI) == 1) or (xI == pos[0] and abs(pos[1] - yI) == 1):
                return i

        return -1
