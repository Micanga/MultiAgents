import numpy as np
from agent import Agent
import position


class UnknownAgent(Agent,object):
    def __init__(self, x, y, direction, index):
        super(UnknownAgent, self).__init__(x, y, direction, index)

        self.agents_parameter_estimation = None

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
        copy_agent = UnknownAgent(x, y, self.direction,self.index)

        copy_agent.agents_parameter_estimation = self.agents_parameter_estimation
        copy_agent.index = self.index
        copy_agent.last_loaded_item_pos = self.last_loaded_item_pos
        copy_agent.item_to_load = self.item_to_load
        copy_agent.next_action = self.next_action
        copy_agent.agent_type = self.agent_type
        copy_agent.level = self.level
        copy_agent.radius = self.radius
        copy_agent.angle = self.angle

        copy_agent.co_radius = self.co_radius
        copy_agent.co_angle = self.co_angle

        copy_agent.memory= self.memory
        copy_agent.intelligent_agent = self.intelligent_agent

        copy_agent.previous_agent_status = self.previous_agent_status
        copy_agent.choose_target_pos = self.choose_target_pos
        copy_agent.choose_target_direction = self.choose_target_direction
        copy_agent.choose_target_state =self.choose_target_state
        copy_agent.choose_target_history = self.choose_target_history
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

    def set_type_parameters(self, sim):
            selected_type = self.agents_parameter_estimation.get_sampled_probability()
            if selected_type != 'w':
                self.agent_type = selected_type
                agents_estimated_values = self.agents_parameter_estimation.get_parameters_for_selected_type(
                    selected_type)
                self.set_parameters(sim, agents_estimated_values.level,
                                   agents_estimated_values.radius,
                                   agents_estimated_values.angle)


    def find_estimated_target(self,sim):

        destination = position.position(-1, -1)
        self.visible_agents_items(sim.items, sim.agents)
        target = self.choose_target(sim.items, sim.agents)

        if target.get_position() != (-1, -1):
            destination = target

        self.memory = destination