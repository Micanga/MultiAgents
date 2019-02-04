from numpy.random import choice
import position
import numpy as np
import UCT
from math import sqrt
import parameter_estimation
import unknown_agent
from copy import copy, deepcopy


class Agent:
    def __init__(self, x, y, direction, is_enemy = False):
        self.position = (int(x), int(y))
        if isinstance(direction, basestring):
            self.direction = self.convert_direction(direction)
        else:
            self.direction = float(direction)

        self.level = None
        self.apply_adversary = None
        self.uct = None

        self.visible_agents = []
        self.is_enemy =  is_enemy

        self.next_action = None
        self.previous_state = None

    ####################################################################################################################
    def initialise_uct(self, uct):
        self.uct = uct

    ####################################################################################################################
    def initialise_visible_agents(self, sim, generated_data_number, PF_add_threshold, train_mode , type_selection_mode,
                                  parameter_estimation_mode, polynomial_degree, apply_adversary,
                                  type_estimation_mode, mutation_rate):

        agent_index = 0
        self.apply_adversary = apply_adversary
        for agent in sim.agents:
            x, y = agent.get_position()
            a = unknown_agent.Agent(x, y, agent.direction,agent_index)

            self.visible_agents.append(a)
            agent_index +=1

        if sim.enemy_agent is not None:
            x, y = sim.enemy_agent.get_position()
            a = unknown_agent.Agent(x, y, sim.enemy_agent.direction,agent_index)

            self.visible_agents.append(a)

        for unknown_a in self.visible_agents:

            param_estim = parameter_estimation.ParameterEstimation(generated_data_number, PF_add_threshold, train_mode,
                                                                   apply_adversary,mutation_rate,unknown_a, sim)

            param_estim.estimation_initialisation()

            param_estim.estimation_configuration(type_selection_mode, parameter_estimation_mode, polynomial_degree,type_estimation_mode)

            unknown_a.agents_parameter_estimation = param_estim

        return
    ####################################################################################################################
    def update_unknown_agents(self, sim):
        enemy_index = 0
        for i in range(len(sim.agents)):
            self.visible_agents[i].previous_agent_status = sim.agents[i]

        enemy_index = i + 1
        if self.apply_adversary:          
            self.visible_agents[enemy_index].previous_agent_status = sim.enemy_agent

    ####################################################################################################################
    def update_unknown_agents_status(self, sim):
        enemy_index = 0
        for i in range(len(sim.agents)):
            self.visible_agents[i].next_action = sim.agents[i].next_action
            self.visible_agents[i].direction = sim.agents[i].direction
            self.visible_agents[i].position = sim.agents[i].position
            if sim.agents[i].next_action == 'L':
                self.visible_agents[i].last_loaded_item_pos = sim.agents[i].last_loaded_item_pos
                self.visible_agents[i].item_to_load = sim.agents[i].item_to_load #todo wrong


        enemy_index = i + 1

        if self.apply_adversary:
            self.visible_agents[enemy_index].next_action = sim.enemy_agent.next_action
            self.visible_agents[enemy_index].direction = sim.enemy_agent.direction
            self.visible_agents[enemy_index].position = sim.enemy_agent.position

    ####################################################################################################################
    def move(self,reuse_tree,main_sim,search_tree, time_step):
        next_action,guess_move, search_tree = self.uct_planning(reuse_tree,main_sim,search_tree, time_step)
        #if self.uct.planning_for_enemy:
            #print 'action for enemy : ', next_action
        #else:
            #print 'action for main : ', next_action
        reward = self.uct.do_move(main_sim, next_action,  real=True)
        return reward , guess_move,search_tree

    ####################################################################################################################

    def uct_planning(self, reuse_tree,main_sim, search_tree, time_step):
        if not reuse_tree:
            next_action,guess_move, search_tree = self.uct.agent_planning(0, None, main_sim, self.is_enemy)
        else:
            next_action,guess_move, search_tree = self.uct.agent_planning(time_step, search_tree, main_sim, self.is_enemy)
        return next_action,guess_move, search_tree
    ####################################################################################################################

    def set_direction(self, direction):
        self.direction = direction

    ####################################################################################################################
    def get_position(self):
        return self.position[0], self.position[1]

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
        copy_agent.level = self.level

        copy_agent.next_action = self.next_action

        copy_agents = list()

        for cagent in self.visible_agents:

            (x, y) = cagent.get_position()

            copy_unknown_agent = unknown_agent.Agent(x, y, cagent.direction,  cagent.index)

            copy_unknown_agent.index = cagent.index
            copy_agents.append(copy_unknown_agent)

        copy_agent.visible_agents = copy_agents

        return copy_agent

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

    ####################################################################################################################
    def estimation(self,time_step,main_sim,enemy_action_prob, types):
        # For the unkown agents, estimating the parameters and types
        for unknown_agent in self.visible_agents:
            if unknown_agent is not None:
                # 1. Selecting the types
                parameter_estimation = unknown_agent.agents_parameter_estimation
                if parameter_estimation.type_selection_mode == 'AS':
                    selected_types = types
                if parameter_estimation.type_selection_mode == 'BS':
                    selected_types = parameter_estimation.UCB_selection(time_step)  # returns l1, l2, f1, f2,w
                
                # 2. Defining the next agent action or appending the action
                # for the history based method
                if parameter_estimation.train_mode == 'history_based':
                    parameter_estimation.action_history.append(unknown_agent.next_action)
                    if unknown_agent.next_action != 'L':
                        parameter_estimation.actions_to_reach_target.append(unknown_agent.next_action)

                # 3. Estimating
                if unknown_agent.next_action is not None:
                    tmp_sim = copy(main_sim)
                    tmp_previous_state = copy(self.previous_state)
                    parameter_estimation.process_parameter_estimations(unknown_agent,\
                        tmp_previous_state, tmp_sim, enemy_action_prob, selected_types)