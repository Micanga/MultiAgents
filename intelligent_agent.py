from numpy.random import choice
import position
import numpy as np
import UCT
from math import sqrt
import parameter_estimation
import unknown_agent
from copy import deepcopy


class Agent:
    def __init__(self, x, y, direction ):
        self.position = (int(x), int(y))
        self.level = None

        if isinstance(direction, basestring):
            self.direction = self.convert_direction(direction)
        else:
            self.direction = float(direction)

        self.visible_agents = []

        self.next_action = None

    ####################################################################################################################
    def initialise_uct(self, uct):

        self.uct = uct

    ####################################################################################################################
    def initialise_visible_agents(self, sim, generated_data_number, PF_add_threshold, train_mode , type_selection_mode,
                                  parameter_estimation_mode, polynomial_degree,apply_adversary):

        for agent in sim.agents:
            x, y = agent.get_position()
            a = unknown_agent.Agent(x, y, agent.direction)
            self.visible_agents.append(a)

        if sim.enemy_agent is not None:
            x, y = sim.enemy_agent.get_position()
            a = unknown_agent.Agent(x, y, sim.enemy_agent.direction)
            self.visible_agents.append(a)

        for unknown_a in self.visible_agents:
            param_estim = parameter_estimation.ParameterEstimation(generated_data_number, PF_add_threshold, train_mode,apply_adversary)
            param_estim.estimation_initialisation()
            param_estim.estimation_configuration(type_selection_mode, parameter_estimation_mode, polynomial_degree)
            param_estim.choose_target_state = deepcopy(sim)
            unknown_a.agents_parameter_estimation = param_estim
            # Initial values for parameters and types

    ####################################################################################################################

    def update_unknown_agents(self, sim):
        for i in range(len(sim.agents)):
            self.visible_agents[i].agents_parameter_estimation.previous_state = sim
            self.visible_agents[i].agents_parameter_estimation.previous_agent_status = sim.agents[i]

    ####################################################################################################################
    def update_unknown_agents_actions(self, sim):
        for i in range(len(sim.agents)):
            self.visible_agents[i].next_action = sim.agents[i].next_action

    ####################################################################################################################
    def estimation(self,time_step,main_sim):
        for u_a in self.visible_agents:
            new_estimated_parameter, x_train = u_a.agents_parameter_estimation.process_parameter_estimations(time_step,
                                                                                                            u_a.next_action,
                                                                                                            main_sim)
            #
            # # print 'x_train in step ',time_step,' is ', x_train
            # # print 'x_train_set is ', x_train_set
            #
            # a_data_set = np.transpose(np.array(x_train))
            # n = time_step
            # if a_data_set != []:
            #     # print ' Calculatinggggggggg mean '
            #     levels = a_data_set[0, :]
            #     angle = a_data_set[1, :]
            #     radius = a_data_set[2, :]
            #
            #     dataMean[n, 0] = np.mean(levels)
            #     dataStd[n, 0] = np.std(levels, ddof=1)
            #     dataMean[n, 1] = np.mean(angle)
            #     dataStd[n, 1] = np.std(angle, ddof=1)
            #     dataMean[n, 2] = np.mean(radius)
            #     dataStd[n, 2] = np.std(radius, ddof=1)
            # else:
            #     dataMean[n, 0] = 0
            #     dataStd[n, 0] = 0
            #     dataMean[n, 1] = 0
            #     dataStd[n, 1] = 0
            #
            #     dataMean[n, 2] = 0
            #     dataStd[n, 2] = 0
            # # print dataMean
            # # print dataStd
            # x_train_set.append(x_train)

            # print 'true parameters:', str(main_sim.agents[i].level),  str(main_sim.agents[i].radius), str(main_sim.agents[i].angle)
            # print 'estimated parameters:', str(new_estimated_parameter.level), str(new_estimated_parameter.radius), str(new_estimated_parameter.angle)

    ####################################################################################################################
    def move(self,reuse_tree,main_sim,search_tree, time_step):
        next_action, search_tree = self.uct_planning(reuse_tree,main_sim,search_tree, time_step)
        reward = self.uct.do_move(main_sim, next_action,  real=True)
        return reward , search_tree

    ####################################################################################################################

    def uct_planning(self, reuse_tree,main_sim, search_tree, time_step):
        if not reuse_tree:
            next_action, search_tree = self.uct.agent_planning(0, None, main_sim, False)
        else:
            next_action, search_tree = self.uct.agent_planning(time_step, search_tree, main_sim, False)
        return next_action, search_tree
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




