from math import *
from numpy.random import choice
import random
from copy import copy
import agent


actions = ['L', 'N', 'E', 'S', 'W']

root = None


class QTableRow:
    def __init__(self, action, q_value, sum_value, trials):
        self.action = action
        self.QValue = q_value
        self.sumValue = sum_value
        self.trials = trials


########################################################################################################################
class State:

    def __init__(self, simulator):
        self.simulator = simulator

    def equals(self, state):
        return self.simulator.equals(state.simulator)

################################################################################################################


class Node:

    def __init__(self, history, depth, state,  parent=None):

        self.parentNode = parent  # "None" for the root node
        self.history = history
        self.depth = depth
        self.childNodes = []
        self.cumulativeRewards = 0
        self.immediateReward = 0
        self.expectedReward = 0

        self.beliefState = state
        self.particleFilter = []
        self.QTable = self.create_empty_table()
        self.visits = 0  # N(h)
        self.value = 0    # V(h)
        self.numItems = state.simulator.items_left()
        self.untriedActions = self.create_possible_actions()

    ####################################################################################################################
    @staticmethod
    def create_empty_table():
        Qt = list()
        Qt.append(QTableRow('L', 0.0, 0.0, 0))
        Qt.append(QTableRow('N', 0.0, 0.0, 0))
        Qt.append(QTableRow('E', 0.0, 0.0, 0))
        Qt.append(QTableRow('S', 0.0, 0.0, 0))
        Qt.append(QTableRow('W', 0.0, 0.0, 0))
        return Qt

    ####################################################################################################################
    def uct_select_child(self):

        # UCB expects mean between 0 and 1
        s = sorted(self.childNodes, key=lambda c: c.expectedReward/self.numItems + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    ###################################################################################################################

    def update(self, action, result):

        # TODO: We should change the table to a dictionary, so that we don't have to find the action
        for i in range(len(self.QTable)):
            if self.QTable[i].action == action:
                self.QTable[i].trials += 1
                self.QTable[i].sumValue += result
                #self.QTable[i].QValue = self.QTable[i].sumValue / self.QTable[i].trials
                self.QTable[i].QValue += (result - self.QTable[i].QValue) / self.QTable[i].trials
                return

    ####################################################################################################################

    def uct_select_action(self):

        maxUCB = -1
        maxA = None

        for a in range(len(self.QTable)):
            if self.valid(self.QTable[a].action):

                # TODO: The exploration constant could be set up in the configuration file
                if self.QTable[a].trials > 0:
                    current_ucb = self.QTable[a].QValue + 0.5 * sqrt(
                        log(float(self.visits)) / float(self.QTable[a].trials))

                else:
                    current_ucb = 0

                if current_ucb > maxUCB:
                    maxUCB = current_ucb
                    maxA = self.QTable[a].action

        if maxA is None:
            maxA = random.choice(actions)

        return maxA

    ####################################################################################################################

    def valid(self, action):  # Check in order to avoid moving out of board.

        # if self.enemy:
        #     (x, y) = self.state.simulator.enemy_agent.get_position()
        # else:
        (x, y) = self.beliefState.simulator.main_agent.get_position()

        m = self.beliefState.simulator.dim_w
        n = self.beliefState.simulator.dim_h
        for obstacle in self.beliefState.simulator.obstacles:
            (x_o, y_o) = obstacle.get_position()
            if x == x_o and y == y_o:
                return False
        if x == 0:
            if action == 'W':
                return False

        if y == 0:
            if action == 'S':
                return False
        if x == m - 1:
            if action == 'E':
                return False

        if y == n - 1:
            if action == 'N':
                return False

        return True


    ####################################################################################################################

    def best_action(self):
        q_table = self.QTable

        tieCases = []

        # Swaps between min and max depending on if its an enemy move or a main agent move
        maxA = None
        maxQ = -100000000000
        # if not self.enemy:
        for a in range(len(q_table)):
            print q_table[a].action, q_table[a].QValue
            if q_table[a].QValue > maxQ and q_table[a].trials > 0:
                maxQ = q_table[a].QValue
                maxA = q_table[a].action

        for a in range(len(q_table)):
            if q_table[a].QValue == maxQ:
                tieCases.append(a)
        if len(tieCases) > 0:
            maxA = q_table[choice(tieCases)].action

        if maxA is None:
            maxA = random.choice(actions)

        return maxA

    ################################################################################################################
    def select_action(self):
        # If all *actions* of the current node have been tried at least once, then Select Child based on UCB

        if self.untriedActions == []:
            return self.uct_select_action()

        # If there is some untried moves we will select a random move from the untried ones
        if self.untriedActions!= []:

            move = choice(self.untriedActions)
            self.untriedActions.remove(move)
            return move

    ####################################################################################################################
    def create_possible_actions(self):

        # if self.enemy:
        #     (x, y) = self.beliefState.simulator.enemy_agent.get_position()
        # else:
        (x, y) = self.beliefState.simulator.main_agent.get_position()

        m = self.beliefState.simulator.dim_w
        n = self.beliefState.simulator.dim_h

        untried_actions = ['N', 'S', 'E', 'W', 'L']

        if x == 0:
            untried_actions.remove('E')

        if y == 0:
            untried_actions.remove('S')

        if x == m - 1:
            untried_actions.remove('W')

        if y == m - 1:
            untried_actions.remove('N')

        return untried_actions


################################################################################################################
class ONode(Node):

    def __init__(self,history, depth, state, observation=None, parent=None):

        Node.__init__(self,history=history, depth=depth, state=state,  parent=parent)
        self.observation = observation  # the Action that got us to this node - "None" for the root node

    def add_child(self, h, s,a):

        n = ANode(h, self.depth + 1, s, a, self)

        self.childNodes.append(n)

        return n


################################################################################################################
class ANode(Node):

    def __init__(self,history, depth, state, action=None, parent=None):

        Node.__init__(self, history=history,depth = depth, state= state,  parent=parent)
        self.action = action  # the Action that got us to this node - "None" for the root node

    def add_child(self, h, s,o):

        n = ONode( history=h, state=s, observation=o, parent=self, depth=self.depth + 1)
     #   self.untriedActions.remove(a)
        self.childNodes.append(n)

        return n


########################################################################################################################
class POMCP:
    def __init__(self,  iteration_max, max_depth, do_estimation):

        self.iteration_max = iteration_max
        self.max_depth = max_depth
        self.do_estimation = do_estimation
        self.planning_for_enemy = False

    ####################################################################################################################
    def do_move(self, sim, action, real=False):  # real: if it is the movement for real move ar the move for simulator

        # todo: elnaz: sim should change to the current state anr anything related to POMCP class
        tmp_m_agent = sim.main_agent

        get_reward = 0

        if action == 'L':
            load_item, (item_position_x, item_position_y) = tmp_m_agent.is_agent_face_to_item(sim)
            if load_item:
                destination_item_index = sim.find_item_by_location(item_position_x, item_position_y)
                if real:
                    print "Try to load item" , item_position_x, item_position_y, "  by M"
                if sim.items[destination_item_index].level <= tmp_m_agent.level:
                    if real:
                        print "loaded by M"
                    sim.items[destination_item_index].loaded = True
                    get_reward += float(1.0)
                else:
                    sim.items[destination_item_index].agents_load_item.append(tmp_m_agent)
        else:
            (x_new, y_new) = tmp_m_agent.new_position_with_given_action(sim.dim_w, sim.dim_h, action)

            # If there new position is empty
            if sim.position_is_empty(x_new, y_new):
                tmp_m_agent.next_action = action
                tmp_m_agent.change_position_direction(sim.dim_w, sim.dim_h)
            else:
                tmp_m_agent.change_direction_with_action(action)

            sim.main_agent = tmp_m_agent

        sim.update_the_map()

        return get_reward

    ########################################################################################
    @staticmethod
    def terminal(state):
        if state.simulator.items_left() == 0:
            return True
        return False

    ################################################################################################################
    def leaf(self,  node):
        main_time_step = 0
        if node.depth >= self.max_depth + 1:
            return True
        return False

    ################################################################################################################
    def simulate_action(self, state, action):

        sim = state.simulator.copy()
        next_state = State(sim)

        # Run the A agent to get the actions probabilities
        tmp_main_agent = sim.main_agent
        for u_a in tmp_main_agent.visible_agents:
            if self.do_estimation is False:
                selected_type = u_a.agent_type
                x, y = u_a.get_position()

                tmp_agent = agent.Agent(x, y, u_a.direction, selected_type, '-1')
                tmp_agent.set_parameters(sim, u_a.level,
                                         u_a.radius,
                                         u_a.angle)

                sim.move_a_agent(tmp_agent)
            else:
                if u_a.agents_parameter_estimation is not None:
                    selected_type = u_a.agents_parameter_estimation.get_sampled_probability()
                    if selected_type != 'w':
                        x, y = u_a.get_position()
                        agents_estimated_values = u_a.agents_parameter_estimation.get_parameters_for_selected_type(selected_type)
                        tmp_agent = agent.Agent(x, y, u_a.direction, selected_type, '-1')
                        tmp_agent.set_parameters(sim, agents_estimated_values.level,agents_estimated_values.radius,
                                                 agents_estimated_values.angle)

                        sim.move_a_agent(tmp_agent)

        m_reward = self.do_move(sim, action)

        a_reward = sim.update_all_A_agents(True)

        if sim.do_collaboration():
            c_reward = float(1)
        else:
            c_reward = 0

        total_reward = float(m_reward + a_reward + c_reward) / totalItems
        observation = sim.take_m_observation()
        return next_state,observation, total_reward

    ################################################################################################################

    def find_new_root(self,previous_root, previous_action,previous_observation):

        if previous_root is None:
            return None

        root_node = None
        action_node = None
        # previous_action = current_state.simulator.main_agent.next_action

        for child in previous_root.childNodes:
            if child.action == previous_action:
                action_node = child
                break

        if action_node is None :
            return root_node



        for child in action_node.childNodes:
            if self.observation_is_equal(child.observation, previous_observation):
                root_node = child
                break

        return root_node

    ################################################################################################################
    def simulate(self, node):

        state = node.beliefState

        history = node.history

        if self.terminal(state):
            return 0
        if self.leaf(node):   # todo: why main_time_step
            return 0

        if node.childNodes == []:
            for action in ['L', 'N', 'E', 'S', 'W']:
                (next_state, observation, reward) = self.simulate_action(state, action)
                # node.add_action_child(a, next_state, node.enemy)
                node.add_child(history, next_state, action)

            # b. rollout
            return self.rollout(state,history, node.depth)

        action = node.select_action()
        history.append(action)

        action_node = None

        for child in node.childNodes:
            if child.action == action:
                action_node = child
                break

        if action_node is None:
            action_node = node.add_child(history, node.beliefState, action)

        (next_state, observation, reward) = self.simulate_action(action_node.beliefState, action)

        node.particleFilter.append(state)
        history.append(observation)

        observation_node = None
        for child in action_node.childNodes:
            if self.observation_is_equal(child.observation, observation):
                observation_node = child
                break

        if observation_node is None:
            observation_node = action_node.add_child(history,  next_state, observation)

        discount_factor = 0.95
        q = reward + discount_factor * self.simulate(observation_node)

        node.update(action, q)
        node.visits += 1

        return q

    ################################################################################################################
    def observation_is_equal(self, observation, other_observation):
        if len(observation) != len(other_observation):
            return False

        for o in range(len(observation)):
            if observation[o] not in other_observation:
                return False

        return True

    ################################################################################################################

    def rollout(self, state , history, depth):

        if depth > self.max_depth or self.terminal(state):
            return 0

        # 1. Choosing the action
        # a ~ pi(h)

        action = random.choice(actions)
        # history.append(action)

        # 2. Simulating the particle
        # (s',o,r) ~ G(s,a)
        (next_state, observation, reward) = self.simulate_action(state, action)
        # history.append(observation)
        # print 'rolllllloooout'
        # next_state.simulator.draw_map()

        # 4. Calculating the reward
        R = reward + 0.95 * self.rollout(next_state,history, depth + 1)

        return R

    ####################################################################################################################
    def monte_carlo_planning(self, search_tree, simulator, history,  enemy):
        global root
        current_state = State(simulator)
        previous_action = current_state.simulator.main_agent.next_action
        previous_observation = history[-1]

        root_node = self.find_new_root(search_tree, previous_action, previous_observation)
        current_observation = simulator.main_agent.history[-1]

        if root_node is None:
            root_node = ONode(history, depth=0, state=current_state, observation = current_observation)

        node = root_node
        root = node

        best_selected_action = self.search(root_node) # main_time_step,

        # best_selected_action = node.best_action()
        enemy_probabilities = None
        print  "M's action: ", best_selected_action
        return best_selected_action, enemy_probabilities, node

    ####################################################################################################################
    def search(self, node):
        sim = node.beliefState.simulator

        iteration_number = 0
        while iteration_number < self.iteration_max:
            if node.particleFilter == list():
                beliefState = State(sim.uniform_sim_sample())
            else:
                beliefState = node.particleFilter[random.randint(0, len(node.particleFilter) - 1)]

            node.beliefState = beliefState

            # b. simulating
            self.simulate(node)

            iteration_number += 1
        return node.best_action()

    ####################################################################################################################
    def agent_planning(self, time_step, search_tree, sim,  enemy):
        global totalItems

        tmp_sim = sim.copy()
        history = copy(sim.main_agent.history)

        # We need total items, because the QValues must be between 0 and 1
        # If we are re-using the tree, I think we should use the initial number of items, and not update it
        if search_tree is None:
            totalItems = tmp_sim.items_left()

        next_move, enemy_prediction, search_tree = self.monte_carlo_planning( search_tree, tmp_sim, history, enemy)

        return next_move, enemy_prediction, search_tree

