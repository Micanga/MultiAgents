from math import *
import agent
from numpy.random import choice

totalItems = 0  # TODO: Adding as a global variable for now

actions = ['L', 'N', 'E', 'S', 'W']

root = None

class Q_table_row:
    def __init__(self, action, QValue, sumValue, trials):
        self.action = action
        self.QValue = QValue
        self.sumValue = sumValue
        self.trials = trials


########################################################################################################################
class State:

    def __init__(self, simulator):
        self.simulator = simulator

    def equals(self, state):
        return self.simulator.equals(state.simulator)


########################################################################################################################
class Node:

    def __init__(self, depth, state, enemy,delta, parent=None):

        self.parentNode = parent  # "None" for the root node
        self.depth = depth

        self.state = state
        self.enemy = enemy

        self.Q_table = self.create_empty_table()

        self.untried_moves = self.create_possible_moves()
        self.childNodes = []
        self.action = None
        self.delta = delta

        self.visits = 0

        self.numItems = state.simulator.items_left()

    # @staticmethod
    ####################################################################################################################
    def create_empty_table(self):
        Qt = list()
        Qt.append(Q_table_row('L', 0, 0, 0))
        Qt.append(Q_table_row('N', 0, 0, 0))
        Qt.append(Q_table_row('E', 0, 0, 0))
        Qt.append(Q_table_row('S', 0, 0, 0))
        Qt.append(Q_table_row('W', 0, 0, 0))
        # if self.enemy:
        #     Qt.append(Q_table_row('V', 0, 0, 0))
        return Qt

    ####################################################################################################################
    def update(self, action, result):

        # TODO: We should change the table to a dictionary, so that we don't have to find the action
        for i in range(len(self.Q_table)):
            if self.Q_table[i].action == action:
                self.Q_table[i].trials += 1
                self.Q_table[i].sumValue += result
                self.Q_table[i].QValue = self.Q_table[i].sumValue / self.Q_table[i].trials
                return

    ####################################################################################################################
    def uct_select_child(self):

        # UCB expects mean between 0 and 1.
        s = \
            sorted(self.childNodes,
                   key=lambda c: c.expectedReward / self.numItems + sqrt(2 * log(self.visits) / c.visits))[
                -1]
        return s

    ####################################################################################################################

    def uct_select_action(self):

        maxUCB = -1
        maxA = None
        if self.enemy:
            maxUCB = 10000000000

        for a in range(len(self.Q_table)):
            if self.valid(self.Q_table[a].action):

                # currentUCB = self.Q_table[a].QValue + sqrt(2.0 * log(float(self.visits)) / float(self.Q_table[a].trials))

                # TODO: The exploration constant could be set up in the configuration file
                if (self.Q_table[a].trials > 0 and not self.enemy):
                    currentUCB = self.Q_table[a].QValue + 0.5 * sqrt(
                        log(float(self.visits)) / float(self.Q_table[a].trials))
                elif (self.Q_table[a].trials > 0 and self.enemy):
                    currentUCB = self.Q_table[a].QValue + 0.5 * sqrt(
                        log(float(self.visits)) / float(self.Q_table[a].trials))
                else:
                    currentUCB = 0

                if  currentUCB > maxUCB:
                    maxUCB = currentUCB
                    maxA = self.Q_table[a].action

        return maxA
    ####################################################################################################################
    def add_child(self, state, enemy):

        n = Node(parent=self, depth=self.depth + 1, state=state,enemy=enemy)
        self.childNodes.append(n)

        return n

    ####################################################################################################################
    def add_child_one_state(self, action, state, enemy, delta = 1):

        n = Node(parent=self, depth=self.depth + 1, state=state , enemy=enemy , delta = delta )
        n.action = action
        self.childNodes.append(n)

        return n

    ####################################################################################################################
    def valid(self, action):  # Check in order to avoid moving out of board.

        if self.enemy:
            (x, y) = self.state.simulator.enemy_agent.get_position()
        else:
            (x, y) = self.state.simulator.main_agent.get_position()

        m = self.state.simulator.dim_w
        n = self.state.simulator.dim_h
        for obstacle in self.state.simulator.obstacles:
            (x_o , y_o) = obstacle.get_position()
            if x == x_o and y==y_o:
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

    def create_possible_moves(self):

        if self.enemy:
            (x, y) = self.state.simulator.enemy_agent.get_position()
        else:
            (x, y) = self.state.simulator.main_agent.get_position()

        m = self.state.simulator.dim_w
        n = self.state.simulator.dim_h

        untried_moves = ['L', 'N', 'E', 'S', 'W']
        # if self.enemy:
        #     # VOID operation, because N(op) and W(ait) are taken
        #     untried_moves.append('V')

        # Check in order to avoid moving out of board.
        if x == 0:
            untried_moves.remove('W')

        if y == 0:
            untried_moves.remove('S')

        if x == m - 1:
            untried_moves.remove('E')

        if y == n - 1:
            untried_moves.remove('N')

        return untried_moves


########################################################################################################################
class UCT:
    def __init__(self,  iteration_max, max_depth, do_estimation, mcts_mode,apply_adversary, enemy):

        self.iteration_max = iteration_max
        self.max_depth = max_depth
        self.do_estimation = do_estimation
        self.mcts_mode = mcts_mode
        self.planning_for_enemy = enemy
        self.apply_adversary = apply_adversary

    ####################################################################################################################
    def do_move(self, sim, move,enemy=False, real=False):

        if enemy:
            tmp_m_agent = sim.enemy_agent
        # elif enemy and not real:
        #     tmp_m_agent = sim.suspect_agent
        else:
            tmp_m_agent = sim.main_agent

        get_reward = 0

        if move == 'L':
            if real and self.planning_for_enemy:
                return 0
            load_item, (item_position_x, item_position_y) = tmp_m_agent.is_agent_face_to_item(sim)
            if load_item :
                destination_item_index = sim.find_item_by_location(item_position_x, item_position_y)
                if sim.items[destination_item_index].level <= tmp_m_agent.level:
                    sim.items[destination_item_index].loaded = True
                    get_reward += float(1.0)
                else:
                    sim.items[destination_item_index].agents_load_item.append(tmp_m_agent)
        else:
            #if real and self.planning_for_enemy:
                #print 'x'
            (x_new, y_new) = tmp_m_agent.new_position_with_given_action(sim.dim_w, sim.dim_h, move)

            # If there new position is empty
            if sim.position_is_empty(x_new, y_new):
                tmp_m_agent.next_action = move
                tmp_m_agent.change_position_direction(sim.dim_w, sim.dim_h)
            else:
                tmp_m_agent.change_direction_with_action(move)

            if enemy:
                sim.enemy_agent = tmp_m_agent
            else:
                sim.main_agent = tmp_m_agent

        sim.update_the_map()

        return get_reward

    ####################################################################################################################
    def best_action(self, node):
        Q_table = node.Q_table

        tieCases = []

        # Swaps between min and max depending on if its an enemy move or a main agent move
        maxA = None
        maxQ = -100000000000
        # if not self.enemy:
        for a in range(len(Q_table)):
            if Q_table[a].QValue > maxQ and Q_table[a].trials > 0:
                maxQ = Q_table[a].QValue
                maxA = Q_table[a].action
        # else:
        #     maxQ = 100000000000
        #     for a in range(len(Q_table)):
        #         if Q_table[a].QValue < maxQ and Q_table[a].trials > 0:
        #             maxQ = Q_table[a].QValue
        #             maxA = Q_table[a].action

        for a in range(len(Q_table)):
            if (Q_table[a].QValue == maxQ):
                tieCases.append(a)
        if len(tieCases) > 0:
            maxA = Q_table[choice(tieCases)].action

        # maxA=node.uct_select_action()
        return maxA

    ########################################################################################

    def best_enemy_action(self, node, action):
        nodes = node.childNodes
        node = None
        #print 'best_enemy_action:',action,node
        for n in range(len(nodes)):
            if nodes[n].action == action:
                node = nodes[n]
        if node is None:
            return 0

        Q_table = node.Q_table
        sumQ = 0
        probabilities = []
        for i in range(len(Q_table)):
            sumQ += Q_table[i].QValue

        for i in range(len(Q_table)):
            if (Q_table[i].trials > 0):
                probabilities.append(sumQ - Q_table[i].QValue)
            else:
                probabilities.append(0)
        sumQ = sum(probabilities)
        if (sumQ > 0):
            sumQ = 1 / sumQ
        else:
            sumQ = 0
        for i in range(len(probabilities)):
            probabilities[i] *= sumQ
        # print "Total after belief is ", sum(probabilities)
        # print "Probabilities: ", probabilities
        # self.print_Q_table(node)

        return max(probabilities)
    ################################################################################################################

    def terminal(self, state):
        if state.simulator.items_left() == 0:
            return True
        return False

    ################################################################################################################
    def leaf(self, main_time_step, node):
        if node.depth == main_time_step + self.max_depth + 1:
            return True
        return False

    ################################################################################################################
    def evaluate(self, node):
        return node.expectedReward

    ################################################################################################################
    def select_action(self, node):
        # If all *actions* of the current node have been tried at least once, then Select Child based on UCB

        if node.untried_moves == []:
            return node.uct_select_action()

        # If there is some untried moves we will select a random move from the untried ones
        if node.untried_moves != []:
            move = choice(node.untried_moves)
            node.untried_moves.remove(move)
            return move

    ################################################################################################################

    def simulate_action(self, state, action,enemy = False):

        sim = state.simulator.copy()
        next_state = State(sim)

        # Run the A agent to get the actions probabilities
        tmp_main_agent = sim.main_agent
        for u_a in tmp_main_agent.visible_agents:
            selected_type = u_a.get_sampled_probability()
            if selected_type != 'w':
                x, y = u_a.get_position()
                agents_estimated_values = u_a.estimated_parameter.get_parameters_for_selected_type (selected_type)
                tmp_agent = agent.Agent(x, y, u_a.direction, selected_type, '-1')
                tmp_agent.set_parameters(sim, agents_estimated_values.level,
                                              agents_estimated_values.radius,
                                              agents_estimated_values.angle)

                sim.move_a_agent(tmp_agent)

        m_reward = self.do_move(sim, action,enemy)

        a_reward = sim.update_all_A_agents(True)

        if sim.do_collaboration():
            c_reward = float(1)
        else:
            c_reward = 0

        total_reward = float(m_reward + a_reward + c_reward) / totalItems

        return next_state, total_reward

    ################################################################################################################
    #@staticmethod
    def find_new_root(self,previous_root, current_state, enemy):
        # Initialise with new node, just in case the child was not yet expanded
        new_root_node = Node(depth = previous_root.depth + 1, state =current_state ,
                         enemy = enemy,
                         delta = previous_root.delta)

        if self.apply_adversary:
            for enemyNode in previous_root.childNodes:
                for child in enemyNode.childNodes:
                    if child.state.equals(current_state):
                        new_root_node = child
                        break
        else:
            for child in previous_root.childNodes:
                if child.state.equals(current_state):
                    new_root_node = child
                    break

        return new_root_node

    ################################################################################################################
    def search(self, main_time_step, node):
        # 1. Defining and evaluating the current
        # simulation state
        state = node.state
        if self.terminal(state) or self.leaf(main_time_step, node):
            return 0

        # 2. Defining the simulation action and
        # calculating its reward
        action = self.select_action(node)
        (next_state, reward) = self.simulate_action(node.state, action,node.enemy)

        # 3. Adding/Getting the action node
        next_node = None
        if self.mcts_mode == 'UCT':
            for child in node.childNodes:
                if child.state.equals(next_state):
                    next_node = child
                    break

            if next_node is None:
                if self.apply_adversary:
                    next_node = node.add_child(next_state, not node.enemy)
                else:
                    next_node = node.add_child(next_state, node.enemy)

        if self.mcts_mode == 'UCTH':
            for child in node.childNodes:
                if child.action == action:
                    next_node = child
                    next_node.state = next_state
                    break

            if next_node is None:
                if self.apply_adversary:
                    next_node = node.add_child_one_state(action, next_state, not node.enemy, -1 * node.delta)
                else:
                    next_node = node.add_child_one_state(action, next_state,  node.enemy, 1)

        # 4. Updating the current node
        discount_factor = 0.95
        q = node.delta * reward + discount_factor *  self.search(main_time_step, next_node)
        
        node.update(action, q)
        node.visits += 1
        return q

    ####################################################################################################################
    # @TODO: make the enemy agent not estimate the main agents actions or not guess who the enemy agent is
    def monte_carlo_planning(self, main_time_step, search_tree, simulator,  enemy):
        global root

        # 1. Defining the current state
        current_state = State(simulator)

        # 2. Defining the current root
        if search_tree is None:
            root_node = Node(depth=0, state=current_state, enemy=enemy,delta = 1)
        else:
            root_node = self.find_new_root(search_tree, current_state, enemy)

        # 3. Updating the root
        node = root_node
        root = node

        # 4. Running the search over the current node
        time_step = 0
        while time_step < self.iteration_max:
            # a. creating a temporary simulation
            tmp_sim = simulator.copy()
            node.state.simulator = tmp_sim

            # b. perform the search process
            self.search(main_time_step, node)
            time_step += 1

        # 5. Selecting the best action
        best_selected_action = self.best_action(node)

        # 6. Evaluating the enemy
        enemy_probabilities = None
        if self.apply_adversary and not self.planning_for_enemy:
            enemy_probabilities = self.best_enemy_action(node, best_selected_action)

        return best_selected_action, enemy_probabilities, node

    ####################################################################################################################
    def agent_planning(self, time_step, search_tree, sim,  enemy):
        global totalItems

        tmp_sim = sim.copy()

        # We need total items, because the QValues must be between 0 and 1
        # If we are re-using the tree, I think we should use the initial number of items, and not update it
        if search_tree is None:
            totalItems = tmp_sim.items_left()

        next_move, enemy_prediction, search_tree = self.monte_carlo_planning(time_step, search_tree, tmp_sim,
                                                            enemy)

        return next_move, enemy_prediction, search_tree

    ####################################################################################################################
    def print_search_tree(self, main_time_step):

        node = root

        for i in range(self.max_depth + main_time_step):
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "Node depth: ",node.depth
            self.print_nodes(node.childNodes)

        print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    ####################################################################################################################
    def print_nodes(self, childNodes):
        print('Total number of children:', len(childNodes))
        for i in range(len(childNodes)):
            print 'Node with action : ',childNodes[i].action
            # self.print_Q_table(childNodes[i])
            # tmpnode = childNodes[i]
            # if len(tmpnode.childNodes) > 0:
            #     self.print_nodes(tmpnode.childNodes)
            # print childNodes[i].state.simulator.draw_map()

    ####################################################################################################################
    def print_Q_table(self, node):
        for a in range(len(node.Q_table)):
            print "Action: ", node.Q_table[a].action, "QValue:", node.Q_table[a].QValue, "sumValue:", node.Q_table[
                a].sumValue, "trials:", node.Q_table[a].trials
