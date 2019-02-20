from copy import deepcopy
from copy import copy
from math import *
from numpy.random import choice
from random import sample
import random

import agent
from UCT import Q_table_row, State, Node, UCT

totalItems = 0  # TODO: Adding as a global variable for now

actions = ['L', 'N', 'E', 'S', 'W']

root = None

class PONode(Node,object):

    def __init__(self, depth, state, enemy, parent=None):
        # Common Parameters
        # >> parentNode, depth, state, enemy, Q_table, untried_moves, 
        # childNodes, action, visits e numItems
        super(PONode,self).__init__(depth, state, enemy, parent)

        # PO Parameters
        self.observation = None
        self.history = list()

    def random_policy_action(self):
        possible_actions = list()
        for c in self.childNodes:
            possible_actions.append(c.action)
        return choice(possible_actions)

    def get_child_by_action(self,action):
        for c in self.childNodes:
            if c.action == action:
                return c
        return None

    def add_child(self, state, enemy):
        n = PONode(parent=self, depth=self.depth + 1, state=state,enemy=enemy)
        n.history = self.history
        self.childNodes.append(n)
        return n

    def add_action_child(self, act, state, enemy):
        new_node = PONode(parent=self, depth=self.depth + 1, state=state , enemy=enemy)
        new_node.action = act
        new_node.history = copy(self.history)
        new_node.history.append(act)
        self.childNodes.append(new_node)
        return new_node

    def add_obs_child(self, obs, state, enemy):
        new_node = PONode(parent=self, depth=self.depth + 1, state=state , enemy=enemy)
        new_node.observation = obs
        new_node.history = copy(self.history)
        self.childNodes.append(new_node)
        return new_node

    def update_depth(self,depth):
        self.depth = depth
        for c in self.childNodes:
            c.update_depth(depth+1)

    def show_q_table(self):
        print '**** Q TABLE ****'
        for row in self.Q_table:
            print row.action,row.QValue
        print '*****************'

    def destroy(self):
        for c in self.childNodes:
            c.destroy()

        self.childNodes = None
        self.Q_table = None
        self.untried_moves = None
        self.observation = None

########################################################################################################################
class POUCT(UCT,object):
    def __init__(self,  iteration_max, max_depth, do_estimation, mcts_mode,apply_adversary, enemy):
        # Common Parameters
        super(POUCT,self).__init__(iteration_max, max_depth, do_estimation, mcts_mode,apply_adversary, enemy)
        # PO Parameters
        self.belief_state = list()
        self.k = 150
        self.max_samples = 750
    
    ####################################################################################################################
    def agent_planning(self, time_step, search_tree, sim,  enemy):
        global totalItems

        # 1. Updating the item count
        if search_tree is None:
            totalItems = sim.items_left()

        # 2. Searching for best main agent movement (+add to history)
        # and the root of MCTS
        next_move, enemy_prediction, search_tree = self.monte_carlo_planning(time_step, search_tree, sim, enemy)
        sim.main_agent.history.append(next_move)

        # 3. Returning the result
        return next_move, enemy_prediction, search_tree

    ####################################################################################################################
    def monte_carlo_planning(self, main_time_step, search_tree, simulator,  enemy):
        global root

        # 1. Initializing the current state
        sim = simulator.copy()
        current_state = State(sim)

        # 2. Initializing the root
        if search_tree is None:
            root_node = PONode(depth=0, state=current_state, enemy=enemy)
        else:
            root_node = search_tree

        root = root_node

        # 3. Taking the best action
        best_selected_action = self.search(main_time_step, root_node)
        
        # root_node.show_q_table()
        # print 'best action:',best_selected_action

        # 4. Taking the anemy action
        enemy_probabilities = None
        if self.apply_adversary and not self.enemy:
            enemy_probabilities = self.best_enemy_action(root_node, best_selected_action)

        # 5. Retuning the best action and the tree search root
        return best_selected_action, enemy_probabilities, root_node

    #####
    def terminal(self, state):
        if state.simulator.main_agent.items_left() == 0:
            return True
        return False

    ################################################################################################################
    def search(self, main_time_step, node):

        sim = node.state.simulator
        state = State(sim)

        # 1. Verifying stop conditions
        if self.terminal(state) or self.leaf(main_time_step, node):
            return 0

        # 2. Starting the search main loop
        it = 0
        while it < self.iteration_max:
            if it % 10 == 0:
                print it,'/',self.iteration_max
            # a. choosing the simulation state
            if node.history == list() or self.belief_state == list():
                state = State(sim.uniform_sim_sample())
            else:
                state = self.belief_state[random.randint(0,len(self.belief_state)-1)]

            node.state = state

            # b. simulating
            self.simulate(node,0)

            # c. increasing the time step
            it += 1

        return self.best_action(node)

    ################################################################################################################
    def simulate(self,node,depth):
        # 0. Setting Simulate Parameters
        state = node.state
        history = node.history

        # 1. Checking the stop condition
        if depth > self.max_depth or self.terminal(state):
            return 0

        # 2. Searching for the history in childs
        if node.childNodes == []:
            for a in ['L', 'N', 'E', 'S', 'W']:
                (next_state, observation, reward) = self.simulate_action(state, a)
                node.add_action_child(a,next_state,node.enemy)
            # b. rollout
            return self.rollout(node,depth)

        #print 'simulate(',depth,')'
        # 3. If the history exists in the tree, choose the best action to simulate
        action, action_idx = self.get_simulate_action(node)
        action_node = node.get_child_by_action(action)

        # 4. Simulate the action
        action_node.visits += 1
        (next_state, observation, reward) = self.simulate_action(state, action)

        # 5. Adding the observation node if it not exists
        new_observation = True
        for child in action_node.childNodes:
            if self.observation_is_equal(child.observation,observation):  
                new_observation = False
                observation_node = child
                break

        if new_observation:
            observation_node = action_node.add_obs_child(observation,next_state,action_node.enemy)
        
        # 6. Calculating the reward
        observation_node.visits += 1
        R = reward + 0.95*self.simulate(observation_node,depth+1)

        # 7. Updating the node
        if depth == 0:
            self.belief_state.append(state)
        node.visits += 1

        value = node.Q_table[action_idx].QValue
        new_value = value + ((R-value)/action_node.visits)
        node.update(action,new_value)

        return R

    ################################################################################################################
    def rollout(self,node,depth):
        #print 'rollouting (',depth,')...'
        if depth > self.max_depth or self.terminal(node.state):
            return 0

        # 1. Choosing the action
        # a ~ pi(h)
        action = node.random_policy_action()

        # 2. Simulating the particle
        # (s',o,r) ~ G(s,a)
        (next_state, observation, reward) = self.simulate_action(node.state, action)

        # 3. Building the rollout node
        new_node = PONode(node.depth+1, next_state, node.enemy,None)
        for a in new_node.create_possible_moves():
            (c_next_state, c_observation, c_reward) = self.simulate_action(new_node.state, a)
            new_node.add_action_child(a,c_next_state,node.enemy)

        # 4. Calculating the reward
        R = reward + 0.95*self.rollout(new_node,depth+1)

        return R

    ################################################################################################################
    def get_simulate_action(self,node):
        action, action_idx = None, None
        actions_value = []
        for a in range(len(node.Q_table)):
            # 1. getting the child node information 
            child = node.get_child_by_action(node.Q_table[a].action)

            # -- if the node was never tried, lets try it
            if child.visits == 0:
                action_idx = a
                action = node.Q_table[a].action
                return action, action_idx

            # 2. calculating the value
            value = node.Q_table[a].QValue + sqrt(log(node.visits)/child.visits)
            actions_value.append((value,a))

        # 3. taking the action and the linked child
        action_idx = max(actions_value,key=lambda item:item[0])[1]
        action = node.Q_table[action_idx].action

        return action, action_idx

    ################################################################################################################
    def best_enemy_action(self, node, action):
        return None

    ################################################################################################################
    def simulate_action(self, state, action):
        # 1. Copying the current simulation state
        sim = state.simulator.copy()
        next_state = State(sim)

        # 2. Setting the A agents actions probabilities
        for i in range(len(sim.agents)):
            sim.agents[i] = sim.move_a_agent(sim.agents[i])

        # 3. Run the M agent simulation
        m_reward, next_state.simulator = self.do_move(next_state.simulator, action)

        # 4. Run the A agents simulation
        a_reward = self.update_all_A_agents(sim,True)

        if sim.do_collaboration():
            c_reward = float(1)
        else:
            c_reward = 0

        # 5. Calculating the total reward and taking the observation
        total_reward = float(m_reward + a_reward + c_reward) / totalItems
        observation = sim.take_m_observation()

        return next_state, observation, total_reward

    ################################################################################################################
    def do_move(self, sim, move,  enemy = False, real=False):

        if enemy:
            tmp_m_agent = sim.enemy_agent
        else:
            tmp_m_agent = sim.main_agent

        get_reward = 0

        if move == 'L':
            load_item, (item_position_x, item_position_y) = tmp_m_agent.is_agent_face_to_item(sim)
            if load_item:
                destination_item_index = sim.find_item_by_location(item_position_x, item_position_y)
                if sim.items[destination_item_index].level <= tmp_m_agent.level:
                    sim.items[destination_item_index].loaded = True
                    get_reward += float(1.0)
                else:
                    sim.items[destination_item_index].agents_load_item.append(tmp_m_agent)
        elif move != 'V':
            (x_new, y_new) = tmp_m_agent.new_position_with_given_action(sim.dim_w, sim.dim_h, move)

            # If there new position is empty
            if sim.position_is_empty(x_new, y_new):
                tmp_m_agent.next_action = move
                tmp_m_agent.change_position_direction(sim.dim_w, sim.dim_h)
            else:
                tmp_m_agent.change_direction_with_action(move)

            if enemy:
                sim.suspect_agent = tmp_m_agent
            else:
                sim.main_agent = tmp_m_agent

        sim.update_the_map()

        return get_reward, sim

    def update_all_A_agents(self, sim, simulated):
        reward = 0

        for i in range(len(sim.agents)):
            # if not (simulated and self.agents[i].get_position() == self.suspect_agent.get_position()):

            next_action = choice(actions,
                                 p=sim.agents[i].get_actions_probabilities())  # random sampling the action

            sim.agents[i].next_action = next_action

            reward += self.update(sim, i)

        return reward
    
    def update(self, sim, a_agent_index):
        reward = 0
        loaded_item = None
        a_agent = sim.agents[a_agent_index]

        if a_agent.next_action == 'L' and a_agent.item_to_load != -1:
            # print 'loading :', a_agent.item_to_load.get_position()
            destination = a_agent.item_to_load

            item_idx = sim.find_item_by_location(destination.get_position()[0],destination.get_position()[1])
                
            if destination.level <= a_agent.level:  # If there is a an item nearby loading process starts

                # load item and and remove it from map  and get the direction of agent when reaching the item.
                a_agent = sim.load_item(a_agent, item_idx)
                loaded_item = sim.items[item_idx]
                reward += 1
            else:
                if not sim.items[item_idx].is_agent_in_loaded_list(a_agent):
                    sim.items[item_idx].agents_load_item.append(a_agent)

        else:
            # If there is no item to collect just move A agent
            (new_position_x, new_position_y) = a_agent.new_position_with_given_action(sim.dim_h,sim.dim_w
                                                                                      , a_agent.next_action)

            if sim.position_is_empty(new_position_x, new_position_y):
                a_agent.position = (new_position_x, new_position_y)
            else:
                a_agent.change_direction_with_action(a_agent.next_action)

        sim.agents[a_agent_index] = a_agent
        sim.update_the_map()
        return reward #, loaded_item

    ################################################################################################################
    def update_belief_state(self,main_sim,search_tree):
        # print '********* Updating search tree root **********'
        # 1. Taking the real action and the real observation
        action = main_sim.main_agent.history[-1]
        observation = main_sim.take_m_observation()

        # 2. Walking on the tree
        # a. root --- go to ---> action node
        for action_child in search_tree.childNodes:
            if action == action_child.action:
                search_tree = action_child
                break

        # b. action node --- go to ---> observation node
        # print 'o',observation
        for obs_child in search_tree.childNodes:
            #print 'c',obs_child.observation
            if self.observation_is_equal(observation,obs_child.observation):
                search_tree = obs_child
                break

        # c. sampling the new particles
        # print 'n',search_tree.observation

        # print '********* Updating the belief state **********'
        non_child = False
        if search_tree.observation == None:
            non_child = True
            state = State(search_tree.state.simulator.copy())
            enemy = search_tree.enemy
            history = copy(search_tree.history)

            search_tree.destroy()

            search_tree = PONode(parent=None, depth=0, state=state , enemy=enemy)
            search_tree.observation = observation
            search_tree.history = history

        self.black_box(action,observation,main_sim.main_agent.previous_state,non_child)

        # print '********* Updating the tree depth **********'
        search_tree.update_depth(0)

        # print 'new root:',search_tree,'depth:',search_tree.depth,'history:',search_tree.history

        return search_tree

    ################################################################################################################
    def black_box(self,action,real_obs,prev_sim,non_child):
        # 1. Copying and cleaning the current belief states
        cur_belief = list()

        for state in self.belief_state:
            (next_state, observation, reward) = self.simulate_action(state, action)
            if self.observation_is_equal(observation,real_obs):
                cur_belief.append(state)

        self.belief_state = list()

        # 2. Sampling new particles while dont get k particles
        sample_counter = 0
        if not non_child and len(cur_belief) > 1: 
            while len(self.belief_state) != self.k and sample_counter < self.max_samples:
                sample_counter += 1

                state = sample(cur_belief,1)[0]
                (next_state, observation, reward) = self.simulate_action(state, action)
                if self.observation_is_equal(real_obs,observation):
                    self.belief_state.append(next_state)
        else:
            print_flag = True
            while len(self.belief_state) != self.k and sample_counter < self.max_samples:
                sample_counter += 1

                if print_flag:
                    #print 'sampled states:',len(self.belief_state),'/',self.k
                    print_flag = False

                tmp_sim = prev_sim.copy()

                if len(cur_belief) > 0:
                    state = sample(cur_belief,1)[0]
                else:
                    state = State(tmp_sim.uniform_sim_sample())

                (next_state, observation, reward) = self.simulate_action(state, action)
                if self.observation_is_equal(real_obs,observation):
                    self.belief_state.append(next_state)
                    if len(self.belief_state) % 10 == 0:
                        print_flag = True

    ################################################################################################################
    def observation_is_equal(self,observation,other_observation):
        if len(observation) != len(other_observation):
            return False

        for o in range(len(observation)):
            if observation[o] not in other_observation:
                return False

        return True
