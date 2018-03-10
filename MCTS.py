from math import *
import random
import simulator
import item
import agent
import a_star


class State:

    def __init__(self, sim):
         # At the root pretend the player just moved is p2 - p1 has the first move
        self.sim = sim
        self.move_options = ['N', 'S', 'E', 'W']

    # Move the agent based on move
    def do_move(self, move):

        get_reward = 0

        # get the position of main agent
        tmp_m_agent = self.sim.main_agent
        (xM, yM) = tmp_m_agent.get_position()

        # assign unknown agent
        tmp_a_agent = self.sim.agents[0]

        # Set the new action to the main agent.
        tmp_m_agent.next_action = move

        # Get new action of main agent and set it to the main agent.
        (xA, yA) = tmp_m_agent.change_position_direction(10, 10)

        if tmp_a_agent.get_position() == (xA, yA):
            return -1

        tmp_m_agent.position = (xA, yA)

        # If the new position of the main agent is the position of an item then do loading item.
        if self.sim.the_map[yA][xA] == 1 or self.sim.the_map[yA][xA] == 4:

            # Find the index and position of item that should be loaded.
            loaded_item_index = self.sim.get_item_by_position(xA, yA)
            (xI, yI) = self.sim.items[loaded_item_index].get_position()

            # Check if the other agent is need help

            (x, y) = self.sim.memory.get_position()
            a_destinantion_item_index = self.sim.get_item_by_position(x, y)
            destination_agent_level = self.sim.items[a_destinantion_item_index].level
            agent_need_help = tmp_a_agent.is_agent_near_destination(x,y) and tmp_a_agent.level < destination_agent_level

            if tmp_m_agent.level >= self.sim.items[loaded_item_index].level:

                # Update the simulator and load the item.
                self.sim.load_item(tmp_m_agent, loaded_item_index)

                # if agent_need_help and a_destinantion_item_index == loaded_item_index:
                #
                #     get_reward += 100
                # else:
                get_reward += 1
            else:

                # If unknown agent is in the loading position of the same item that main agent wants to collect.
                a_load = tmp_a_agent.is_agent_near_destination(xI, yI) and tmp_a_agent.next_action == 'L'

                # Check if two agents can load the item together
                if a_load and tmp_m_agent.level + tmp_a_agent.level >= self.sim.items[loaded_item_index].level:
                    self.sim.load_item(tmp_m_agent, loaded_item_index)

                    # if agent_need_help and a_destinantion_item_index == loaded_item_index:
                    #
                    #     get_reward += 100
                    # else:
                    get_reward += 1

                    # a_agent movement
                    new_position = (xI, yI)
                    self.sim.memory = (0, 0)
                    self.sim.update_map(tmp_a_agent.position, new_position)

        self.sim.main_agent = tmp_m_agent

        # Update the map
        self.sim.update_map_mcts((xM, yM), (xA, yA))
       # print get_reward
        return get_reward

    def get_moves(self):
        return self.move_options


################################################################################################################
class Node:

    def __init__(self, tree_level, position, move=None, parent=None, numItems = 0 ):

        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.tree_level = tree_level
        self.position = position
        self.childNodes = []
        self.cumulativeRewards = 0
        self.immediateReward = 0
        self.visits = 0
        self.expectedReward = 0
        self.numItems = numItems
        self.untriedMoves = self.create_possible_moves()

    def uct_select_child(self):

        ## UCB expects mean between 0 and 1.
        s = sorted(self.childNodes, key=lambda c: c.expectedReward/self.numItems + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, m, child_position):

        n = Node( position=child_position, move=m, parent=self, tree_level=self.tree_level + 1, numItems=self.numItems)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)

        return n

    def create_possible_moves(self):

        (x,y) = self.position
        m =10
        untriedMoves = ['N', 'S', 'E', 'W']

        if x == 0:
            untriedMoves.remove('E')

        if y == 0:
            untriedMoves.remove('S')

        if x == m-1:
            untriedMoves.remove('W')

        if y == m-1:
            untriedMoves.remove('N')

        return untriedMoves

    def update(self, result):

        self.visits += 1
        self.cumulativeRewards += result + self.immediateReward
        self.expectedReward = self.cumulativeRewards/self.visits
        return self.expectedReward * 0.95  # discount factor = 0.95


def create_temp_simulator(items, agents, main_agent):

    local_map = []
    row = [0] * 10

    for i in range(10):
        local_map.append(list(row))

    local_items = []
    for i in range(len(items)):
        (item_x, item_y) = items[i].get_position()
        local_item = item.item(item_x, item_y, items[i].level, i)
        local_item.loaded = items[i].loaded
        local_items.append(local_item)
        if not local_item.loaded:
            local_map[item_y][item_x] = 1

    local_agents = list()

    (a_agent_x, a_agent_y) = agents[0].get_position()
    local_map[a_agent_y][a_agent_x] = 8
    local_agent = agent.Agent(a_agent_x, a_agent_y, 'l1', 0)
    local_agents.append(local_agent)

    (m_agent_x, m_agent_y) = main_agent.get_position()
    local_map[m_agent_y][m_agent_x] = 9
    local_main_agent = agent.Agent(m_agent_x, m_agent_y, 'l1', 1)

    tmp_sim = simulator.simulator(local_map, local_items, local_agents, local_main_agent, 10, 10)
    return tmp_sim


def monte_carlo_tree_search(local_sim, iteration_max, parameters_estimation):
    print("***************monte_carlo_tree_search***************")
    #print parameters_estimation
    num_items = local_sim.items_left()

    node_position = local_sim.main_agent.get_position()
    root_node = Node(tree_level=0,position=node_position, numItems=num_items)
    node = root_node
    
    for i in range(iteration_max):

        # Start to move from the current state and update tree rewards
        tmp_sim = create_temp_simulator(local_sim.items, local_sim.agents, local_sim.main_agent)
        tmp_state = State(tmp_sim)

        # Trace the tree by selecting the child node with utc algorithm, till reaching the leaf
        # node is fully expanded and non-terminal
        # if we try all possible moves and current node has a child then select a node to expand
        # We will move till reaching a leaf which don't have any child and we don't
        while node.untriedMoves == [] and node.childNodes != [] and node.numItems > 0:
            # Use UCB to determine the best action, then move the simulation into that state
            node = node.uct_select_child()
            tmp_state.do_move(node.move)

        # Expand the tree and select one of the untried moves and  if we can expand (i.e. state/node is non-terminal)
        if node.untriedMoves != [] and node.numItems > 0:
            m = random.choice(node.untriedMoves)

            # Move the agent with the selected random move and update the rewards.
            get_reward = tmp_state.do_move(m)
            new_position = tmp_state.sim.main_agent.get_position()

            # Add child with selected random move and descend tree
            node = node.add_child(m, new_position)

            node.immediateReward = get_reward

            node.numItems = tmp_sim.items_left()

            # Nothing to do if the scenario is cleared, we created a terminal node
            if node.numItems == 0:
                continue

        node_reward = 0

        # Roll out - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        if node.numItems > 0:

            roll_out_max = 100
            roll_out_count = 0

            roll_out_sim = create_temp_simulator(tmp_state.sim.items, tmp_state.sim.agents,tmp_state.sim.main_agent )
            roll_out_state = State(roll_out_sim)

            # while state is non-terminal
            while roll_out_count < roll_out_max:
               # roll_out_state.sim.draw_map_with_level()
                move = random.choice(roll_out_state.get_moves())
                get_reward = roll_out_state.do_move(move)
                if get_reward == -1:
                    continue

                if get_reward:
                    node_reward += get_reward * (0.95 ** roll_out_count)

                roll_out_state.sim.agents[0].set_parameters(parameters_estimation[0], parameters_estimation[1], parameters_estimation[2])
                unknown_agent = roll_out_state.sim.run_and_update(roll_out_state.sim.agents[0])
                # If next action is load and the item is loaded as well.
                if unknown_agent.next_action == 'L' and roll_out_state.sim.memory.get_position() == (0, 0):
                    node_reward += 1 * (0.95 ** roll_out_count)


                #print "reward:" , node_reward
                roll_out_count += 1

        # TO CHECK: Are we handling the "no items" case correctly?

        # Back propagate from the expanded node and work back to the root node
        while 1 == 1:
            node_reward = node.update(node_reward)
    
            node = node.parentNode

            # it is root node and iteration should stop here and just update the root node
            if node.tree_level == 0:
                node.update(node_reward)
                break

    # for n in root_node.childNodes:
    #     print n.move
    #     print n.expectedReward/10
    #     print n.cumulativeRewards
    #     print n.visits

    # return the move that was most visited
    return sorted(root_node.childNodes, key=lambda c: c.visits)[-1].move


################################################################################################################
def move_agent(agents, items, main_agent, parameters):

    real_sim = create_temp_simulator(items, agents, main_agent)
    
    next_move = monte_carlo_tree_search(real_sim, iteration_max=100, parameters_estimation=parameters)

    return next_move

