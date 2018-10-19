# Types for agents are 'L1','L2','F1','F2'
import agent
import poagent
import item
import obstacle
import position
import a_star
import simulator

from copy import deepcopy
from math import sqrt
import operator

import numpy as np
from random import randrange
from numpy.random import choice
from collections import defaultdict


##########################################################
#   SENSOR            							##########
##########################################################
def angle_of_gradient(eye_pos, direction, obj_pos):
	xt = obj_pos[0] - eye_pos[0]
	yt = obj_pos[1] - eye_pos[1]

	x = np.cos(direction)*xt + np.sin(direction)*yt
	y = -np.sin(direction)*xt + np.cos(direction)*yt

	return np.arctan2(y, x)

def distance(a, b):
	x_dist = (b[0] - a[0])
	y_dist = (b[1] - a[1])
	return sqrt( (x_dist**2) + (y_dist**2))

def do_collaboration(sim):
    c_reward = 0

    for item in sim.items:
        agents_total_level = 0
        for agent in item.agents_load_item:
            agents_total_level += agent.level
        if agents_total_level >= item.level and item.agents_load_item !=[]:
            item.loaded = True
            for agent in item.agents_load_item:
                sim.agents[agent.index].reset_memory()
            item.agents_load_item = list()
            c_reward += 1

    update_the_map(sim)

    return

##########################################################
#   MOVE M AGENT								##########
##########################################################
def do_m_agent_move(sim, move):
	# 1. Initializing the move sim params and
    # updating the main_agent hostory
    get_reward = 0
    sim.main_agent.history.append(move)

    # 2. If action is load
    if move == 'L':
    	#  a. verify if is possible to load and count the reward
        exist_item, (item_position_x, item_position_y) = is_agent_face_to_item(sim.main_agent,sim)
        if exist_item:
            destination_item_index = find_item_by_location(item_position_x, item_position_y, sim)
            if sim.items[destination_item_index].level <= sim.main_agent.level:
                sim.main_agent = load_item(sim,sim.main_agent, destination_item_index)
                loaded_item = sim.items[destination_item_index]
                sim.items[destination_item_index].loaded = True
                get_reward += float(1.0)
            else:
                sim.items[destination_item_index].agents_load_item.append(sim.main_agent)
	# 3. Else move
    else:
        (x_new, y_new) = sim.main_agent.new_position_with_given_action(move)

        # If there new position is empty
        if position_is_empty(x_new, y_new, sim):
            sim.main_agent.next_action = move
            sim.main_agent.change_position_direction(sim.dim_w, sim.dim_h)

        else:
            sim.main_agent.change_direction_with_action(move)

    # 4. Updating the map
    update_the_map(sim)

    # 5. Getting the new observation
    sim.refresh_visible_agents_items()
    observation = (len(sim.main_agent.visible_agents),len(sim.main_agent.visible_items))
    sim.main_agent.history.append(observation)

    return get_reward

##########################################################
#   MOVE A AGENT								##########
##########################################################
def evaluate_a_agent_action(a_agent, sim):
	# 1. Initializing the move parameters
	location = a_agent.position  # Location of main agent
	destination = position.position(-1, -1)
	target = position.position(-1, -1)

	# 2. Verifying if the agent destination is valid (task avaible)
	if destination_loaded_by_other_agents(a_agent,sim):  # item is loaded by other agents so reset the memory to choose new target.
		a_agent.reset_memory()

	# 3. If the taks continues avaible, keep on it
	# If the target is selected before we have it in memory variable and we can use it
	if a_agent.memory.get_position() != (-1, -1) and location != a_agent.memory: #here
		destination = a_agent.memory

	# 4. If there is no target we should choose a target based on visible items and agents.
	else:  
		# a. updating the visible items and agents
		a_agent.visible_agents_items(sim.items, sim.agents)

		# b. choosing a target based on current visibility
		target = a_agent.choose_target(sim.items, sim.agents)
		a_agent.choose_target_state = deepcopy(sim)

		# c. if the position is valid so we found a target
		if target.get_position() != (-1, -1):
			destination = target
		a_agent.memory = destination

	# 5. If there is no destination the probabilities for all of the actions are same.
	if destination.get_position() == (-1, -1):
		a_agent.set_actions_probability(0.2, 0.2, 0.2, 0.2, 0.2)
		a_agent.set_random_action()
		return a_agent

	# 6. Else we need to define the best policy
	else:
		# a. getting the destination coordinates and the task index
		(x_destination, y_destination) = destination.get_position()  # Get the target position
		destination_index = find_item_by_location(x_destination, y_destination,sim)

		# b. update map with target position
		sim.the_map.problem_map[y_destination][x_destination] = 4

		# c. verifies if the agent can catch the reward;complete the task
		load = a_agent.is_agent_near_destination(x_destination, y_destination)

		# d. if there is a an item nearby loading process starts
		if load:
			a_agent.item_to_load = sim.items[destination_index]

			a_agent.set_actions_probabilities('L')

		# e. else we use the A* pathfinder algorithm to continue
		else:
			# i. initializing the tree
			a = a_star.a_star(sim, a_agent)  # Find the whole path  to reach the destination with A Star

			# ii. find the route
			(x_agent, y_agent) = a_agent.get_position()  # Get agent's current position
			route = a.pathFind(x_agent, y_agent, x_destination, y_destination)

			# iii. if exist route, move in the policy
			#if len(route) > 1:
			#	self.mark_route_map(route,x_agent, y_agent)
			a_agent.route_actions = convert_route_to_action(route)

			# iv. if does not exist route, move randomly
			if len(route) == 0:
				a_agent.set_actions_probability(0.2, 0.2, 0.2, 0.2, 0.2)
				a_agent.set_random_action()
				return a_agent

			# vi. lets rock
			action = get_first_action(route)  # Get first action of the path
			a_agent.set_actions_probabilities(action)

		return a_agent

def update_all_A_agents(sim):
    reward = 0

    for i in range(len(sim.agents)):
        # print self.agents[i].get_actions_probabilities()
        actions = sim.agents[i].actions
        next_action = choice(actions, p=sim.agents[i].get_actions_probabilities())  # random sampling the action

        sim.agents[i].next_action = next_action
        
        reward += update(sim,i)

    return reward

def update(sim, a_agent_index):
    reward = 0
    loaded_item = None
    a_agent = sim.agents[a_agent_index]

    if a_agent.next_action == 'L' and a_agent.item_to_load != -1:
        # print 'loading :', a_agent.item_to_load.get_position()
        destination = a_agent.item_to_load

        if destination.level <= a_agent.level:  # If there is a an item nearby loading process starts

            # load item and and remove it from map  and get the direction of agent when reaching the item.
            a_agent = load_item(sim,a_agent, destination.index)
            loaded_item = sim.items[destination.index]
            reward += 1
        else:
            if not sim.items[destination.index].is_agent_in_loaded_list(a_agent):
                sim.items[destination.index].agents_load_item.append(a_agent)

    else:
        # If there is no item to collect just move A agent
        (new_position_x, new_position_y) = a_agent.new_position_with_given_action(sim.dim_h,sim.dim_w
                                                                                  , a_agent.next_action)

        if position_is_empty(new_position_x, new_position_y,sim):
            a_agent.position = (new_position_x, new_position_y)
        else:
            a_agent.change_direction_with_action(a_agent.next_action)

    sim.agents[a_agent_index] = a_agent
    update_the_map(sim)
    return reward #, loaded_item

##########################################################
#   ITEM SIMULATION								##########
##########################################################
def destination_loaded_by_other_agents(agent,sim):
	# Check if item is collected by other agents so we need to ignore it and change the target.

	(memory_x, memory_y) = agent.get_memory()
	destination_index = find_item_by_location(memory_x, memory_y,sim)

	item_loaded = False

	if destination_index != -1:
		item_loaded = sim.items[destination_index].loaded

	return item_loaded

def find_item_by_location(x, y, sim):
	for i in range(len(sim.items)):
		(item_x, item_y) = sim.items[i].get_position()
		if item_x == x and item_y == y:
			return i
	return -1

def load_item(sim, agent, destination_item_index):

    sim.items[destination_item_index].loaded = True
    (agent_x, agent_y) = agent.get_position()
    sim.items[destination_item_index].remove_agent(agent_x, agent_y)
    agent.last_loaded_item = deepcopy(agent.item_to_load)
    agent.item_to_load = -1
    if(agent != sim.main_agent):
        agent.reset_memory()

    return agent

def is_agent_face_to_item(agent,sim):
        dx = [-1, 0, 1,  0]  # 0:W ,  1:N , 2:E  3:S
        dy = [ 0, 1, 0, -1]

        x_diff = 0
        y_diff = 0

        x, y = agent.get_position()

        if agent.direction == 2 * np.pi / 2:
            # Agent face to West
            x_diff = dx[0]
            y_diff = dy[0]

        if agent.direction == np.pi / 2:
            # Agent face to North
            x_diff = dx[1]
            y_diff = dy[1]

        if agent.direction == 0 * np.pi / 2:
            # Agent face to East
            x_diff = dx[2]
            y_diff = dy[2]

        if agent.direction == 3 * np.pi / 2:
            # Agent face to South
            x_diff = dx[3]
            y_diff = dy[3]

        if 0 <= x + x_diff < sim.dim_w and 0 <= y + y_diff < sim.dim_h and \
                is_there_item_in_position(x + x_diff, y + y_diff, sim) != -1:

            return True, (x + x_diff, y + y_diff)

        return False,(-1,-1)

def is_there_item_in_position(x, y, sim):

    for i in range(len(sim.items)):
        if not sim.items[i].loaded:
            (item_x, item_y) = sim.items[i].get_position()
            if (item_x, item_y) == (x, y):
                return i

    return -1

def items_left(sim):
    items_count= 0
    for i in range(0,len(sim.items)):

        if not sim.items[i].loaded:

            items_count += 1

    return items_count

##########################################################
#   MAP SIMULATION								##########
##########################################################
def update_the_map(sim):
	# 1. Reseting the map
    sim.create_empty_map()

    # 2. Positioning the items
    for i in range(len(sim.items)):
        (item_x, item_y) = sim.items[i].get_position()
        if sim.items[i].loaded :
            sim.the_map.problem_map[item_y][item_x] = 0
        else:
            sim.the_map.problem_map[item_y][item_x] = 1

    # 3. Positioning the A agents
    for i in range(len(sim.agents)):
        (agent_x, agent_y) = sim.agents[i].get_position()
        sim.the_map.problem_map[agent_y][agent_x] = 8

        (memory_x, memory_y) = sim.agents[i].get_memory()
        if (memory_x, memory_y) != (-1, -1):
            sim.the_map.problem_map[memory_y][memory_x] = 4

    # 4. Positioning the Obstacles
    for i in range(len(sim.obstacles)):
        (obs_x, obs_y) = sim.obstacles[i].get_position()
        sim.the_map[obs_y][obs_x] = 5

    # 5. Positioning the Main Agent
    if sim.main_agent is not None:
        (m_agent_x, m_agent_y) = sim.main_agent.get_position()
        sim.the_map.problem_map[m_agent_y][m_agent_x] = 9

def position_is_empty(x, y, sim):

    for i in range(len(sim.items)):
        (item_x, item_y) = sim.items[i].get_position()
        if (item_x, item_y) == (x,y) and not sim.items[i].loaded:
            return False

    for i in range(len(sim.agents)):
        (agent_x, agent_y) = sim.agents[i].get_position()
        if (agent_x, agent_y) == (x, y):
            return False

    if sim.main_agent is not None:
        (m_agent_x, m_agent_y) =sim.main_agent.get_position()
        if (m_agent_x, m_agent_y) == (x, y):
            return False

    return True

def convert_route_to_action(route):
	#  This function is to find the first action afte finding the path by  A Star
	actions = []

	for dir in route:

		if dir == '0':
			actions.append('W')
		if dir == '1':
			actions.append('N')
		if dir == '2':
			actions.append('E')
		if dir == '3':
			actions.append('S')
	return actions

def get_first_action(route):
	#  This function is to find the first action afte finding the path by  A Star

	dir = route[0]

	if dir == '0':
		return 'W'
	if dir == '1':
		return 'N'
	if dir == '2':
		return 'E'
	if dir == '3':
		return 'S'