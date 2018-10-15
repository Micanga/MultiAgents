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

class POSimulator:

	def __init__(self,pomdp = None,items = None,agents = None, main_agent = None):
		# 1. Getting map information
		self.the_map = pomdp
		self.dim_w = pomdp.map_dimension[0]
		self.dim_h = pomdp.map_dimension[1]

		# 2. Getting items information
		self.items = items

		# 2. Getting obstacles	 information
		self.obstacles = []

		# 4. Getting A agents information
		self.agents = agents

		# 5. Getting M agent information
		self.main_agent = main_agent

		# 6. Initializing the outvision array
		self.outvision_positions = list()

		# 7. Initializing the sample lists
		self.sample_agn = []
		self.sample_item = []
		
	##########################################################
	#   REAL SIMULATION METHODS						##########
	##########################################################
	def real_run(self,action):
		# 1. Moving the A agnets
		for agn in self.agents:
			agn = self.move_a_agent(agn)

		# 2. Acting and taking the observation and reward with M agent
		new_state,direction = self.go(self.main_agent,action)
		self.main_agent.position = new_state
		self.main_agent.direction = direction

		self.visible_agents_items()

		observation = (len(self.main_agent.visible_agents),len(self.main_agent.visible_items))
		reward = self.the_map.problem_map[new_state[0]][new_state[1]]

		# 3. Returning the results
		return new_state, observation, reward

	##########################################################
	#   SAMPLED SIMULATION METHODS					##########
	##########################################################
	def run(self,state,action):
		# 1. Getting the visible items and A agents and drawing
		# in the current map
		self.visible_agents_items()
		self.get_outvision_positions()

		# 2. Sampling the invisible Items and A agents positions
		self.sample_item = []
		for itm in self.items:
			if itm not in self.main_agent.visible_items:
				idx = randrange(len(self.outvision_positions))
				pos = self.outvision_positions[idx]
				self.outvision_positions.remove(pos)
				sitm = deepcopy(itm)
				sitm.position = pos
				self.sample_item.append(sitm)

		self.sample_agn = []
		for agn in self.agents:
			if agn not in self.main_agent.visible_agents:
				idx = randrange(len(self.outvision_positions))
				pos = self.outvision_positions[idx]
				self.outvision_positions.remove(pos)
				sagn = deepcopy(agn)
				sagn.position = pos
				self.sample_agn.append(agn)

		# 3. Moving the sampled agnts
		for agn in self.sample_agn:
			agn = self.move_a_agent(agn)

		# 4. Acting and taking the observation and reward
		new_state,direction = self.go(self.main_agent,action)
		self.visible_agents_items()
		observation = (len(self.main_agent.visible_agents),len(self.main_agent.visible_items))
		reward = self.the_map.problem_map[new_state[0]][new_state[1]]

		return new_state, observation, reward

	def possible_actions_given_state(self,state):
		possible_actions = []
		for a in self.main_agent.actions:
			for (p,s) in self.main_agent.T(state,a):
				if a != 'L':
					if s not in [ag.position for ag in self.agents]:
						if s not in [item.get_position() for item in self.items]:
							possible_actions.append(a)
				else:
					if self.is_item_nearby():
						possible_actions.append(a)
		return possible_actions

	def m_random_action_given_state(self,state):
		possible_actions = self.possible_actions_given_state(state)
		return choice(possible_actions)

	def is_item_nearby(self):
		pos = self.main_agent.position
		for itm in self.items:
			if not itm.loaded:
				(xI, yI) = itm.position.get_position()
				if (yI == pos[1] and abs(pos[0] - xI) == 1) or (xI == pos[0] and abs(pos[1] - yI) == 1):
					return True
		return False


	def go(self,agent,action):
		# 1. Initializing the action results
		action_diff = {'L': (0,0),'W': (-1,0), 'N': (0,1), 'E': (1,0), 'S': (0,-1)}
		action_dirc = {'W': (2*np.pi/2), 'N': (np.pi/2), 'E': (0), 'S': (3*np.pi/2)}

		# 2. Updating the state
		state1 = tuple(map(operator.add, agent.position, action_diff[action]))
		if(state1 in self.the_map.states):
			if(state1 not in [agn.position for agn in self.main_agent.visible_agents]):
				if(state1 not in [item.get_position for item in self.main_agent.visible_items]):
					if(state1 not in [agn.position for agn in self.sample_agn]):
						if(state1 not in [item.get_position for item in self.sample_item]):
							new_position = state1
		else:
			new_position = agent.position

		# 3. Updating the direction
		direction = agent.direction
		if(action != 'L'):
			direction = action_dirc[action]

		# 4. Returning the new position
		return new_position, direction

	##########################################################
	#   VISIBILITY METHODS							##########
	##########################################################
	def visible_agents_items(self):
		# 1. Clearing the visible lists
		self.main_agent.visible_agents = list()
		self.main_agent.visible_items = list()

		# 2. Calculating the radius and the angle
		radius = self.main_agent.co_radius * self.main_agent.radius
		angle = self.main_agent.angle * self.main_agent.co_angle

		# 3. Searching for items
		for item in self.items:
			if not item.loaded:
				if self.distance(self.main_agent.position,item.get_position()) < radius:
					if -angle/2 <= self.angle_of_gradient(self.main_agent.position, self.main_agent.direction, item.get_position()) <= angle/2:
						self.main_agent.visible_items.append(item)

		# 4. Searching for agents
		for i in range(0, len(self.agents)):
			if self.distance(self.main_agent.position,self.agents[i].position) < radius:
				if -angle/2 <= self.angle_of_gradient(self.main_agent.position, self.main_agent.direction, self.agents[i].position) <= angle/2:
					self.main_agent.visible_agents.append(self.agents[i])

	def angle_of_gradient(self,eye_pos, direction, obj_pos):
		xt = obj_pos[0] - eye_pos[0]
		yt = obj_pos[1] - eye_pos[1]

		x = np.cos(direction)*xt + np.sin(direction)*yt
		y = -np.sin(direction)*xt + np.cos(direction)*yt

		return np.arctan2(y, x)

	def distance(self,a, b):
		x_dist = (b[0] - a[0])
		y_dist = (b[1] - a[1])
		return sqrt( (x_dist**2) + (y_dist**2))

	def get_outvision_positions(self):
		# 1. Calculating the radius and the angle
		radius = self.main_agent.co_radius * self.main_agent.radius
		angle = self.main_agent.angle * self.main_agent.co_angle

		# 2. Taking the invisible positions
		self.outvision_positions = list()
		for s in self.the_map.states:
			if self.distance(self.main_agent.position,s) > radius:
				self.outvision_positions.append(s)
			elif not(-angle/2 <= self.angle_of_gradient(self.main_agent.position,self.main_agent.direction,s) <= angle/2):
				self.outvision_positions.append(s)

	##########################################################
	#   A AGENT METHODS   							##########
	##########################################################
	def move_a_agent(self, a_agent):
		# 1. Initializing the move parameters
		location = a_agent.position  # Location of main agent
		destination = position.position(-1, -1)
		target = position.position(-1, -1)

		# 2. Verifying if the agent destination is valid (task avaible)
		if self.destination_loaded_by_other_agents(a_agent):  # item is loaded by other agents so reset the memory to choose new target.
			a_agent.reset_memory()

		# 3. If the taks continues avaible, keep on it
		# If the target is selected before we have it in memory variable and we can use it
		if a_agent.memory.get_position() != (-1, -1) and location != a_agent.memory: #here
			destination = a_agent.memory

		# 4. If there is no target we should choose a target based on visible items and agents.
		else:  
			# a. updating the visible items and agents
			a_agent.visible_agents_items(self.items, self.agents)

			# b. choosing a target based on current visibility
			target = a_agent.choose_target(self.items, self.agents)
			a_agent.choose_target_state = deepcopy(self)

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
			destination_index = self.find_item_by_location(x_destination, y_destination)

			# b. update map with target position
			self.the_map.problem_map[y_destination][x_destination] = 4

			# c. verifies if the agent can catch the reward;complete the task
			load = a_agent.is_agent_near_destination(x_destination, y_destination)

			# d. if there is a an item nearby loading process starts
			if load:
				a_agent.item_to_load = self.items[destination_index]

				a_agent.set_actions_probabilities('L')

			# e. else we use the A* pathfinder algorithm to continue
			else:
				# i. initializing the tree
				a = a_star.a_star(self, a_agent)  # Find the whole path  to reach the destination with A Star

				# ii. find the route
				(x_agent, y_agent) = a_agent.get_position()  # Get agent's current position
				route = a.pathFind(x_agent, y_agent, x_destination, y_destination)

				# iii. if exist route, move in the policy
				#if len(route) > 1:
				#	self.mark_route_map(route,x_agent, y_agent)
				a_agent.route_actions = self.convert_route_to_action(route)

				# iv. if does not exist route, move randomly
				if len(route) == 0:
					a_agent.set_actions_probability(0.2, 0.2, 0.2, 0.2, 0.2)
					a_agent.set_random_action()
					return a_agent

				# vi. lets rock
				action = self.get_first_action(route)  # Get first action of the path
				a_agent.set_actions_probabilities(action)

			return a_agent

	def destination_loaded_by_other_agents(self, agent):
		# Check if item is collected by other agents so we need to ignore it and change the target.

		(memory_x, memory_y) = agent.get_memory()
		destination_index = self.find_item_by_location(memory_x, memory_y)

		item_loaded = False

		if destination_index != -1:
			item_loaded = self.items[destination_index].loaded

		return item_loaded

	def find_item_by_location(self, x, y):
		for i in range(len(self.items)):
			(item_x, item_y) = self.items[i].get_position()
			if item_x == x and item_y == y:
				return i
		return -1

	def convert_route_to_action(self, route):
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

	def get_first_action(self,route):
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

	##########################################################
	#   INTERFACE METHODS 							##########
	##########################################################
	def show(self,state):
		for x in range(self.the_map.map_dimension[0]):
			for y in range(self.the_map.map_dimension[1]):
				if (x,y) == state:
					print 'M', '\t',
				elif (x,y) in [item.get_position() for item in self.items]:
					if item in self.main_agent.visible_items:
						print 'I', '\t',
					else:
						print 'i', '\t',
				elif (x,y) in [ag.position for ag in self.agents]:
					if ag in self.main_agent.visible_agents:
						print 'A', '\t',
					else:
						print 'a', '\t',
				else:
					print '_', '\t',
			print '\n'

	def show(self):
		for x in range(self.the_map.map_dimension[0]):
			for y in range(self.the_map.map_dimension[1]):
				if (x,y) == self.main_agent.position:
					print 'M', '\t',
				elif (x,y) in [item.get_position() for item in self.items]:
					if item in self.main_agent.visible_items:
						print 'I', '\t',
					else:
						print 'i', '\t',
				elif (x,y) in [ag.position for ag in self.agents]:
					if ag in self.main_agent.visible_agents:
						print 'A', '\t',
					else:
						print 'a', '\t',
				else:
					print '_', '\t',
			print '\n'