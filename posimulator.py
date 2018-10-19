# Types for agents are 'L1','L2','F1','F2'
import agent
import POMDP
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

import simulatorCommonMethods

class POSimulator:

	def __init__(self,pomdp = None,items = [],agents = [], main_agent = None):
		# 1. Getting map information
		self.the_map = pomdp
		if self.the_map != None:
			self.dim_w = pomdp.map_dimension[0]
			self.dim_h = pomdp.map_dimension[1]
		else:
			self.dim_w = 0
			self.dim_h = 0

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
		self.sampled_agents = []
		self.sampled_item = []

		# 8. Flag variables
		self.do_estimation = True

	##########################################################
	#   SAMPLE SIMULATION METHODS					##########
	##########################################################
	def run(self,state,action,agents_parameter_estimation):
		# 1. Getting the visible items and A agents and drawing
		# in the current map
		self.refresh_visible_agents_items()
		self.refresh_outvision_positions()

		# 2. Sampling the invisible Items and A agents positions
		self.sampled_items = self.perform_item_sample()
		self.sampled_agents = self.perform_agent_sample()

		# 3. Creating a temporary simulation with sampled information
		tmp_agents = []
		for agn in self.main_agent.visible_agents:
			tmp_agents.append(agn)
		for agn in self.sampled_agents:
			tmp_agents.append(agn)

		tmp_items = []
		for itm in self.main_agent.visible_items:
			tmp_items.append(itm)
		for itm in self.sampled_items:
			tmp_items.append(itm)

		tmp_sim = deepcopy(self)
		tmp_sim.agents = tmp_agents
		tmp_sim.items = tmp_items
		tmp_sim.main_agent.position = state

		# 4. Moving the A agents of simulation
		for i in range(len(tmp_agents)):
			if self.do_estimation:
				estimated_parameter = agents_parameter_estimation[i]
				selected_type = estimated_parameter.get_sampled_probability()
				agents_estimated_values = estimated_parameter.get_parameters_for_selected_type(selected_type)
				tmp_agents[i].set_parameters(tmp_sim, agents_estimated_values.level, agents_estimated_values.radius,
											 agents_estimated_values.angle)

			tmp_agents[i] = simulatorCommonMethods.evaluate_a_agent_action(tmp_agents[i],tmp_sim)

		# 4. Acting and taking the observation and reward
		m_reward = simulatorCommonMethods.do_m_agent_move(tmp_sim, action)
		new_state = tmp_sim.main_agent.position
		tmp_sim.refresh_visible_agents_items()
		observation = (len(tmp_sim.main_agent.visible_agents),len(tmp_sim.main_agent.visible_items))

		# 5. Updating the A agents
		a_reward = simulatorCommonMethods.update_all_A_agents(tmp_sim)

		if simulatorCommonMethods.do_collaboration(tmp_sim):
			c_reward = float(1)
		else:
			c_reward = 0

		total_reward = float(m_reward + a_reward + c_reward) / len(tmp_sim.items)

		return new_state, observation, total_reward

	def perform_agent_sample(self):
		sampled_agents = []
		for agn in self.agents:
			if agn not in self.main_agent.visible_agents:
				# samplig a position to invisible agent
				idx = randrange(len(self.outvision_positions))
				pos = self.outvision_positions[idx]

				# removing the position from possibilities
				self.outvision_positions.remove(pos)

				# copying and updating the reffered item
				sagn = deepcopy(agn)
				sagn.position = pos

				# adding to the sampled list
				sampled_agents.append(sagn)
		return sampled_agents

	def perform_item_sample(self):
		sampled_items = []
		for itm in self.items:
			if itm not in self.main_agent.visible_items:
				# samplig a position to invisible item
				idx = randrange(len(self.outvision_positions))
				pos = self.outvision_positions[idx]

				# removing the position from possibilities
				self.outvision_positions.remove(pos)

				# copying and updating the reffered item
				sitm = deepcopy(itm)
				sitm.position = position.position(pos[0],pos[1])

				# adding to the sampled list
				sampled_items.append(sitm)
		return sampled_items

	##########################################################
	#   CREATION METHODS  							##########
	##########################################################
	def create_empty_map(self):

		self.the_map.problem_map = list()

		row = [0] * self.dim_w

		for i in range(self.dim_h):
			self.the_map.problem_map.append(list(row))

	@staticmethod
	def is_comment(string):
		for pos in range(len(string)):
			if string[pos] == ' ' or string[pos] == '\t':
				continue
			if string[pos] == '#':
				return True
			else:
				return False	

	def loader(self, path):
		"""
		Takes in a csv file and stores the necessary instances for the simulation object. The file path referenced
		should point to a file of a particular format - an example of which can be found in utils.py txt_generator().
		The exact order of the file is unimportant - the three if statements will extract the needed information.
		:param path: File path directory to a .csv file holding the simulation information
		:return:
		"""
		# 1. Load and store csv file
		info = defaultdict(list)
		print path
		with open(path) as info_read:
			for line in info_read:
				if not self.is_comment(line):
					data = line.strip().split(',')
					key, val = data[0], data[1:]
					info[key].append(val)

		# 2. Creating the problem map
		self.dim_w = int(info['grid'][0][0])
		self.dim_h = int(info['grid'][0][1])

		problem_map = list()
		row = [0] * self.dim_w
		for i in range(self.dim_h):
			problem_map.append(list(row))

		self.the_map = POMDP.POMDP(self.dim_w,self.dim_h,problem_map,[],[],0.95)

		# 3. Add items and agents to the environment
		nitems, nagents, nobs = 0, 0, 0
		for k, v in info.items():
			if 'item' in k:
				self.items.append(item.item(v[0][0], v[0][1], v[0][2], nitems))
				nitems += 1
			elif 'agent' in k:
				#import ipdb; ipdb.set_trace()
				agnt = agent.Agent(v[0][0], v[0][1], v[0][2], v[0][3], nagents)
				agnt.set_parameters(self, v[0][4], v[0][5], v[0][6])
				agnt.choose_target_state = deepcopy(self)
				self.agents.append(agnt)
				nagents += 1
			elif 'main' in k:
				# x-coord, y-coord, direction, type, index
													#x        y       dir     type      idx   lv radius angle, pomdp
				self.main_agent = poagent.POAgent(v[0][0], v[0][1], v[0][2], v[0][3], v[0][4], 1, 0.5, np.pi, self.the_map)
			elif 'obstacle' in k:
				self.obstacles.append(obstacle.Obstacle(v[0][0], v[0][1]))
				nobs += 1

		# 4. Run Checks
		assert len(self.items) == nitems, 'Incorrect Item Loading'
		assert len(self.agents) == nagents, 'Incorrect Ancillary Agent Loading'
		assert len(self.obstacles) == nobs, 'Incorrect Obstacle Loading'

		# 5. Print Simulation Description and updating the map
		print('Grid Size: {} \n{} Items Loaded\n{} Agents Loaded\n{} Obstacles Loaded'.format(self.dim_w,len(self.items),len(self.agents),len(self.obstacles)))
		simulatorCommonMethods.update_the_map(self)
		self.main_agent.pomdp = self.the_map

	##########################################################
	#   ACTION EVALUATE METHODS						##########
	##########################################################
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
						exist_item, (item_position_x, item_position_y) = simulatorCommonMethods.is_agent_face_to_item(self.main_agent,self)
						if exist_item :
							possible_actions.append(a)
		return possible_actions

	def is_item_nearby(self):
		pos = self.main_agent.position
		for itm in self.items:
			if not itm.loaded:
				(xI, yI) = itm.position.get_position()
				if (yI == pos[1] and abs(pos[0] - xI) == 1) or (xI == pos[0] and abs(pos[1] - yI) == 1):
					return True
		return False

	def m_random_action_given_state(self,state):
		possible_actions = self.possible_actions_given_state(state)
		return choice(possible_actions)

	def move_a_agent(self,a_agent):
		# 1. Initializing the move parameters
		location = a_agent.position  # Location of main agent
		destination = position.position(-1, -1)
		target = position.position(-1, -1)

		# 2. Verifying if the agent destination is valid (task avaible)
		if simulatorCommonMethods.destination_loaded_by_other_agents(a_agent,self):  # item is loaded by other agents so reset the memory to choose new target.
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
			destination_index = simulatorCommonMethods.find_item_by_location(x_destination, y_destination,self)

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
				a_agent.route_actions = simulatorCommonMethods.convert_route_to_action(route)

				# iv. if does not exist route, move randomly
				if len(route) == 0:
					a_agent.set_actions_probability(0.2, 0.2, 0.2, 0.2, 0.2)
					a_agent.set_random_action()
					return a_agent

				# vi. lets rock
				action = simulatorCommonMethods.get_first_action(route)  # Get first action of the path
				a_agent.set_actions_probabilities(action)

			return a_agent

	##########################################################
	#   VISIBILITY METHODS							##########
	##########################################################
	def refresh_visible_agents_items(self):
		# 1. Clearing the visible lists
		self.main_agent.visible_agents = list()
		self.main_agent.visible_items = list()

		# 2. Calculating the radius and the angle
		radius = self.main_agent.co_radius * self.main_agent.radius
		angle = self.main_agent.angle * self.main_agent.co_angle

		# 3. Searching for items
		for item in self.items:
			if not item.loaded:
				if simulatorCommonMethods.distance(self.main_agent.position,item.get_position()) < radius:
					if -angle/2 <= simulatorCommonMethods.angle_of_gradient(self.main_agent.position, self.main_agent.direction, item.get_position()) <= angle/2:
						self.main_agent.visible_items.append(item)

		# 4. Searching for agents
		for i in range(0, len(self.agents)):
			if simulatorCommonMethods.distance(self.main_agent.position,self.agents[i].position) < radius:
				if -angle/2 <= simulatorCommonMethods.angle_of_gradient(self.main_agent.position, self.main_agent.direction, self.agents[i].position) <= angle/2:
					self.main_agent.visible_agents.append(self.agents[i])

	def refresh_outvision_positions(self):
		# 1. Calculating the radius and the angle
		radius = self.main_agent.co_radius * self.main_agent.radius
		angle = self.main_agent.angle * self.main_agent.co_angle

		# 2. Taking the invisible positions
		self.outvision_positions = list()
		for s in self.the_map.states:
			if simulatorCommonMethods.distance(self.main_agent.position,s) > radius:
				self.outvision_positions.append(s)
			elif not(-angle/2 <= simulatorCommonMethods.angle_of_gradient(self.main_agent.position,self.main_agent.direction,s) <= angle/2):
				self.outvision_positions.append(s)

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

	def draw_map(self):

		for y in range(self.dim_h - 1, -1, -1):
			for x in range(self.dim_w):
				xy = self.the_map.problem_map[y][x]
				if xy == 0:
					print '.',  # space
				elif xy == 1:
					print 'I',  # Items
				elif xy == 2:
					print 'S',  # start
				elif xy == 3:
					print 'R',  # route
				elif xy == 4:
					print 'D',  # finish
				elif xy == 5:
					print '+',  # Obstacle
				elif xy == 8:
					print 'A',  # A Agent
				elif xy == 9:
					print 'M',  # Main Agent
			print

	def log_map(self, file):
		line =''
		for y in range(self.dim_h - 1, -1, -1):
			for x in range(self.dim_w):
				xy = self.the_map.problem_map[x][y]
				if xy == 0:
					line = line + '.'  # space
				elif xy == 1:
					line = line + 'I'  # Items
				elif xy == 2:
					line = line + 'S'  # start
				elif xy == 3:
					line = line + 'R'  # route
				elif xy == 4:
					line = line + 'D'  # finish
				elif xy == 5:
					line = line + '+'  # Obstacle
				elif xy == 8:
					line = line + 'A'  # A Agent
				elif xy == 9:
					line = line + 'M'  # Main Agent

			file.write(line+ '\n')
			line = ''
		file.write('*********************\n')
