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

from simulatorCommumMethods import *

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

			tmp_agents[i] = evaluate_a_agent_action(tmp_agents[i],tmp_sim)

		# 4. Acting and taking the observation and reward
		m_reward = do_m_agent_move(tmp_sim, action)
		tmp_sim.refresh_visible_agents_items()
		observation = (tmp_sim.main_agent.visible_agents,tmp_sim.main_agent.visible_items)

		# 5. Updating the A agents
		a_reward = update_all_A_agents(sim)

		if do_collaboration(sim):
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
				sitm.position = pos

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
				if distance(self.main_agent.position,item.get_position()) < radius:
					if -angle/2 <= angle_of_gradient(self.main_agent.position, self.main_agent.direction, item.get_position()) <= angle/2:
						self.main_agent.visible_items.append(item)

		# 4. Searching for agents
		for i in range(0, len(self.agents)):
			if distance(self.main_agent.position,self.agents[i].position) < radius:
				if -angle/2 <= angle_of_gradient(self.main_agent.position, self.main_agent.direction, self.agents[i].position) <= angle/2:
					self.main_agent.visible_agents.append(self.agents[i])

	def refresh_outvision_positions(self):
		# 1. Calculating the radius and the angle
		radius = self.main_agent.co_radius * self.main_agent.radius
		angle = self.main_agent.angle * self.main_agent.co_angle

		# 2. Taking the invisible positions
		self.outvision_positions = list()
		for s in self.the_map.states:
			if distance(self.main_agent.position,s) > radius:
				self.outvision_positions.append(s)
			elif not(-angle/2 <= angle_of_gradient(self.main_agent.position,self.main_agent.direction,s) <= angle/2):
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