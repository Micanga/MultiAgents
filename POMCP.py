import POUCT
import posimulator
from copy import deepcopy
from math import sqrt, log
from numpy.random import choice
from random import sample

class POMCP:

	def __init__(self,poagent, search_tree ,max_iteration = 1000, max_depth = 10):
		self.pouct = POUCT.POUCT(poagent,search_tree)
		self.poagent = poagent
		self.pomdp = poagent.pomdp
		self.posimulation = None
		self.sample_belief = set()
		self.max_iteration = max_iteration
		self.max_depth = max_depth

	def m_poagent_planning(self,posim):
		# . Copying the current simulation
		self.posimulation = deepcopy(posim)

		# . Running the POMCP planning
		next_move, next_root = self.po_monte_carlo_planning(posimulation)

		return next_move, next_root


	def po_monte_carlo_planning(self,posim):
		# 1. planning
		next_action,next_node = self.search(self.poagent.history,posim)
		return next_action, next_node
		 

	def search(self,history,posim):
		print history
		it = 0	
		while it < self.max_iteration:
			# 1. Sampling a state
			if(history == []):
				state = sample(self.pomdp.states,1)[0]
			else:
				state = sample(self.sample_belief,1)[0]

			# 2. Simulating
			self.simulate(state,history,len(history),posim)

			# 3. Incrementing the iteration counter
			it = it + 1

		# 4. Returning the best action/node
		cur_node = None
		best_node, best_action = POUCT.Node(1,-99999,0,dict(),[],0,None),'L'
		for a in posim.possible_actions_given_state(state):
			hb = deepcopy(self.poagent.history)
			hb.append(a)
			cur_node = self.pouct.search_history(hb)
			if cur_node.value > best_node.value and cur_node != self.pouct.search_tree.root:
				best_node = cur_node
				best_action = a

		return best_action,best_node

  	def evaluate_simulate_actions(self,state,found_node,posim):
  		for a in posim.possible_actions_given_state(state):
  			new_h = deepcopy(found_node.history)
  			new_h.append(a)
  			found_node.add_child(1,0,0,dict(),new_h)

  	def evaluate_simulate_observation(self,observation,choosen_child):
  		new_h = deepcopy(choosen_child.history)
  		new_h.append(observation)
  		choosen_child.add_child(1,0,0,dict(),new_h)

	def simulate(self,state,history,depth,posim):
		# 1. depth verification
		if depth > self.max_depth:
			return 0

		if self.pomdp.state_is_terminal(state):
			return 0

		# 2. if the history doesn't exist in the search tree:
		# add the history/node and bring the answer by rollout
		found_node = self.pouct.search_history(history)
		if(history != found_node.history or found_node.child_nodes == []):
			# a. adding the new possible nodes and rolling out to
			# calculates the your value
			if(found_node.child_nodes == []):
				self.evaluate_simulate_actions(state,found_node,posim)
			return self.rollout(state,history,depth,posim)

		# 3. else we update our node
		# a. taking the best action : Q-function
		c, gamma = 1, self.pomdp.gamma

		# a <- argmax V(hb) + c sqrt( log(N(h)) / N(hb))
		actions_value = [(child.value + 1*sqrt(log(found_node.visits))/child.visits,child.history[len(child.history)-1],child) for child in found_node.child_nodes]
		action = max(actions_value,key=lambda item:item[0])[1]
		choosen_child = max(actions_value,key=lambda item:item[0])[2]

		# b. updating the agent belief states
		# (s',o,r) ~ G(s,a)
		new_sim = deepcopy(posim)
		new_state,observation,reward = new_sim.run(state,action)

		# c. calculating the reward
		# R <- r + gamma*SIMULATE(s',hao,depth+1)
  		new_history = deepcopy(choosen_child.history)
  		new_history.append(observation)
  		found_node = self.pouct.search_history(new_history)
		if(new_history != found_node.history):
			self.evaluate_simulate_observation(observation,choosen_child)

		R = reward + gamma*self.simulate(new_state,new_history,depth+2,new_sim)

		# d. updating infos
		self.sample_belief.add(state)
		found_node.visits = found_node.visits + 1
		choosen_child.visits = choosen_child.visits + 1
		choosen_child.value = choosen_child.value + (R - choosen_child.value)/choosen_child.visits

		return R
		

	def rollout(self,state,history,depth,posim):
		if(depth > self.max_depth):
			return 0

		if self.pomdp.state_is_terminal(state):
			return 0

		# 1. Choosing the action
		# a ~ pi(h)
		action = posim.m_random_action_given_state(state)

		# 2. Simulating the particle
		# (s',o,r) ~ G(s,a)
		new_sim = deepcopy(posim)
		new_state, observation, reward = new_sim.run(state,action)

		# 3. Building the new history
		new_history = deepcopy(history)
		new_history.append(action)
		new_history.append(observation)

		# 4. Calculating the reward
		return reward + self.rollout(new_state,new_history,depth+2,new_sim)
	
	def show(self):
		for x in range(self.pomdp.map_dimension[0]):
			for y in range(self.pomdp.map_dimension[1]):
				if self.poagent.position != (x,y):
					print '|\t \t|',
				else:
					print '|\tM\t|',
			print '\n'