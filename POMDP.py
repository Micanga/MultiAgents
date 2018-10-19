# -*- coding: utf-8 -*-
import sys
import operator, random

class POMDP:

	global width, height
	width = 0
	height = 1

	def __init__(self,dim_w,dim_h,problem_map,terminals, evidences,gamma):
		# 1. def the map
		self.problem_map =  problem_map # grid with rewards
		self.terminals = terminals		# array with problem terminals
		self.map_dimension = [dim_w,dim_h]# size of the grid

		# 2. init the mdp components
		self.reward = {}
		self.ret = {}
		self.states  = set()
		for x in range(self.map_dimension[width]):
			for y in range(self.map_dimension[height]):
				self.reward[(x, y)] = self.problem_map[x][y]
				self.ret[(x,y)] = self.problem_map[x][y]
				if self.problem_map[x][y] is not None:
					self.states.add((x, y))

		# 3. init the pomdps components
		self.evidences =  evidences 	# grid with all evidences in each state
		self.evidences_list = {}
		for x in range(self.map_dimension[width]):
			for y in range(self.map_dimension[height]):
				if self.evidences != [] and self.evidences[x][y] not in self.evidences_list:
					ne = self.count_evidence(self.evidences[x][y])
					self.evidences_list[self.evidences[x][y]] = (1.0/ne)
		self.gamma = gamma
		self.belief = dict([(s,1.0/float((len(self.states)-len(self.terminals))) if s not in self.terminals else 0) for s in self.states])

	def R(self, state):
		return self.reward[state]

	def O(self,s1):
		return self.evidences_list[self.evidences[s1[width]][s1[height]]]

	def count_evidence(self,e):
		ne = 0
		for x in range(self.map_dimension[width]):
			for y in range(self.map_dimension[height]):
				if self.evidences[x][y] == e:
					ne = ne + 1
		return float(ne)

	def state_is_terminal(self,state):
		if state in self.terminals:
			return True
		return False

	def show(self):
		for x in range(self.map_dimension[width]):
			for y in range(self.map_dimension[height]):
				print self.problem_map[x][y],self.evidences[x][y], '\t',
			print '\n'