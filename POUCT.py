from math import sqrt

class POUCT:

	def __init__(self, poagent, search_tree):
		self.poagent = poagent
		if search_tree != None:
			self.search_tree = search_tree
		else:
			self.search_tree = Search_Tree()
			self.search_tree.set_root(Node())

	def search_history(self,history):
		cur_node = self.search_tree.root

		depth = 0
		while cur_node.child_nodes != [] and cur_node.history != history and depth < len(history):
			child = None
			for child in cur_node.child_nodes:
				if child.history[depth] == history[depth]:
					cur_node = child
			depth = depth + 1

		return cur_node

	def print_search_tree(self):
		if(self.search_tree == None):
			print 'None'
		else:
			self.search_tree.show(self.search_tree.root)


class Search_Tree:

	def __init__(self):
		self.root = None

	def set_root(self,root):
		self.root = root

	def show(self,cur_node):
		if(cur_node != None):
			cur_node.show()
		else:
			print 'None'
			return

		for c in cur_node.child_nodes:
			self.show(c)

	def update_depth_from(self,cur_node):
		if(cur_node.depth != 0):
			cur_node.depth = cur_node.parent.depth+1
			
		for c in cur_node.child_nodes:
			self.update_depth_from(c)

		return

	def destroy_from(self,cur_node):
		for c in cur_node.child_nodes:
			self.destroy_from(c)

		cur_node.child_nodes = None
		cur_node.parent = None
		del cur_node

		return

	def change_root(self,new_root):
		depth = 0
		cur_node = self.root

		while depth < len(new_root.history):
			for c in cur_node.child_nodes:
				if c.history[depth] != new_root.history[depth]:
					self.destroy_from(c) 

			for c in cur_node.child_nodes:
				if c.history[depth] == new_root.history[depth]:
					cur_node = c

			print 'change'
			cur_node.show()
			del cur_node.parent
			cur_node.parent = None
			depth = depth + 1

		self.root = new_root

class Node:

	def __init__(self,visits = 1, value = 0, reward = 0, belief = dict(),history = [],depth = 0,parent = None):
		self.visits = visits
		self.value = value
		self.belief = belief
		self.reward = reward

		self.depth = depth
		self.parent = parent
		self.history = history
		self.child_nodes = []

	def add_child(self,visits = 1, value = 0, reward = 0, belief = dict(),history = None):
		new_child = Node(visits,value,reward,belief,history,self.depth+1,self)
		self.child_nodes.append(new_child)

	def show(self):
		print self.depth,';',self.visits,';',self.value,';',self.history,';',self.reward