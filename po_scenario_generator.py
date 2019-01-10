from random import randint
from random import choice
import random
import csv
import os
import sys

from numpy import pi

# (1) CONFIG.CSV - INFORMATION
# Defining the parameter estimation modes and
# the mcts modes
parameter_estimation_modes = ['ABU','AGA','MIN']
mcts_modes = ['UCT','UCTH']
max_depth_set = ['50']
iteration_max_set = ['50']

# (2) SIM.CSV - INFORMATION
# Defining the parameter of simulation file
square_grid_size 	= [10]
number_of_agents 	= [1]
number_of_items		= [10]
possible_directions = ['N','S','E','W']
agent_types 		= ['l1','l2'] 

def create_config_file(current_folder,parameter_estimation_mode,mcts_mode,train_mode):

	filename = current_folder + 'po_config.csv'
	with open(filename, 'wb+') as file:
		writer = csv.writer(file, delimiter=',')
		GRID = ['type_selection_mode', 'AS']
		writer.writerows([['type_selection_mode', 'AS']])
		writer.writerows([['parameter_estimation_mode', parameter_estimation_mode]])
		writer.writerows([['train_mode', train_mode]])
		writer.writerows([['generated_data_number', '100']])
		writer.writerows([['reuseTree','False']])
		writer.writerows([['mcts_mode', mcts_mode]])
		writer.writerows([['PF_add_threshold', '0.8']])
		writer.writerows([['PF_del_threshold', '0.9']])
		writer.writerows([['PF_weight', '1.2']])
		writer.writerows([['iteration_max', iteration_max_set[randint(0,len(iteration_max_set)-1)]]])
		writer.writerows([['max_depth', max_depth_set[randint(0,len(max_depth_set)-1)]]])
		writer.writerows([['sim_path', 'posim.csv']])


def generateRandomNumber (grid,gridValue):
	while True:
		testXValue = randint(0, gridValue - 1)
		testYValue = randint(0, gridValue - 1)

		if(grid[testXValue][testYValue] != 1):
			grid[testXValue][testYValue] = 1
			return testXValue,testYValue,grid


def main():
	# 0. Checking the terminal input
	if len(sys.argv) != 3:
		print 'usage: python po_run_world.py [main_radius] [angle]'
		exit(0)

	# 1. Taking the information
	MAIN_INFO = sys.argv[1] + '_' + sys.argv[2]
	LEVEL = 1
	RADIUS = int(sys.argv[1])
	if sys.argv[2] == '0pi':
		ANGLE = 0
	elif sys.argv[2] == '1pi2':
		ANGLE = (1/2)*pi
	elif sys.argv[2] == '1pi':
		ANGLE = pi
	elif sys.argv[2] == '3pi2':
		ANGLE = (3/2)*pi
	elif sys.argv[2] == '2pi':
		ANGLE = 2*pi
	else:
		exit(1)

	# 1. Creating the possible configuration files
	for parameter_estimation_mode in parameter_estimation_modes:
		# a. choosing the parameter estimation mode
		if parameter_estimation_mode == 'MIN':
			train_mode= 'history_based'
		else:
			train_mode= 'none_history_based'

		# b. choosing the mcts mode
		for mcts_mode in mcts_modes:
			if mcts_mode == 'UCT' :
				MC_type = 'M'
			else:
				MC_type = 'O'
			sub_dir = 'PO_'+ MC_type + '_' + parameter_estimation_mode
			current_folder = "po_inputs/" + sub_dir + '/'

			if not os.path.exists(current_folder):
					os.mkdir(current_folder, 0755)

			# c. creating the config files
			create_config_file(current_folder, parameter_estimation_mode, mcts_mode,train_mode)

	# 2. Creating the files
	global MAIN_INFO,LEVEL, RADIUS, ANGLE
	for path,all_sub_dir,files in os.walk("po_inputs/"):
		for cur_dir in all_sub_dir:
			print all_sub_dir
			# a. setting the file name
			filename = path + cur_dir + '/' + 'posim.csv'
			print filename

			# b. creating the a csv file
			with open(filename,'wb+') as file:
				writer = csv.writer(file,delimiter = ',')

				# c. choosing the grid size
				grid_size = square_grid_size[randint(0,len(square_grid_size)-1)]
				grid = [[0 for col in range(grid_size)] for row in range(grid_size)]
				GRID = ['grid',grid_size,grid_size]
				writer.writerows([GRID])

				# d. defining the main agent parameters
				mainx,mainy,grid = generateRandomNumber(grid,grid_size)
				mainDirection    = choice(possible_directions)
				mainType  = 'm'
				mainLevel = LEVEL
				mainRadius, mainAngle = RADIUS, ANGLE
				MAIN = ['main',mainx,mainy,mainDirection,mainType,mainLevel,mainRadius,mainAngle]
				writer.writerows([MAIN])

				# e. defining the commum agents
				nagents = 1 #number_of_agents[randint(0,len(number_of_agents)-1)]
				for agent_idx in range(nagents):
					agentx,agenty,grid = generateRandomNumber(grid,grid_size)
					agentDirection = choice(possible_directions)
					agentType = 'l1'
					agentLevel = round(random.uniform(0,1), 3)
					agentRadius = round(random.uniform(0.1,1), 3)
					agentAngle = round(random.uniform(0.1,1), 3)

					AGENT = ['agent'+ str(agent_idx),agentx,agenty,agentDirection,agentType,agentLevel,agentRadius,agentAngle]
					writer.writerows([AGENT])

				nitems = 10 #number_of_items[randint(0,len(number_of_items)-1)]
				for item_idx in range(nitems):
					itemx,itemy,grid = generateRandomNumber(grid,grid_size)
					itemLevel = round(random.uniform(0,1), 3)

					ITEM = ['item'+ str(item_idx),itemx,itemy,itemLevel]
					writer.writerows([ITEM])

if __name__ == '__main__':
    main()
