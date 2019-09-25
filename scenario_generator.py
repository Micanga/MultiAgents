from random import randint
from random import choice
import random
import csv
import os
import sys

from numpy import pi

# (1) CONFIG.CSV - INFORMATION
# Defining the parameter estimation modes and
max_depth_set = ['100']
iteration_max_set = ['100']

# (2) SIM.CSV - INFORMATION
# Defining the parameter of simulation file
possible_directions = ['N','S','E','W']
agent_types 		= ['l1','l2']
selected_types 		= [False,False]

experiment_type_set =['ABU', 'AGA', 'MIN']# ['TRUE','ABU', 'AGA', 'MIN']
type_estimation_mode_set = ['BPTE']#['LPTE','BTE','BPTE','PTE']
mutation_rate_set = ['0.2']#,'0.3','0.5','0.7','0.9']
apply_adversary = False
round_count = 1

def random_pick(set_):
	return set_[randint(0,len(set_)-1)]

def create_config_file(current_folder,parameter_estimation_mode,mcts_mode,train_mode,mutation_rate,type_estimation_mode):

	filename = current_folder + 'config.csv'
	with open(filename, 'wb+') as file:
		writer = csv.writer(file, delimiter=',')
		GRID = ['type_selection_mode', 'AS']
		writer.writerows([['type_selection_mode', 'AS']])
		writer.writerows([['parameter_estimation_mode', parameter_estimation_mode]])
		writer.writerows([['train_mode', train_mode]])
		writer.writerows([['generated_data_number', '100']])
		writer.writerows([['reuseTree','False']])
		writer.writerows([['mcts_mode', mcts_mode]])
		writer.writerows([['PF_add_threshold', '0.9']])
		writer.writerows([['PF_del_threshold', '0.9']])
		writer.writerows([['PF_weight', '1.2']])
		writer.writerows([['type_estimation_mode', type_estimation_mode]]) # BTE:Bayesian Type Estimation, PTE:Particle Type Estimation,
														   #  BPTE:Bayesian Particle Type Estimation
		writer.writerows([['apply_adversary', apply_adversary]])
		writer.writerows([['round_count', round_count]])
		writer.writerows([['mutation_rate', mutation_rate]])
		writer.writerows([['iteration_max', iteration_max_set[0]]])
		writer.writerows([['max_depth', max_depth_set[0]]])
		writer.writerows([['sim_path', 'sim.csv']])


def generateRandomNumber (grid,gridValue):
	while True:
		testXValue = randint(0, gridValue - 1)
		testYValue = randint(0, gridValue - 1)

		if(grid[testXValue][testYValue] != 1):
			grid[testXValue][testYValue] = 1
			return testXValue,testYValue,grid

def selectType():
	global agent_types, selected_types

	# 1. Selecting a ramdom type
	agentType = choice(agent_types)

	# 2. Verifing if is able to generate this type
	# follower 1 needs lider 1
	if agentType == 'f1' and selected_types[0] == False:
		agentType = 'l1'
		selected_types[0] = True
	# follower 2 needs lider 2
	if agentType == 'f2' and selected_types[1] == False:
		agentType = 'l2'
		selected_types[1] = True
	# lider 1 and 2 as Selected Type
	if agentType == 'l1':
		selected_types[0] = True
	elif agentType == 'l2':
		selected_types[1] = True

	return agentType

def main():
	# 0. Checking the terminal input
	if len(sys.argv) != 6:
		print 'usage: python scenario_generator.py [size] [nagents] [nitems] [map_count] [type_estimation_mode]'
		exit(0)

	# 1. Taking the information
	size = int(sys.argv[1])
	nagents = int(sys.argv[2])
	nitems = int(sys.argv[3])
	map_count = sys.argv[-1]
	tem = sys.argv[5]
	print 'type estimation mode:',tem

	# 2. Defining the simulation
	grid_size = size
	grid = [[0 for col in range(grid_size)] for row in range(grid_size)]
	GRID = ['grid',grid_size,grid_size]

	# d. defining the main agent parameters
	mainx,mainy,grid = generateRandomNumber(grid,grid_size)
	mainDirection    = choice(possible_directions)
	mainType  = 'm'
	mainLevel = 1
	MAIN = ['main',mainx,mainy,mainDirection,mainType,mainLevel]

	# e. defining the commum agents
	AGENTS = []
	for agent_idx in range(nagents):
		agentx,agenty,grid = generateRandomNumber(grid,grid_size)
		agentDirection = choice(possible_directions)
		agentType = selectType()
		agentLevel = round(random.uniform(0.5,1), 3)
		agentRadius = round(random.uniform(0.5,1), 3)
		agentAngle = round(random.uniform(0.5,1), 3)
		AGENTS.append(['agent'+ str(agent_idx),str(agent_idx),agentx,agenty,agentDirection,agentType,agentLevel,agentRadius,agentAngle])

	ITEMS = []
	for item_idx in range(nitems):
		itemx,itemy,grid = generateRandomNumber(grid,grid_size)
		itemLevel = round(random.uniform(0,0.8), 3)
		ITEMS.append(['item'+ str(item_idx),itemx,itemy,itemLevel])

	# 3. Creating the possible configuration files
	# a. choosing the parameter estimation mode
	for experiment in experiment_type_set:
		#for tem in type_estimation_mode_set:
			for mutation_rate in mutation_rate_set:

				if experiment == 'MIN':
					train_mode = 'history_based'
				else:
					train_mode = 'none_history_based'

				# b. choosing the mcts mode
				mcts_mode = 'UCTH'
				MC_type = 'O'

				# c. creating the necessary folder
				sub_dir = 'FO_'+ MC_type + '_' + experiment
				current_folder = "inputs/" + sub_dir + '/'
				if not os.path.exists(current_folder):
					os.mkdir(current_folder, 0755)

				# d. creating the config files
				create_config_file(current_folder, experiment, mcts_mode,train_mode,mutation_rate, tem)

				# 4. Creating the files
				# a. setting the file name
				filename = current_folder + 'sim.csv'
				print filename

				# b. creating the a csv file
				with open(filename,'wb+') as file:
					writer = csv.writer(file,delimiter = ',')

					# i. grid
					writer.writerows([GRID])

					# ii. main agent
					writer.writerows([MAIN])

					# iii. commum agents
					for agent_idx in range(nagents):
						writer.writerows([AGENTS[agent_idx]])

					# iv. items
					for item_idx in range(nitems):
						writer.writerows([ITEMS[item_idx]])

			# c. saving map
				with open('maps/'+map_count+'.csv','wb+') as file:
					writer = csv.writer(file,delimiter = ',')

					# i. grid
					writer.writerows([GRID])

					# ii. main agent
					writer.writerows([MAIN])

					# iii. commum agents
					for agent_idx in range(nagents):
						writer.writerows([AGENTS[agent_idx]])

					# iv. items
					for item_idx in range(nitems):
						writer.writerows([ITEMS[item_idx]])

if __name__ == '__main__':
    main()
