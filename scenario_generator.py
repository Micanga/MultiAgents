from random import randint
import random
import csv
import os

agentValues = [1,2,3,4]
agentValueUpperLimit = 2 # set as default, use this variable to set the range in which agentValues can lie!
gridSize = [10,15,20,25]
gridValueUpperLimit = 25 # default value set, use this variable to set range of grid
directions = ['N','S','E','W']
types = ['l1','l2'] #['l1','l2','f1','f2']

def create_config(current_folder,parameter_estimation_mode,mcts_mode,train_mode):

	print(current_folder)
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
		writer.writerows([['PF_weight', '1.2']])
		writer.writerows([['iteration_max', '100']])
		writer.writerows([['max_depth', '100']])
		writer.writerows([['sim_path', 'sim.csv']])



def generateRandomNumber (grid,gridValue):
	print gridValue

	while 1==1:
		testXValue = randint(0, gridValue - 1)
		testYValue = randint(0, gridValue - 1)
		print testXValue,testYValue
		if(grid[testXValue][testYValue] != 1):
			
			print grid
			grid[testXValue][testYValue] = 1
			return testXValue,testYValue,grid
		#else:
		#	generateRandomNumber(grid,gridValue)

parameter_estimation_modes = ['ABU','AGA','MIN']

mcts_modes =['UCT','UCTH']
dataFilesNumber = 1
count=0
while count < 5:
	for parameter_estimation_mode in parameter_estimation_modes:
		if parameter_estimation_mode == 'MIN':
			train_mode= 'history_based'
		else:
			train_mode= 'none_history_based'
		for mcts_mode in mcts_modes:
			 
			for i in range(0,dataFilesNumber):
				agent = 2
				gridValue = 20
				if mcts_mode == 'UCT' :
					MC_type = 'M'
				else:
					MC_type = 'O'
				sub_dir = str(gridValue) + 'S_'+ str(agent) + 'A_'+ MC_type + '_' +parameter_estimation_mode+ str(count)
				current_folder = "input/" + sub_dir + '/'
				if not os.path.exists(current_folder):
					os.mkdir(current_folder, 0755)

				filename = current_folder + 'sim.csv'
				with open(filename,'wb+') as file:
					writer = csv.writer(file,delimiter = ',')
					angleValue = round(random.uniform(0.1,1), 3) # rounds the value upto 3 places
					index = randint(0,3)
					GRID = ['grid',gridValue,gridValue]
					writer.writerows([GRID])
					print gridValue

					grid = [[0 for col in range(gridValue)] for row in range(gridValue)]



					mainx,mainy,grid = generateRandomNumber(grid,gridValue)
					mainDirection = directions[randint(0,3)]
					mainType ='m'# types[randint(0,3)]
					mainLevel = 1#round(random.uniform(0,1), 3)
					MAIN = ['main',mainx,mainy,mainDirection,mainType,mainLevel]
					writer.writerows([MAIN])

					agentXValues = []
					agentYValues = []
					agentValuesDict = {}
					for i in range(0,agent):
						agentx,agenty,grid = generateRandomNumber(grid,gridValue)
						agentDirection = directions[randint(0,3)]
						agentType = 'l1' #types[randint(0,3)]
						agentLevel = round(random.uniform(0,1), 3)
						agentLevel = 1
						agentRadius = round(random.uniform(0.1,1), 3)
						agentAngle = angleValue

						AGENT = ['agent'+ str(i),agentx,agenty,agentDirection,agentType,agentLevel,agentRadius,agentAngle]
						agentXValues.append(agentx)

						agentYValues.append(agenty)
						writer.writerows([AGENT])

					for i in range(0,gridValue):
						itemx,itemy,grid = generateRandomNumber(grid,gridValue)
						itemLevel = round(random.uniform(0,1), 3)
						itemLevel = 0
						ITEM = ['item'+ str(i),itemx,itemy,itemLevel]
						writer.writerows([ITEM])

				create_config(current_folder, parameter_estimation_mode, mcts_mode,train_mode)
	count +=1
