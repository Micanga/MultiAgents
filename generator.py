from random import randint
import random
import csv

agentValues = [1,2,3,4]
agentValueUpperLimit = 2 # set as default, use this variable to set the range in which agentValues can lie!
gridSize = [10,15,20,25]
gridValueUpperLimit = 25 # default value set, use this variable to set range of grid
directions = ['N','S','E','W']
types = ['l1','l2','f1','f2']

def generateRandomNumber (grid,gridValue):
	testXValue = randint(0,gridValue -1)
	testYValue = randint(0,gridValue -1)
	print testXValue , testYValue
	if(grid[testXValue][testYValue] != 1):
		grid[testXValue][testYValue] = 1
		return testXValue,testYValue,grid
	else:
		generateRandomNumber(grid,gridValue)

dataFilesNumber = 2
for i in range(0,dataFilesNumber):
	filename = 'input/sim %s.csv'%i
	with open(filename,'wb+') as file:
		writer = csv.writer(file,delimiter = ',')

		angleValue = round(random.uniform(0.1,1), 3) # rounds the value upto 3 places

		index = randint(0,3)

		agent = 2 # randint(1,agentValueUpperLimit)
		gridValue = 20 #randint(10,gridValueUpperLimit)

		GRID = ['grid',gridValue,gridValue]
		writer.writerows([GRID])

		grid = [[0 for col in range(gridValue)] for row in range(gridValue)]


		mainx = randint(0,gridValue)
		mainy = randint(0,gridValue)
		mainDirection = directions[randint(0,3)]
		mainType = types[randint(0,3)]
		mainLevel = round(random.uniform(0,1), 3)
		grid[mainx][mainy] = 1 # this place in the grid has been occupied!
		MAIN = ['main',mainx,mainy,mainDirection,mainType,mainLevel]
		writer.writerows([MAIN])

		agentXValues = []
		agentYValues = []
		agentValuesDict = {} 
		for i in range(0,agent):
			agentx,agenty,grid = generateRandomNumber(grid,gridValue)
			agentDirection = directions[randint(0,3)]
			agentType = types[randint(0,3)]
			agentLevel = round(random.uniform(0,1), 3)
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







