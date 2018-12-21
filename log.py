import simulator
import UCT
import time
from collections import defaultdict
from copy import deepcopy
import os
import datetime
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import parameter_estimation
import psutil
import train_data

def create_log_file(path):
        file = open(path, 'w')
        return file

def plot_data_set(agent, estimated_parameter):
    parameters = estimated_parameter.l1_estimation.estimation_history
    # parameters = []
    # for ds in d:
    #     parameters.append(ds["parameter"])

    true_level = agent.level
    true_angle = agent.angle
    true_radius = agent.radius

    levels = []
    angles = []
    radius = []

    # for pr in parameters:
    #     print pr.level,pr.angle,pr.radius
    #     levels.append(pr.level)
    #     angles.append(pr.angle)
    #     radius.append(pr.radius)

    for pr in parameters:
        levels.append(abs(true_level - pr.level))
        angles.append(abs(true_angle - pr.angle))
        radius.append(abs(true_radius - pr.radius))

    fig = plt.figure(1)
#   w = main_sim.agents[0].estimated_parameter.l1_estimation.weight

    plt.subplot(3, 1, 1)
    print levels
    plt.plot([i for i in range(len(levels))], levels,
             label='levels',
             linestyle='-',
             color='cornflowerblue',
             linewidth=1)
    ax = plt.gca()
    ax.set_ylabel('Level error')
    ax.legend(loc="upper right", shadow=True, fontsize='x-large')
    plt.subplot(3, 1, 2)

    plt.plot([i for i in range(len(angles))], angles, label='Angle', linestyle='-', color='cornflowerblue',
             linewidth=1)
    ax = plt.gca()
    ax.set_ylabel('Angle error')

    plt.subplot(3, 1, 3)

    plt.plot([i for i in range(len(radius))], radius, label='Radius', linestyle='-', color='cornflowerblue',
             linewidth=1)

    ax = plt.gca()
    ax.set_ylabel('radius error')
    ax.set_xlabel('weight')


    fig.savefig("./plots/dataset_history_based.jpg")
    plt.show()

def plot_errors(iterations_number, x_train_set):
    # level_set = []
    # angle_set = []
    # radius_set = []
    # for x_train in x_train_set:
    #     a_data_set = np.transpose(np.array(x_train))
    #
    #     if a_data_set != []:
    #
    #         levels = a_data_set[0, :]
    #         angle = a_data_set[1, :]
    #         radius = a_data_set[2, :]
    #
    #         level_set.append(levels)
    #         angle_set.append(angle)
    #         radius_set.append(radius)
    #     else:
    #         level_set.append([])
    #         angle_set.append([])
    #         radius_set.append([])


    parameters = ["Level", "Angle","Radius"]
    parametersLabels = ["Level", "Angle","Radius"]

    parametersSymbol = ["o","v","s"]

    nIterations = [i for i in range(1,iterations_number)]

    #
    # # data = np.zeros((len(nIterations), len(parameters), len(range(max_time_steps))))
    # data = []
    # data.append(level_set)
    # data.append(angle_set)
    # data.append(radius_set)
    #
    # count = np.zeros(2 * len(nIterations))
    #
    #
    # dataMean = np.zeros((len(nIterations)+1, len(parameters)))
    #
    # dataStd = np.zeros((len(nIterations)+1, len(parameters)))
    # print "iterations_number : ", iterations_number
    # print '********Data************'
    # print len(x_train_set)
    # print x_train_set
    # print '********Data Mean************'
    # print len(dataMean)
    # print dataMean
    # print '********Count************'
    # print len(count)
    # print count


    # for n in range(iterations_number):
    #     if level_set[n] == []:
    #         dataMean[n, 0] = np.mean(level_set[n])
    #         dataStd[n, 0] = np.std(level_set[n], ddof=1)
    #     else :
    #         dataMean[n, 0] = 0
    #         dataStd[n, 0] = 0
    #
    # for n in range(iterations_number):
    #     if angle_set[n] == []:
    #         dataMean[n, 1] = np.mean(angle_set[n])
    #         dataStd[n, 1] = np.std(angle_set[n], ddof=1)
    #     else :
    #         dataMean[n, 1] = 0
    #         dataStd[n, 1] = 0
    #
    # for n in range(iterations_number):
    #     if radius_set[n] == []:
    #         dataMean[n, 2] = np.mean(radius_set[n])
    #         dataStd[n, 2] = np.std(radius_set[n], ddof=1)
    #     else :
    #         dataMean[n, 2] = 0
    #         dataStd[n, 2] = 0
    #     # for a in range(len(parameters)):
    #     #     print data[n, a]
    #         # dataMean[n, a] = np.mean(data[n, a, 0:int(count[(n * 2) + a])])
    #         # dataStd[n, a] = np.std(data[n, a, 0:int(count[(n * 2) + a])], ddof=1)
    #

    plt.figure(figsize=(4, 3.0))

    for a in range(len(parameters)):
        print parameters[a]
        print dataMean[:len(nIterations), a]
        print dataStd[:len(nIterations), a]
        plt.errorbar(nIterations, dataMean[:len(nIterations), a], yerr=dataStd[:len(nIterations), a],
                      label=parametersLabels[a], marker=parametersSymbol[a])
        # plt.errorbar(nIterations,  dataMean[:len(nIterations), a],
        #              yerr=[m - n for m, n in zip(dataStd[:len(nIterations), a], dataMean[:len(nIterations), a])],
        #              label=parametersLabels[a],
        #              marker=parametersSymbol[a])

    plt.legend(loc=9, prop={'size': 9})

    plt.ylim(ymax=1)
    plt.ylim(ymin=0)

    plt.xlim([0,len(nIterations) + 2 ])
    plt.xlabel("Iteration number")
    plt.ylabel("Error")
    plt.savefig("plots/Error.pdf", bbox_inches='tight')
    plt.show()

def print_result(main_sim,  time_steps, begin_time, end_time,mcts_mode,estimated_parameter):

    file = open(current_folder + "/results.txt", 'w')
    pickleFile = open(current_folder + "/pickleResults.txt", 'wb')

    dataList = []

    systemDetails = {}

    file.write('sim width:' + str(main_sim.dim_w) + '\n')
    file.write('sim height:' + str(main_sim.dim_h) + '\n')
    file.write('agents counts:' + str(len(main_sim.agents)) + '\n')
    file.write('items counts:' + str(len(main_sim.items)) + '\n')
    file.write('time steps:' + str(time_steps) + '\n')
    file.write('begin time:' + str(begin_time) + '\n')
    file.write('end time:' + str(end_time) + '\n')
    file.write('estimation mode:' + str(parameter_estimation_mode) + '\n')
    file.write('type selection mode:' + str(type_selection_mode) + '\n')
    file.write('iteration max:' + str(iteration_max) + '\n')
    file.write('max depth:' + str(max_depth) + '\n')
    file.write('generated data number:' + str(generated_data_number) + '\n')
    file.write('reuseTree:' + str(reuseTree) + '\n')

    systemDetails['simWidth'] = main_sim.dim_w
    systemDetails['simHeight'] = main_sim.dim_h
    systemDetails['agentsCounts'] = len(main_sim.agents)
    systemDetails['itemsCounts'] = len(main_sim.items)
    systemDetails['timeSteps'] = end_cpu_time - begin_cpu_time
    systemDetails['beginTime'] = begin_time
    systemDetails['endTime'] = end_time
    systemDetails['CPU_Time'] = end_time
    systemDetails['memory_usage'] = memory_usage

    systemDetails['estimationMode'] = parameter_estimation_mode
    systemDetails['typeSelectionMode'] = type_selection_mode
    systemDetails['iterationMax'] = iteration_max
    systemDetails['maxDepth'] = max_depth
    systemDetails['generatedDataNumber'] = generated_data_number
    systemDetails['reuseTree'] = reuseTree
    systemDetails['mcts_mode'] = mcts_mode
    systemDetails['PF_del_threshold'] = PF_del_threshold
    systemDetails['PF_add_threshold'] = PF_add_threshold
    systemDetails['PF_weight'] = PF_weight

    agentDictionary = {}

    for i in range(len(main_sim.agents)):
        agentData = {}
        file.write('#level,radius,angle\n')
        file.write('true type:' + str(main_sim.agents[i].agent_type) + '\n')
        file.write('true parameters:' + str(main_sim.agents[i].level) + ',' + str(main_sim.agents[i].radius)+ ',' +
                   str(main_sim.agents[i].angle) + '\n')
        agentData['trueType'] = main_sim.agents[i].agent_type
        trueParameters = [main_sim.agents[i].level,main_sim.agents[i].radius,main_sim.agents[i].angle]
        agentData['trueParameters'] = trueParameters

        file.write('#probability of type ,level,radius,angle\n')
        # L1 ******************************

        estimated_value = estimated_parameter[i].l1_estimation.get_last_estimation()

        # Result
        file.write('l1:' + str(estimated_parameter[i].l1_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ',' + str(estimated_value.angle)
                   + '\n')
        file.write(str(estimated_parameter[i].l1_estimation.type_probabilities) + '\n')
        file.write(str(estimated_parameter[i].l1_estimation.get_estimation_history()) + '\n')

        # pickleResults
        agentData['l1LastProbability'] = estimated_parameter[i].l1_estimation.get_last_type_probability()
        l1 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['l1'] = l1

        l1EstimationHistory = estimated_parameter[i].l1_estimation.get_estimation_history()
        agentData['l1EstimationHistory'] = l1EstimationHistory
        agentData['l1TypeProbHistory'] = estimated_parameter[i].l1_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        # L2  ******************************

        estimated_value = estimated_parameter[i].l2_estimation.get_last_estimation()

        # Result
        file.write('l2:' + str(estimated_parameter[i].l2_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ','
                       + str(estimated_value.angle) + '\n')
        file.write(str(estimated_parameter[i].l2_estimation.type_probabilities) + '\n')
        file.write(str(estimated_parameter[i].l2_estimation.get_estimation_history()) + '\n')

        # pickleResults
        agentData['l2LastProbability'] = estimated_parameter[i].l2_estimation.get_last_type_probability()
        l2 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['l2'] = l2
        l2EstimationHistory = estimated_parameter[i].l2_estimation.get_estimation_history()
        agentData['l2EstimationHistory'] = l2EstimationHistory
        agentData['l2TypeProbHistory'] = estimated_parameter[i].l2_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        # F1  ******************************

        estimated_value = estimated_parameter[i].f1_estimation.get_last_estimation()

        # Result
        file.write('f1:' + str(estimated_parameter[i].f1_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ','
                       + str(estimated_value.angle) + '\n')
        file.write(str(estimated_parameter[i].f1_estimation.type_probabilities) + '\n')
        file.write(str(estimated_parameter[i].f1_estimation.get_estimation_history()) + '\n')

        # pickleResults

        agentData['f1LastProbability'] = estimated_parameter[i].f1_estimation.get_last_type_probability()
        f1 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['f1'] = f1
        f1EstimationHistory = estimated_parameter[i].f1_estimation.get_estimation_history()
        agentData['f1EstimationHistory'] = f1EstimationHistory
        agentData['f1TypeProbHistory'] = estimated_parameter[i].f1_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        # F2  ******************************

        estimated_value = estimated_parameter[i].f2_estimation.get_last_estimation()

        # Result
        file.write('f2:' + str(estimated_parameter[i].f2_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ','
                       + str(estimated_value.angle) + '\n')
        file.write(str(estimated_parameter[i].f2_estimation.type_probabilities) + '\n')
        file.write(str(estimated_parameter[i].f2_estimation.get_estimation_history()) + '\n')

        # pickleResults

        agentData['f2LastProbability'] = estimated_parameter[i].f2_estimation.get_last_type_probability()
        f2 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['f2'] = f2
        f2EstimationHistory = estimated_parameter[i].f2_estimation.get_estimation_history()
        agentData['f2EstimationHistory'] = f2EstimationHistory
        agentData['f2TypeProbHistory'] = estimated_parameter[i].f2_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        agentDictionary[i]=agentData

    dataList.append(systemDetails)
    dataList.append(agentDictionary)
    print "writing to pickle file."
    pickle.dump(dataList,pickleFile)
    print "writing over "


# print_result(main_sim, time_step, begin_time, end_time,mcts_mode,agants_parameter_estimation)

