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


x_train_set = []
level_set = []
angle_set = []
radius_set = []


dataMean = np.zeros((100, 3))
dataStd = np.zeros((100, 3))

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
        # print parameters[a]
        # print dataMean[:len(nIterations), a]
        # print dataStd[:len(nIterations), a]
        # plt.errorbar(nIterations, dataMean[:len(nIterations), a], yerr=dataStd[:len(nIterations), a],
        #               label=parametersLabels[a], marker=parametersSymbol[a])
        plt.errorbar(nIterations,  dataMean[:len(nIterations), a],
                     yerr=[m - n for m, n in zip(dataStd[:len(nIterations), a], dataMean[:len(nIterations), a])],
                     label=parametersLabels[a],
                     marker=parametersSymbol[a])

    plt.legend(loc=9, prop={'size': 9})

    plt.ylim(ymax=1)
    plt.ylim(ymin=0)

    plt.xlim([0,len(nIterations) + 2 ])
    plt.xlabel("Iteration number")
    plt.ylabel("Error")
    plt.savefig("plots/Error.pdf", bbox_inches='tight')
    plt.show()


memory_usage = 0
iMaxStackSize = 2000
sys.setrecursionlimit(iMaxStackSize)
types = ['l1', 'l2', 'f1', 'f2']

iteration_max = None
type_selection_mode = None
parameter_estimation_mode = None
generated_data_number = None
reuse_tree = None
max_depth = None
sim_path = None
do_estimation = True
mcts_mode = None
PF_add_threshold = None
apply_adversary = False
train_mode = None


# ============= Set Input/Output ============
now = datetime.datetime.now()
# sub_dir = now.strftime("%Y-%m-%d %H:%M")
sub_dir = str(now.day) + "_"+ str(now.hour)+ "_" + str(now.minute)
current_folder = "outputs/" + sub_dir + '/'
if not os.path.exists(current_folder):
    os.mkdir(current_folder, 0755)

dir = ""
if len(sys.argv) > 1 :
    dir = str(sys.argv[1])

dir = "inputs/history/"
# path = 'config.csv'
path = dir + 'config.csv'
print 'path: ',path
info = defaultdict(list)
with open(path) as info_read:
    for line in info_read:
        data = line.strip().split(',')
        key, val = data[0], data[1:]
        info[key].append(val)


# ============= Read configuration ============
for k, v in info.items():

    if 'type_selection_mode' in k:
        type_selection_mode = str(v[0][0]).strip()

    if 'parameter_estimation_mode' in k:
        parameter_estimation_mode = str(v[0][0]).strip()

    if 'train_mode' in k:
        train_mode = str(v[0][0]).strip()

    if 'generated_data_number' in k:
        generated_data_number = int(v[0][0])

    if 'reuseTree' in k:
        reuse_tree = v[0][0]

    if 'iteration_max' in k:
        iteration_max = int(v[0][0])

    if 'max_depth' in k:
        max_depth = int(v[0][0])

    if 'PF_add_threshold' in k:
        PF_add_threshold = float(v[0][0])

    if 'PF_del_threshold' in k:
        PF_del_threshold = float(v[0][0])

    if 'PF_weight' in k:
        PF_weight = float(v[0][0])

    if 'do_estimation' in k:
        if v[0][0] == 'False':
            do_estimation = False
        else:
            do_estimation = True

    if 'apply_adversary' in k:
        if v[0][0] == 'False':
            apply_adversary = False
        else:
            apply_adversary = True

    if 'sim_path' in k:
        sim_path = dir + str(v[0][0]).strip()

    if 'mcts_mode' in k:
        mcts_mode = str(v[0][0]).strip()



main_sim = simulator.Simulator()

main_sim.loader(sim_path)
logfile = main_sim.create_log_file(current_folder + "log.txt")

# ======================================================================================================
for i in range(len(main_sim.agents)):
    print 'true values : level :', main_sim.agents[i].level, ' radius: ', main_sim.agents[i].radius, ' angle: ' \
        , main_sim.agents[i].angle

main_sim.draw_map()
main_sim.log_map(logfile)

# ============= Initialization =========================================================================================

time_step = 0
begin_time = time.time()
begin_cpu_time = psutil.cpu_times()
used_mem_before = psutil.virtual_memory().used
polynomial_degree = 4

agents_parameter_estimation = []
agents_previous_step_info = []


if main_sim.main_agent is not None:
    main_agent = main_sim.main_agent
    search_tree = None

    main_sim.main_agent.initialise_visible_agents(main_sim,generated_data_number, PF_add_threshold, train_mode,
                                              type_selection_mode, parameter_estimation_mode, polynomial_degree,apply_adversary)
    uct = UCT.UCT(iteration_max, max_depth, do_estimation, mcts_mode, apply_adversary,enemy=False)
    main_sim.main_agent.initialise_uct(uct)

if apply_adversary:
    enemy_agent = main_sim.enemy_agent
    enemy_search_tree = None
    if main_sim.enemy_agent is not None:
        main_sim.enemy_agent.initialise_visible_agents(main_sim,generated_data_number, PF_add_threshold, train_mode,
                                                  type_selection_mode, parameter_estimation_mode, polynomial_degree,apply_adversary)
        enemy_uct = UCT.UCT(iteration_max, max_depth, do_estimation, mcts_mode,apply_adversary, enemy=True )
        main_sim.enemy_agent.initialise_uct(enemy_uct)


while main_sim.items_left() > 0:

    print '-------------------------------Iteration number ', time_step, '--------------------------------------'

    if main_sim.main_agent is not None:
        main_sim.main_agent.update_unknown_agents(main_sim)

    for i in range(len(main_sim.agents)):

        main_sim.agents[i] = main_sim.move_a_agent(main_sim.agents[i])

        print 'target: ', main_sim.agents[i].get_memory()


    # print('****** Movement of Intelligent agent based on MCTS ****************************************************')
    if main_sim.main_agent is not None:
        r,search_tree = main_sim.main_agent.move(reuse_tree, main_sim, search_tree, time_step)

    if main_sim.enemy_agent is not None:
        r, enemy_search_tree = main_sim.enemy_agent.move(reuse_tree, main_sim, enemy_search_tree, time_step)

    main_sim.update_all_A_agents(False)
    main_sim.do_collaboration()
    main_sim.main_agent.update_unknown_agents_actions(main_sim)
    '********* Estimation for selfish agents ******'
    if do_estimation:
        main_sim.main_agent.estimation(time_step,main_sim)

    time_step += 1
    # print '---x_train_set in time step ', time_step ,' is :  '
    # for xts in x_train_set:
    #     print xts



    main_sim.draw_map()
    main_sim.log_map(logfile)

    if main_sim.items_left() == 0:
        break

    print "left items", main_sim.items_left()
    print('***********************************************************************************************************')

# plot_data_set(main_sim.agents[0],agents_parameter_estimation[0])
#plot_errors(time_step,x_train_set)
main_sim.main_agent.visible_agents[0].agents_parameter_estimation.plot_data_set()

end_time = time.time()
used_mem_after = psutil.virtual_memory().used
end_cpu_time = psutil.cpu_times()
memory_usage = used_mem_after - used_mem_before


def print_result(main_sim,  time_steps, begin_time, end_time,mcts_mode):

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
    file.write('reuseTree:' + str(reuse_tree) + '\n')

    systemDetails['simWidth'] = main_sim.dim_w
    systemDetails['simHeight'] = main_sim.dim_h
    systemDetails['agentsCounts'] = len(main_sim.agents)
    systemDetails['itemsCounts'] = len(main_sim.items)
#    systemDetails['timeSteps'] = end_cpu_time - begin_cpu_time
    systemDetails['beginTime'] = begin_time
    systemDetails['endTime'] = end_time
    systemDetails['CPU_Time'] = end_time
    systemDetails['memory_usage'] = memory_usage

    systemDetails['estimationMode'] = parameter_estimation_mode
    systemDetails['typeSelectionMode'] = type_selection_mode
    systemDetails['iterationMax'] = iteration_max
    systemDetails['maxDepth'] = max_depth
    systemDetails['generatedDataNumber'] = generated_data_number
    systemDetails['reuseTree'] = reuse_tree
    systemDetails['mcts_mode'] = mcts_mode
    systemDetails['PF_del_threshold'] = PF_del_threshold
    systemDetails['PF_add_threshold'] = PF_add_threshold
    systemDetails['PF_weight'] = PF_weight

    agentDictionary = {}

    for i in range(len(main_sim.agents)):
        u_a = main_sim.main_agent.visible_agents[i]
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

        estimated_value = u_a.l1_estimation.get_last_estimation()

        # Result
        file.write('l1:' + str(u_a.agents_parameter_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ',' + str(estimated_value.angle)
                   + '\n')
        file.write(str(u_a.agents_parameter_estimation.l1_estimation.type_probabilities) + '\n')
        file.write(str(u_a.agents_parameter_estimation.l1_estimation.get_estimation_history()) + '\n')

        # pickleResults
        agentData['l1LastProbability'] = u_a.agents_parameter_estimation.l1_estimation.get_last_type_probability()
        l1 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['l1'] = l1

        l1EstimationHistory = u_a.agents_parameter_estimation.l1_estimation.get_estimation_history()
        agentData['l1EstimationHistory'] = l1EstimationHistory
        agentData['l1TypeProbHistory'] = u_a.agents_parameter_estimation.l1_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        # L2  ******************************

        estimated_value = u_a.agents_parameter_estimation.l2_estimation.get_last_estimation()

        # Result
        file.write('l2:' + str(u_a.agents_parameter_estimation.l2_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ','
                       + str(estimated_value.angle) + '\n')
        file.write(str(u_a.agents_parameter_estimation.l2_estimation.type_probabilities) + '\n')
        file.write(str(u_a.agents_parameter_estimation.l2_estimation.get_estimation_history()) + '\n')

        # pickleResults
        agentData['l2LastProbability'] = u_a.agents_parameter_estimation.l2_estimation.get_last_type_probability()
        l2 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['l2'] = l2
        l2EstimationHistory = u_a.agents_parameter_estimation.l2_estimation.get_estimation_history()
        agentData['l2EstimationHistory'] = l2EstimationHistory
        agentData['l2TypeProbHistory'] = u_a.agents_parameter_estimation.l2_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        # F1  ******************************

        estimated_value = u_a.agents_parameter_estimation.f1_estimation.get_last_estimation()

        # Result
        file.write('f1:' + str(u_a.agents_parameter_estimation.f1_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ','
                       + str(estimated_value.angle) + '\n')
        file.write(str(u_a.agents_parameter_estimation.f1_estimation.type_probabilities) + '\n')
        file.write(str(u_a.agents_parameter_estimation.f1_estimation.get_estimation_history()) + '\n')

        # pickleResults

        agentData['f1LastProbability'] = u_a.agents_parameter_estimation.f1_estimation.get_last_type_probability()
        f1 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['f1'] = f1
        f1EstimationHistory = u_a.agents_parameter_estimation.f1_estimation.get_estimation_history()
        agentData['f1EstimationHistory'] = f1EstimationHistory
        agentData['f1TypeProbHistory'] = u_a.agents_parameter_estimation.f1_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        # F2  ******************************

        estimated_value = u_a.agents_parameter_estimation.f2_estimation.get_last_estimation()

        # Result
        file.write('f2:' + str(u_a.agents_parameter_estimation.f2_estimation.get_last_type_probability()))
        file.write(',' + str(estimated_value.level) + ',' + str(estimated_value.radius) + ','
                       + str(estimated_value.angle) + '\n')
        file.write(str(u_a.agents_parameter_estimation.f2_estimation.type_probabilities) + '\n')
        file.write(str(u_a.agents_parameter_estimation.f2_estimation.get_estimation_history()) + '\n')

        # pickleResults

        agentData['f2LastProbability'] = u_a.agents_parameter_estimation.f2_estimation.get_last_type_probability()
        f2 = [estimated_value.level,estimated_value.radius,estimated_value.angle]
        agentData['f2'] = f2
        f2EstimationHistory = u_a.agents_parameter_estimation.f2_estimation.get_estimation_history()
        agentData['f2EstimationHistory'] = f2EstimationHistory
        agentData['f2TypeProbHistory'] = u_a.agents_parameter_estimation.f2_estimation.type_probabilities
        agentData['last_estimated_value'] = estimated_value

        agentDictionary[i]=agentData

    dataList.append(systemDetails)
    dataList.append(agentDictionary)
    print "writing to pickle file."
    pickle.dump(dataList,pickleFile)
    print "writing over "
print '============================================================================================================================================='
print_result(main_sim,  time_step, begin_time, end_time,mcts_mode)


