import posimulator
import POUCT
import time
from collections import defaultdict
from copy import copy
import os
import datetime
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import parameter_estimation
import psutil
import train_data
import gc

x_train_set = []
level_set = []
angle_set = []
radius_set = []


dataMean = np.zeros((100, 3))
dataStd = np.zeros((100, 3))

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
from random import randint
sub_dir = str(now.day) + "_"+ str(now.hour)+ "_" + str(now.minute) + "_k" + str(randint(0,now.day+now.hour+now.minute))
current_folder = "po_outputs/" + sub_dir + '/'
if not os.path.exists(current_folder):
    os.mkdir(current_folder, 0755)

dir = ""
if len(sys.argv) > 1 :
    dir = str(sys.argv[1])

path = dir + 'po_config.csv'
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
        if len(sys.argv) > 1:
            sim_path = dir + str(v[0][0]).strip()
        else:
            sim_path = str(v[0][0]).strip()

    if 'mcts_mode' in k:
        mcts_mode = str(v[0][0]).strip()

print 'max it:',iteration_max,'/max depth:',max_depth
main_sim = posimulator.POSimulator()

main_sim.loader(sim_path)
logfile = main_sim.create_log_file(current_folder + "log.txt")

# ======================================================================================================================
for i in range(len(main_sim.agents)):
    print 'true values : level :', main_sim.agents[i].level, ' radius: ', main_sim.agents[i].radius, ' angle: '\
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
    uct = POUCT.POUCT(iteration_max, max_depth, do_estimation, mcts_mode, apply_adversary,enemy=False)
    main_sim.main_agent.initialise_uct(uct)

if apply_adversary:
    enemy_agent = main_sim.enemy_agent
    enemy_search_tree = None
    if main_sim.enemy_agent is not None:
        main_sim.enemy_agent.initialise_visible_agents(main_sim,generated_data_number, PF_add_threshold, train_mode,
                                                  type_selection_mode, parameter_estimation_mode, polynomial_degree,apply_adversary)
        enemy_uct = POUCT.POUCT(iteration_max, max_depth, do_estimation, mcts_mode,apply_adversary, enemy=True )
        main_sim.enemy_agent.initialise_uct(enemy_uct)


for v_a in main_sim.main_agent.visible_agents:
    v_a.choose_target_state = copy(main_sim)
    # print v_a.agents_parameter_estimation.get_highest_type_probability()
for i_a in main_sim.main_agent.invisible_agents:
    i_a.choose_target_state = copy(main_sim)

while main_sim.items_left() > 0:

    print '-------------------------------Iteration number ', time_step, '--------------------------------------'

    if main_sim.main_agent is not None:
        print('****** UPDATE UNKNOWN AGENT **********')
        main_sim.main_agent.previous_state = main_sim.copy()
        main_sim.main_agent.update_unknown_agents(main_sim)

    print('****** MOVE AGENT **********')
    for i in range(len(main_sim.agents)):
        main_sim.agents[i] = main_sim.move_a_agent(main_sim.agents[i])

        print('main_sim.agents[i].next_action=', main_sim.agents[i].next_action)
        # print 'agent ',main_sim.agents[i].index,' target: ', main_sim.agents[i].get_memory()

    print('****** Movement of Intelligent agent based on MCTS ****************************************************')
    if main_sim.main_agent is not None:
        r,enemy_action_prob,search_tree = main_sim.main_agent.move(reuse_tree, main_sim, search_tree, time_step)

    if main_sim.enemy_agent is not None:
        # print('****** Movement of Enemy agent based on MCTS ****************************************************')
        r, main_action_prob,enemy_search_tree = main_sim.enemy_agent.move(reuse_tree, main_sim, enemy_search_tree, time_step)

    actions = main_sim.update_all_A_agents(False)
    main_sim.do_collaboration()
    main_sim.main_agent.update_unknown_agents_status(main_sim)
    main_sim.draw_map()


    print '********* Estimation for selfish agents ******'
    if do_estimation:
        main_sim.main_agent.estimation(time_step,main_sim,enemy_action_prob,actions )

    time_step += 1
    # print '---x_train_set in time step ', time_step ,' is :  '
    # for xts in x_train_set:
    #     print xts

    main_sim.log_map(logfile)

    if main_sim.items_left() == 0:
        break

    search_tree = uct.update_belief_state(main_sim,search_tree)
    print "main agent left items", main_sim.main_agent.items_left()

    print "left items", main_sim.items_left()
    gc.collect()

    print('***********************************************************************************************************')

# plot_data_set(main_sim.agents[0],agents_parameter_estimation[0])
# plot_errors(time_step,x_train_set)

for v_a in main_sim.main_agent.visible_agents:
    print v_a.agents_parameter_estimation.get_highest_type_probability()
# main_sim.main_agent.visible_agents[0].agents_parameter_estimation.plot_data_set()

end_time = time.time()
used_mem_after = psutil.virtual_memory().used
end_cpu_time = psutil.cpu_times()
memory_usage = used_mem_after - used_mem_before


def print_result(main_sim,  time_steps, begin_time, end_time,mcts_mode):

    pickleFile = open(current_folder + "/pickleResults.txt", 'wb')

    dataList = []

    systemDetails = {}


    systemDetails['simWidth'] = main_sim.dim_w
    systemDetails['simHeight'] = main_sim.dim_h
    systemDetails['mainAgentRadius'] = main_sim.main_agent.vision.radius
    systemDetails['mainAgentAngle'] =main_sim.main_agent.vision.angle
    systemDetails['agentsCounts'] = len(main_sim.agents)
    systemDetails['itemsCounts'] = len(main_sim.items)
    systemDetails['timeSteps'] = time_steps
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

    systemDetails['PF_add_threshold'] = PF_add_threshold
    systemDetails['PF_weight'] = PF_weight

    agentDictionary = {}

    for i in range(len(main_sim.agents)):
        u_a = main_sim.main_agent.agent_memory[i]
        agentData = {}

        agentData['trueType'] = main_sim.agents[i].agent_type
        trueParameters = [main_sim.agents[i].level,main_sim.agents[i].radius,main_sim.agents[i].angle]
        agentData['trueParameters'] = trueParameters

        agentData['maxProbability'] = u_a.agents_parameter_estimation.get_highest_type_probability()

        l1EstimationHistory = u_a.agents_parameter_estimation.l1_estimation.get_estimation_history()
        agentData['l1EstimationHistory'] = l1EstimationHistory
        agentData['l1TypeProbHistory'] = u_a.agents_parameter_estimation.l1_estimation.type_probabilities
        print u_a.agents_parameter_estimation.l1_estimation.type_probabilities

        l2EstimationHistory = u_a.agents_parameter_estimation.l2_estimation.get_estimation_history()
        agentData['l2EstimationHistory'] = l2EstimationHistory
        agentData['l2TypeProbHistory'] = u_a.agents_parameter_estimation.l2_estimation.type_probabilities
        print u_a.agents_parameter_estimation.l2_estimation.type_probabilities

        f1EstimationHistory = u_a.agents_parameter_estimation.f1_estimation.get_estimation_history()
        agentData['f1EstimationHistory'] = f1EstimationHistory
        agentData['f1TypeProbHistory'] = u_a.agents_parameter_estimation.f1_estimation.type_probabilities

        f2EstimationHistory = u_a.agents_parameter_estimation.f2_estimation.get_estimation_history()
        agentData['f2EstimationHistory'] = f2EstimationHistory
        agentData['f2TypeProbHistory'] = u_a.agents_parameter_estimation.f2_estimation.type_probabilities

        agentDictionary[i]=agentData

    dataList.append(systemDetails)
    dataList.append(agentDictionary)
    print "writing to pickle file."
    pickle.dump(dataList,pickleFile)
    print "writing over "
print '============================================================================================================================================='
print_result(main_sim,  time_step, begin_time, end_time,mcts_mode)


