#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import os, sys
import gc

# FO Imports
import simulator
import UCT

# PO Imports
import POMDP
import poagent
import POUCT
import POMCP
import posimulator
import simulatorCommonMethods

# General Imports
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

"""
██╗      ██████╗  ██████╗ 
██║     ██╔═══██╗██╔════╝ 
██║     ██║   ██║██║  ███╗
██║     ██║   ██║██║   ██║
███████╗╚██████╔╝╚██████╔╝
╚══════╝ ╚═════╝  ╚═════╝                       
"""
from log import *

# 1. Oppening/Starting the log files and directories
current_folder = "outputs/"
if not os.path.exists(current_folder):
    os.mkdir(current_folder, 0755)

dir = ""
if len(sys.argv) > 1 :
    dir = str(sys.argv[1])


"""
███████╗███████╗████████╗████████╗██╗███╗   ██╗ ██████╗ ███████╗
██╔════╝██╔════╝╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝ ██╔════╝
███████╗█████╗     ██║      ██║   ██║██╔██╗ ██║██║  ███╗███████╗
╚════██║██╔══╝     ██║      ██║   ██║██║╚██╗██║██║   ██║╚════██║
███████║███████╗   ██║      ██║   ██║██║ ╚████║╚██████╔╝███████║
╚══════╝╚══════╝   ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝

"""
gc.collect()
# 1. Setting memory parameters
dataMean, dataStd = np.zeros((100, 3)), np.zeros((100, 3))
memory_usage, iMaxStackSize = 0, 2000
sys.setrecursionlimit(iMaxStackSize)

# 2. Setting simulation parameters
x_train_set, level_set = [], []
angle_set, radius_set = [], []

type_selection_mode = None
types = ['l1', 'l2', 'f1', 'f2']

do_estimation = True
parameter_estimation_mode = None
generated_data_number = None
sim_path = None

reuseTree = None
max_depth, iteration_max = None, None

mcts_mode = None # Multiple State Per Action (MSPA)/ One State Per Action (OSPA)
PF_add_threshold = None

train_mode = None

simulation_visibility = 'PO' # 'FO' or 'PO'

# 3. Oppenning and reading the world's configuration
path = 'config.csv'
info = defaultdict(list)
with open(path) as info_read:
    for line in info_read:
        data = line.strip().split(',')
        key, val = data[0], data[1:]
        info[key].append(val)

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
        reuseTree = v[0][0]
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
    if 'sim_path' in k:
        sim_path = dir + str(v[0][0]).strip()
    if 'mcts_mode' in k:
        mcts_mode = str(v[0][0]).strip()

"""
██████╗ ██╗   ██╗███╗   ██╗    ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗ 
██╔══██╗██║   ██║████╗  ██║    ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
██████╔╝██║   ██║██╔██╗ ██║    ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
██╔══██╗██║   ██║██║╚██╗██║    ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
██║  ██║╚██████╔╝██║ ╚████║    ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝     ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝ 

"""
# 1. Starting the runtime counter
now = datetime.datetime.now()
sub_dir = str(now.day) + "_"+ str(now.hour)+ "_" + str(now.minute)

# 2. Starting the main simulation
main_sim = None
if simulation_visibility == 'FO':
    main_sim = simulator.Simulator()
    main_sim.loader(sim_path)
    logfile = main_sim.create_log_file(current_folder + "log.txt")
elif simulation_visibility == 'PO':
    main_sim = posimulator.POSimulator()
    main_sim.loader(sim_path)
    logfile = create_log_file(current_folder + "log.txt")
else:
    print ':: invalid simulation visibility.\nusage: run_world::85:: $ simulation_visibility = \'FO\'  or \'PO\' ::'
    exit(1)

# 2. Starting the the MCT/POMCP tree
uct, pomcp = None, None
if simulation_visibility == 'FO':
    uct = UCT.UCT(reuseTree, iteration_max, max_depth, do_estimation, mcts_mode)
elif simulation_visibility == 'PO':
    pomcp = POMCP.POMCP(main_sim.main_agent,None,iteration_max,max_depth)

# 3. Printing the main sim A agents
for i in range(len(main_sim.agents)):
    print 'true values : level :', main_sim.agents[i].level, ' radius: ', main_sim.agents[i].radius, ' angle: ' \
        , main_sim.agents[i].angle
print 'true values : level :', main_sim.main_agent.level, ' radius: ', main_sim.main_agent.radius, ' angle: ', main_sim.main_agent.angle
main_agent = main_sim.main_agent

# 4. Printing the initial map
main_sim.draw_map()
main_sim.log_map(logfile)

# 5. Retriving memory log info
used_mem_before = psutil.virtual_memory().used

begin_time = time.time()
begin_cpu_time = psutil.cpu_times()

"""
██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗███████╗██████╗   
██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔════╝██╔══██╗  
██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   █████╗  ██████╔╝  
██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══╝  ██╔══██╗  
██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ███████╗██║  ██║  

███████╗███████╗████████╗██╗███╗╚═╝███╗ █████╗═████████╗██╗ ██████╗╝███╗ ╚═██╗
██╔════╝██╔════╝╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
█████╗  ███████╗   ██║   ██║██╔████╔██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
██╔══╝  ╚════██║   ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
███████╗███████║   ██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚══════╝╚══════╝   ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

"""
polynomial_degree = 4
agents_parameter_estimation = []
agents_previous_step_info = []
for i in range(len(main_sim.agents)):
    # train_d = train_data.TrainData(generated_data_number, PF_add_threshold,  train_mode)
    param_estim = parameter_estimation.ParameterEstimation(generated_data_number, PF_add_threshold, train_mode)

    param_estim.estimation_initialisation()

    param_estim.estimation_configuration(type_selection_mode, parameter_estimation_mode, polynomial_degree )
    param_estim.choose_target_state = deepcopy(main_sim)
    # Each element of list have estimation for each selfish agent
    agents_parameter_estimation.append(param_estim)

"""
 ██████╗  █████╗ ███╗   ███╗███████╗    ██████╗ ██╗   ██╗███╗   ██╗
██╔════╝ ██╔══██╗████╗ ████║██╔════╝    ██╔══██╗██║   ██║████╗  ██║
██║  ███╗███████║██╔████╔██║█████╗      ██████╔╝██║   ██║██╔██╗ ██║
██║   ██║██╔══██║██║╚██╔╝██║██╔══╝      ██╔══██╗██║   ██║██║╚██╗██║
╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗    ██║  ██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

"""
time_step = 0
search_tree = None
while simulatorCommonMethods.items_left(main_sim) > 0:

    print 'main run count: ', time_step

    # 1. Moving the A agents
    print('****** Movement of Commum agent based on A-Star ****************************************************')
    for i in range(len(main_sim.agents)):
        agents_parameter_estimation[i].previous_agent_status = deepcopy(main_sim.agents[i])
        agents_parameter_estimation[i].previous_state = main_sim

        if simulation_visibility == 'FO':
            main_sim.agents[i] = main_sim.move_a_agent(main_sim.agents[i])
        else:
            main_sim.agents[i] = simulatorCommonMethods.evaluate_a_agent_action(main_sim.agents[i],main_sim)
        print 'agent_next_action: ', main_sim.agents[i].next_action
        print 'target: ', main_sim.agents[i].get_memory()

    print('****** Movement of Intelligent agent based on MCTS ****************************************************')
    if main_sim.main_agent is not None:

        if simulation_visibility == 'FO':
            if not reuseTree:
                main_agent_next_action, search_tree = uct.m_agent_planning(0, None, main_sim, agents_parameter_estimation)
            else:
                main_agent_next_action, search_tree = uct.m_agent_planning(time_step, search_tree, main_sim,
                                                                       agents_parameter_estimation)
            r = uct.do_move(main_sim, main_agent_next_action)

        else:
            if main_sim.main_agent.history != []:
                print 'updating the sampled state'
                pomcp.black_box(main_sim.main_agent.history,main_sim,agents_parameter_estimation)
                print 'updated - real position:', main_sim.main_agent.position

            if not reuseTree:
                main_agent_next_action, next_node = pomcp.m_poagent_planning(main_sim,agents_parameter_estimation)
            else:
                main_agent_next_action, next_node = pomcp.m_poagent_planning(main_sim,agents_parameter_estimation)

            print 'main ag action:',main_agent_next_action
            main_sim.draw_map()
            r = simulatorCommonMethods.do_m_agent_move(main_sim, main_agent_next_action)
            print 'change root'
            next_node.show()
            print 'main history', main_sim.main_agent.history
            next_node = pomcp.pouct.search_history(main_sim.main_agent.history)
            print 'found_node'
            next_node.show()
            pomcp.pouct.search_tree.change_root(next_node)

    if simulation_visibility == 'FO':
        main_sim.update_all_A_agents()
        main_sim.do_collaboration()
    else:
        simulatorCommonMethods.update_all_A_agents(main_sim)
        simulatorCommonMethods.do_collaboration(main_sim)

    '********* Estimation for selfish agents ******'
    if do_estimation:
        for i in range(len(agents_parameter_estimation)):
            p_agent = main_sim.agents[i]

            new_estimated_parameter, x_train = agents_parameter_estimation[i].process_parameter_estimations(time_step,
                                                                                    p_agent.next_action, main_sim)

            a_data_set = np.transpose(np.array(x_train))
            n = time_step
            if a_data_set != []:
                levels = a_data_set[0, :]
                angle = a_data_set[1, :]
                radius = a_data_set[2, :]

                dataMean[n, 0] = np.mean(levels)
                dataStd[n, 0] = np.std(levels, ddof=1)
                dataMean[n, 1] = np.mean(angle)
                dataStd[n, 1] = np.std(angle, ddof=1)
                dataMean[n, 2] = np.mean(radius)
                dataStd[n, 2] = np.std(radius, ddof=1)
            else:
                dataMean[n, 0] = 0
                dataStd[n, 0] = 0
                dataMean[n, 1] = 0
                dataStd[n, 1] = 0

                dataMean[n, 2] = 0
                dataStd[n, 2] = 0

            x_train_set.append(x_train)

    time_step += 1

    print('***********************************************************************************************************')

    main_sim.draw_map()
    main_sim.log_map(logfile)

    left_items = simulatorCommonMethods.items_left(main_sim)
    if left_items == 0:
        break

    print "left items", left_items


"""
██████╗ ███████╗███████╗██╗   ██╗██╗  ████████╗███████╗
██╔══██╗██╔════╝██╔════╝██║   ██║██║  ╚══██╔══╝██╔════╝
██████╔╝█████╗  ███████╗██║   ██║██║     ██║   ███████╗
██╔══██╗██╔══╝  ╚════██║██║   ██║██║     ██║   ╚════██║
██║  ██║███████╗███████║╚██████╔╝███████╗██║   ███████║
╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝   ╚══════╝

"""
plot_errors(time_step,x_train_set)
end_time = time.time()
used_mem_after = psutil.virtual_memory().used
end_cpu_time = psutil.cpu_times()
memory_usage = used_mem_after - used_mem_before