# Python Imports
from collections import defaultdict
from copy import copy,deepcopy
from random import sample
import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
import sys
import time

# My Python Imports
import log
import parameter_estimation
import train_data
import posimulator
import POUCT

# ============= Set Configurations ============
# System Configuration
sys.setrecursionlimit(2000)
memory_usage = 0

# Simulation Configuration
sim_path = None

types = []
type_selection_mode = None

iteration_max = None
max_depth = None

do_estimation = True
train_mode = None
parameter_estimation_mode = None

type_estimation_mode = None
mutation_rate = None

generated_data_number = None
reuse_tree = None

mcts_mode = None
PF_add_threshold = None
PF_del_threshold = None
PF_weight = 0.0
apply_adversary = False


# ============= Set Input/Output ============
if len(sys.argv) > 1:
    input_folder = sys.argv[1]
else:
    input_folder = log.get_input_folder()
output_folder = log.create_output_folder('PO')

# ============= Read Configuration ============
# 1. Reading the sim configuration file
info = defaultdict(list)
with open(input_folder+'poconfig.csv') as info_read:
    for line in info_read:
        data = line.strip().split(',')
        key, val = data[0], data[1:]
        info[key].append(val)

# 2. Getting the parameters
for k, v in info.items():

    if 'types' in k:
        types = [str(v[0][i]).strip() for i in range(len(v[0]))]

    if 'sim_path' in k:
        sim_path = input_folder + str(v[0][0]).strip()

    if 'type_selection_mode' in k:
        type_selection_mode = str(v[0][0]).strip()

    if 'iteration_max' in k:
        iteration_max = int(v[0][0])

    if 'max_depth' in k:
        max_depth = int(v[0][0])

    if 'do_estimation' in k:
        if v[0][0] == 'False':
            do_estimation = False
        else:
            do_estimation = True

    if 'train_mode' in k:
        train_mode = str(v[0][0]).strip()

    if 'parameter_estimation_mode' in k:
        parameter_estimation_mode = str(v[0][0]).strip()

    if 'generated_data_number' in k:
        generated_data_number = int(v[0][0])

    if 'type_estimation_mode' in k:
        type_estimation_mode = str(v[0][0]).strip()

    if 'mutation_rate' in k:
        mutation_rate = float(v[0][0])

    if 'reuseTree' in k:
        reuse_tree = v[0][0]

    if 'mcts_mode' in k:
        mcts_mode = str(v[0][0]).strip()

    if 'PF_add_threshold' in k:
        PF_add_threshold = float(v[0][0])

    if 'PF_del_threshold' in k:
        PF_del_threshold = float(v[0][0])

    if 'PF_weight' in k:
        PF_weight = float(v[0][0])

    if 'apply_adversary' in k:
        if v[0][0] == 'False':
            apply_adversary = False
        else:
            apply_adversary = True

print 'types',types
sim_configuration = {'sim_path':sim_path,\
    'types':types,'type_selection_mode':type_selection_mode,\
    'iteration_max':iteration_max,'max_depth':max_depth,\
    'do_estimation':do_estimation,'train_mode':train_mode,\
    'parameter_estimation_mode':parameter_estimation_mode,\
    'type_estimation_mode':type_estimation_mode,\
    'mutation_rate':mutation_rate,\
    'generated_data_number':generated_data_number,\
    'reuse_tree':reuse_tree,'mcts_mode':mcts_mode,\
    'PF_add_threshold':PF_add_threshold,\
    'PF_del_threshold':PF_del_threshold,\
    'PF_weight':PF_weight,\
    'apply_adversary':apply_adversary}

# ============= Set Simulation and Log File ============
main_sim = posimulator.POSimulator()
main_sim.loader(sim_path)

log_file = log.create_log_file(output_folder + "log.txt")
log_file.write('Grid Size: {} - {} Items - {} Agents - {} Obstacles\n'.\
        format(main_sim.dim_w,len(main_sim.items),len(main_sim.agents),len(main_sim.obstacles)))
log.write_configurations(log_file,sim_configuration)
log_file.write('***** Initial map *****\n')
log.write_map(log_file,main_sim)

# ============= Simulation Initialization ==================
# 1. Log Variables Init
begin_time = time.time()
begin_cpu_time = psutil.cpu_times()
used_mem_before = psutil.virtual_memory().used

# 2. Sim estimation Init
polynomial_degree = 4
agents_parameter_estimation = []
agents_previous_step_info = []

# 3. Ad hoc Agents
if main_sim.main_agent is not None:
    main_agent = main_sim.main_agent
    search_tree = None

    main_sim.main_agent.initialise_visible_agents(main_sim,generated_data_number, PF_add_threshold, train_mode,
                                                  type_selection_mode, parameter_estimation_mode, polynomial_degree,
                                                  apply_adversary,type_estimation_mode,mutation_rate)
    uct = POUCT.POUCT(iteration_max, max_depth, do_estimation, mcts_mode, apply_adversary,enemy=False)
    main_sim.main_agent.initialise_uct(uct)

if apply_adversary:
    enemy_agent = main_sim.enemy_agent
    enemy_search_tree = None
    if main_sim.enemy_agent is not None:
        main_sim.enemy_agent.initialise_visible_agents(main_sim,generated_data_number, PF_add_threshold, train_mode,
                                                       type_selection_mode, parameter_estimation_mode, polynomial_degree,
                                                       apply_adversary,type_estimation_mode,mutation_rate)
        enemy_uct = POUCT.POUCT(iteration_max, max_depth, do_estimation, mcts_mode,apply_adversary, enemy=True )
        main_sim.enemy_agent.initialise_uct(enemy_uct)


for v_a in main_sim.main_agent.visible_agents:
    v_a.choose_target_state = deepcopy(main_sim) #todo: remove deepcopy and add just agents and items location
    v_a.choose_target_pos = v_a.get_position()
    v_a.choose_target_direction = v_a.direction

# ============= Start Simulation ==================
time_step = 0
while main_sim.items_left() > 0:
    progress = 100 * (len(main_sim.items) - main_sim.items_left())/len(main_sim.items)
    sys.stdout.write("Experiment progress: %d%% | step: %d   \r" % (progress,time_step) )
    sys.stdout.flush()

    log_file.write('***** Iteration #'+str(time_step)+' *****\n')

    # 1. Updating Unkown Agents
    if main_sim.main_agent is not None:
        log_file.write('1) Updating Unknown Agents for Main Agent ')
        main_sim.main_agent.previous_state = main_sim.copy()
        main_sim.main_agent.update_unknown_agents(main_sim)
        log_file.write('- OK\n')

    # 2. Move Common Agent
    for i in range(len(main_sim.agents)):
        log_file.write('2) Move Common Agent '+str(i))
        main_sim.agents[i] = main_sim.move_a_agent(main_sim.agents[i])
        main_sim.main_agent.update_unknown_agents(main_sim)
        log_file.write(' - OK\ntarget: '+str(main_sim.agents[i].get_memory())+'\n')

    # 3. Move Main Agent
    if main_sim.main_agent is not None:
        log_file.write('3) Move Main Agent ')
        r,enemy_action_prob,search_tree = main_sim.main_agent.move(reuse_tree, main_sim, search_tree, time_step)
        log_file.write(' - OK\n')

    # 4. Move Adversary
    if main_sim.enemy_agent is not None:
        log_file.write('4) Move Adversary Agent ')
        r, main_action_prob,enemy_search_tree = main_sim.enemy_agent.move(reuse_tree, main_sim, enemy_search_tree, time_step)
        log_file.write(' - OK\n')

    # 5. Updating the Map
    log_file.write('5) Updating Map\n')
    actions = main_sim.update_all_A_agents(False)
    main_sim.do_collaboration()
    main_sim.main_agent.update_unknown_agents_status(main_sim)
    main_sim.draw_map()
    log.write_map(log_file,main_sim)

    # 6. Updating the PO-MCT and the current belief state
    log_file.write('6) Updating the belief state')
    search_tree = uct.update_belief_state(main_sim,search_tree)
    
    if len(main_sim.main_agent.uct.belief_state) > 0:
        current_belief_state = copy((sample(main_sim.main_agent.uct.belief_state,1)[0]).simulator)
    else:
        current_belief_state = None
    log_file.write(' - OK\n')

    # 7. Estimating
    log_file.write('7) Estimating')
    if do_estimation:
        main_sim.main_agent.estimation(time_step,main_sim,enemy_action_prob,\
            types,actions,current_belief_state)
    log_file.write(' - OK\n')

    time_step += 1
    gc.collect()

    if main_sim.items_left() == 0:
        break

    log_file.write("left items: "+str(main_sim.items_left())+'\n')
    log_file.write('*********************\n')
progress = 100 * (len(main_sim.items) - main_sim.items_left())/len(main_sim.items)
sys.stdout.write("Experiment progress: %d%% | step: %d   \n" % (progress,time_step) )
    
# ============= Finish Simulation ==================
end_time = time.time()
used_mem_after = psutil.virtual_memory().used
end_cpu_time = psutil.cpu_times()
memory_usage = used_mem_after - used_mem_before

log.print_result(main_sim,  time_step, begin_time, end_time,\
    mcts_mode, parameter_estimation_mode, type_selection_mode,\
    iteration_max,max_depth, generated_data_number,reuse_tree,\
    PF_add_threshold, PF_weight,\
    type_estimation_mode,mutation_rate ,\
    end_cpu_time, memory_usage,log_file,output_folder,True)
