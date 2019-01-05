import datetime
import os
import pickle
from random import randint

def get_input_folder():
    return ""

def create_output_folder():
    # 1. Getting the experiment time
    now = datetime.datetime.now()

    # 2. Defining the sub directory to save the log
    sub_dir = str(now.day) + "_"+ str(now.hour)+ "_" + str(now.minute)\
    + "-" + str(randint(0,now.day+now.hour+now.minute))
    
    # 3. Verifing the current folder
    current_folder = "outputs/" + sub_dir + '/'
    if not os.path.exists(current_folder):
        os.mkdir(current_folder, 0755)
    else:
        return create_output_folder()

    # 4. Returning the created folder
    return current_folder

def create_log_file(path):
    file = open(path, 'w')
    return file

def write_configurations(file,sim_configuration):
    for key in sim_configuration:
        file.write(key+' : '+str(sim_configuration[key])+'\n')

def write_map(file, sim):
    line =''
    for y in range(sim.dim_h - 1, -1, -1):
        for x in range(sim.dim_w):
            xy = sim.the_map[y][x]
            if xy == 0:
                line = line + '.'  # space
            elif xy == 1:
                line = line + 'I'  # Items
            elif xy == 2:
                line = line + 'S'  # start
            elif xy == 3:
                line = line + 'R'  # route
            elif xy == 4:
                line = line + 'D'  # finish
            elif xy == 5:
                line = line + '+'  # Obstacle
            elif xy == 8:
                line = line + 'A'  # A Agent
            elif xy == 9:
                line = line + 'M'  # Main Agent
            elif xy == 10:
                line = line + 'W'  # Enemy Agent

        file.write(line+ '\n')
        line = ''
    file.write('\n')

def print_result(main_sim,  time_steps, begin_time, end_time,mcts_mode, parameter_estimation_mode,\
 type_selection_mode, iteration_max, max_depth, generated_data_number,\
 reuse_tree, PF_add_threshold, PF_weight, end_cpu_time, memory_usage,log_file, current_folder):

    pickleFile = open(current_folder + "/pickleResults.txt", 'wb')
    dataList = []

    # Simulation Information
    systemDetails = {}
    systemDetails['simWidth'] = main_sim.dim_w
    systemDetails['simHeight'] = main_sim.dim_h
    systemDetails['agentsCounts'] = len(main_sim.agents)
    systemDetails['itemsCounts'] = len(main_sim.items)
    systemDetails['timeSteps'] = time_steps
    systemDetails['beginTime'] = begin_time
    systemDetails['endTime'] = end_time
    systemDetails['CPU_Time'] = end_cpu_time
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

    # Agents Information
    agentDictionary = {}
    for i in range(len(main_sim.agents)):
        u_a = main_sim.main_agent.visible_agents[i]
        agentData = {}

        agentData['trueType'] = main_sim.agents[i].agent_type
        trueParameters = [main_sim.agents[i].level,main_sim.agents[i].radius,main_sim.agents[i].angle]
        agentData['trueParameters'] = trueParameters

        agentData['maxProbability'] = u_a.agents_parameter_estimation.get_highest_type_probability()

        l1EstimationHistory = u_a.agents_parameter_estimation.l1_estimation.get_estimation_history()
        agentData['l1EstimationHistory'] = l1EstimationHistory
        agentData['l1TypeProbHistory'] = u_a.agents_parameter_estimation.l1_estimation.type_probabilities

        l2EstimationHistory = u_a.agents_parameter_estimation.l2_estimation.get_estimation_history()
        agentData['l2EstimationHistory'] = l2EstimationHistory
        agentData['l2TypeProbHistory'] = u_a.agents_parameter_estimation.l2_estimation.type_probabilities

        f1EstimationHistory = u_a.agents_parameter_estimation.f1_estimation.get_estimation_history()
        agentData['f1EstimationHistory'] = f1EstimationHistory
        agentData['f1TypeProbHistory'] = u_a.agents_parameter_estimation.f1_estimation.type_probabilities

        f2EstimationHistory = u_a.agents_parameter_estimation.f2_estimation.get_estimation_history()
        agentData['f2EstimationHistory'] = f2EstimationHistory
        agentData['f2TypeProbHistory'] = u_a.agents_parameter_estimation.f2_estimation.type_probabilities

        agentDictionary[i] = agentData

    dataList.append(systemDetails)
    dataList.append(agentDictionary)

    log_file.write("writing to pickle file.\n")
    pickle.dump(dataList,pickleFile)
    log_file.write("writing over\n")
    