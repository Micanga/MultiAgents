import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import subprocess

from information import Information


results = list()


AGA_max_len_hist = 0
ABU_max_len_hist = 0
PF_max_len_hist = 0

AGA_errors = list()
ABU_errors = list()
PF_errors = list()

AGA_timeSteps = list()
ABU_timeSteps = list()
PF_timeSteps = list()

AGA_estimationHistError = []

AGA_estimationHist = list()
ABU_estimationHist = list()
PF_estimationHist = list()

AGA_typeProbHistory= list()
ABU_typeProbHistory= list()
PF_typeProbHistory= list()

AGA_trueParameter = list()
ABU_trueParameter = list()
PF_trueParameter = list()


########################################################################################################################
def read_files():
    for root, dirs, files in os.walk('outputs'):
       # print root
        if 'pickleResults.txt' in files:
            with open(os.path.join(root,'pickleResults.txt'),"r") as pickleFile:

                estimationDictionary = {}
                dataList = pickle.load(pickleFile)

                data = dataList[1]
                systemDetails = dataList[0]

                # Simulator Information
                simWidth = systemDetails['simWidth']
                simHeight = systemDetails['simHeight']
                agentsCounts = systemDetails['agentsCounts']
                itemsCounts = systemDetails['itemsCounts']
                generatedDataNumber = systemDetails['generatedDataNumber']

                estimationDictionary['typeSelectionMode'] = systemDetails['typeSelectionMode']

                beginTime = systemDetails['beginTime']
                endTime = systemDetails['endTime']

                estimationDictionary['computationalTime'] = int(endTime) - int(beginTime)
                estimationDictionary['estimationMode'] = systemDetails['estimationMode']

                agentDictionary = data[0]
                trueType = agentDictionary['trueType']

                if trueType == 'l1':
                    typeProbHistory = agentDictionary['l1TypeProbHistory']
                    historyParameters = ast.literal_eval(agentDictionary['l1EstimationHistory'])

                elif trueType == 'l2':
                    typeProbHistory = agentDictionary['l2TypeProbHistory']
                    historyParameters = ast.literal_eval(agentDictionary['l2EstimationHistory'])

                elif trueType == 'f1':
                    typeProbHistory = agentDictionary['f1TypeProbHistory']
                    historyParameters = ast.literal_eval(agentDictionary['f1EstimationHistory'])

                else:
                    typeProbHistory = agentDictionary['f2TypeProbHistory']
                    historyParameters = ast.literal_eval(agentDictionary['f2EstimationHistory'])

                estimationDictionary['timeSteps'] = len(typeProbHistory)
                trueParameters = agentDictionary['trueParameters']
                print typeProbHistory
                estimationDictionary['typeProbHistory'] = typeProbHistory
                estimationDictionary['trueParameters'] = trueParameters
                estimationDictionary['historyParameters'] = historyParameters

                results.append(estimationDictionary)

    return results


########################################################################################################################
def plot_ave_error(results):
    levels = []
    angles = []
    radius = []
    for result in results:
        estimated_parameter_history = result['historyParameters']
        true_parameter = result['trueParameters']
        true_level = true_parameter[0]  # level
        true_angle = true_parameter[1]  # angle
        true_radius = true_parameter[2]  # radius

        for pr in estimated_parameter_history:
            levels.append(abs(true_level - pr[0]))
            angles.append(abs(true_angle - pr[1]))
            radius.append(abs(true_radius - pr[2]))


########################################################################################################################
def plot_one_data_set(true_parameter, estimated_parameter_history):

    true_level = true_parameter[0]#level
    true_angle = true_parameter[1]#angle
    true_radius = true_parameter[2]#radius

    levels = []
    angles = []
    radius = []

    for pr in estimated_parameter_history:
        levels.append(abs(true_level - pr[0]))
        print abs(true_angle - pr[1])
        print pr[1]
        angles.append(abs(true_angle - pr[1]))
        radius.append(abs(true_radius - pr[2]))

    fig = plt.figure(1)

    plt.subplot(3, 1, 1)

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


    # fig.savefig("./plots/error_"".jpg")
    plt.show()


########################################################################################################################
def plot_error_AGA():
    global AGA_errors
    AGA_errors = normalise_arrays(AGA_max_len_hist,AGA_errors)

    radius = extract_radius_errors(AGA_errors)
    levels = extract_level_errors(AGA_errors)
    angles = extract_angles_errors(AGA_errors)
    plot_errors(levels, angles, radius,'AGA')


########################################################################################################################
def plot_error_ABU():
    global ABU_errors
    ABU_errors = normalise_arrays(ABU_max_len_hist,ABU_errors)

    radius = extract_radius_errors(ABU_errors)
    levels = extract_level_errors(ABU_errors)
    angles = extract_angles_errors(ABU_errors)
    plot_errors(levels, angles, radius,'ABU')


########################################################################################################################
def plot_error_PF():
    global PF_errors
    print PF_errors
    PF_errors = normalise_arrays(PF_max_len_hist,PF_errors)

    radius = extract_radius_errors(PF_errors)
    levels = extract_level_errors(PF_errors)
    angles = extract_angles_errors(PF_errors)
    plot_errors(levels, angles, radius,'Particle Filter')


########################################################################################################################
def plot_typeProb(plot_type, typeProbHistory, max_len_hist):

    x=type_prob_mean(max_len_hist, typeProbHistory)
    fig = plt.figure(1)
    # x = typeProbHistory
    # for t in x:
    #     print t
    # plt.subplot(3, 1, 1)

    plt.plot([i for i in range(len(x))], x,
             color='cornflowerblue',
             linewidth=1)

    plt.title(plot_type)
    plt.ylabel('Probability of True Type')
    plt.xlabel('iteration')


    plt.show()


########################################################################################################################
def type_prob_mean(max_len_hist,prob_hist):
    prob_hist = normalise_arrays(max_len_hist,prob_hist)

    a = np.array(prob_hist)
    return a.mean(axis=0).tolist()


########################################################################################################################
def plot_errors(levels,angles,radius,plot_type):
    fig = plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.title("Parameters errors for " + plot_type, fontsize=22, weight ='bold')

    plt.plot([i for i in range(len(levels))], levels,
             color='#3F5D7D',
             linewidth=2)

    ax = plt.gca()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel('Level Error')
    ax.legend(loc="upper right", shadow=True, fontsize='x-large')
    plt.subplot(3, 1, 2)

    plt.plot([i for i in range(len(angles))], angles,
             color='#33057D',
              linestyle='-',
             linewidth=2)
    ax = plt.gca()
    ax.set_ylabel('Angle Error')

    plt.subplot(3, 1, 3)

    plt.plot([i for i in range(len(radius))], radius,
             linestyle='-',
             linewidth=2)

    ax = plt.gca()
    ax.set_ylabel('Radius Error')
    ax.set_xlabel('Iteration Numbers')
    fig.tight_layout()
    fig.savefig("./plots/my_error_"+plot_type+".jpg")
    # plt.show()


########################################################################################################################
def extract_information():
    global results
    global AGA_max_len_hist
    global ABU_max_len_hist
    global PF_max_len_hist

    for result in results:

        if result['estimationMode'] == 'AGA':
            if len(result['historyParameters']) > AGA_max_len_hist:
                AGA_max_len_hist = len(result['historyParameters'])
            AGA_timeSteps.append(result['timeSteps'])
            AGA_estimationHist.append(result['historyParameters'])
            AGA_typeProbHistory.append(result['typeProbHistory'])
            AGA_trueParameter.append( result['trueParameters'])
            AGA_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))

        if result['estimationMode'] == 'ABU':
            if len(result['historyParameters']) > ABU_max_len_hist:
                ABU_max_len_hist = len(result['historyParameters'])

            ABU_timeSteps.append(result['timeSteps'])
            ABU_estimationHist.append(result['historyParameters'])
            ABU_typeProbHistory.append(result['typeProbHistory'])
            ABU_trueParameter.append(result['trueParameters'])
            ABU_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))

        if result['estimationMode'] == 'MIN':
            if len(result['historyParameters']) > PF_max_len_hist:
                PF_max_len_hist = len(result['historyParameters'])

            PF_timeSteps.append(result['timeSteps'])
            PF_typeProbHistory.append(result['typeProbHistory'])
            PF_estimationHist.append(result['historyParameters'])
            PF_trueParameter.append(result['trueParameters'])
            PF_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))


########################################################################################################################
def calculate_error(true_parameter, estimated_parameter_history):

    parameter_errors = []
    true_level = true_parameter[0]  # level
    true_angle = true_parameter[1]  # angle
    true_radius = true_parameter[2]  # radius


    for pr in estimated_parameter_history:
        error = []
        error.append(abs(true_level - pr[0]))
        error.append(abs(true_angle - pr[1]))
        error.append(abs(true_radius - pr[2]))
        parameter_errors.append(error)

    return parameter_errors


########################################################################################################################
def normalise_arrays(max_value , errors_list):

    for e_l in errors_list:
        last_value = e_l[ - 1]
        diff = max_value - len(e_l)
        for i in range(diff):
            e_l.append(last_value)
    return errors_list


########################################################################################################################
def extract_level_errors(error_histories):
    level_error_hist = []

    for error_history in error_histories:
        level_error = []
        for e_h in error_history:
            level_error.append(e_h[0])
        level_error_hist.append(level_error)

    a=np.array(level_error_hist)
    return a.mean(axis=0).tolist()


########################################################################################################################
def extract_angles_errors(error_histories):
    angle_error_hist = []

    for error_history in error_histories:
        angle_error = []
        for e_h in error_history:
            angle_error.append(e_h[1])
            angle_error_hist.append(angle_error)

    a=np.array(angle_error_hist)
    return a.mean(axis=0).tolist()

########################################################################################################################
def extract_radius_errors(error_histories):
    radius_error_hist = []

    for error_history in error_histories:
        radius_error = []
        for e_h in error_history:
            radius_error.append(e_h[2])
        radius_error_hist.append(radius_error)

    a = np.array(radius_error_hist)
    return a.mean(axis=0).tolist()


results = read_files()


extract_information()
print PF_typeProbHistory
#
plot_typeProb('AGA', AGA_typeProbHistory, AGA_max_len_hist)
plot_typeProb('ABU', ABU_typeProbHistory, ABU_max_len_hist)
plot_typeProb('PF', PF_typeProbHistory, PF_max_len_hist)
# plot_error_AGA()
# plot_error_ABU()
#plot_error_PF()
