import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
import subprocess

from math import sqrt
from information import Information

def read_files(root_dir,size,nagents,nitems,radius=None):
    print '***** reading the files *****'
    results = list()
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if 'pickleResults.txt' in files:
            with open(os.path.join(root,'pickleResults.txt'),"r") as pickleFile:
                progress = 100 * float(count/2632.0)
                sys.stdout.write("Progress: %.1f%% | file #%d   \r" % (progress,count) )
                sys.stdout.flush()
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

                trueParameters = agentDictionary['trueParameters']

                estimationDictionary['typeProbHistory'] = typeProbHistory
                estimationDictionary['trueParameters'] = trueParameters
                estimationDictionary['historyParameters'] = historyParameters
                estimationDictionary['path'] = root
                if size == str(simWidth):
                    if nagents == str(agentsCounts):
                        if nitems == str(itemsCounts):
                            if root_dir == 'Outputs_POMCP':
                                if radius == str(systemDetails['mainAgentRadius']):
                                    estimationDictionary['mainAgentRadius'] = str(systemDetails['mainAgentRadius'])
                                    results.append(estimationDictionary)
                            else:
                                results.append(estimationDictionary)
            count += 1
    #import ipdb; ipdb.set_trace()
    progress = 100 * float(count/2632.0)
    sys.stdout.write("Progress: %.1f%% | file #%d      \n" % (progress,count) )
    return results

########################################################################################################################
def extract_information(results,name,radius=None):
    print '***** extracting the information *****'
    info = Information(name)

    for result in results:
        if result['estimationMode'] == 'AGA':
            if len(result['historyParameters']) > info.AGA_max_len_hist:
                info.AGA_max_len_hist = len(result['historyParameters'])

            info.AGA_estimationHist.append(result['historyParameters'])
            info.AGA_typeProbHistory.append(result['typeProbHistory'])
            info.AGA_trueParameter.append( result['trueParameters'])
            info.AGA_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))

        if result['estimationMode'] == 'ABU':
            if len(result['historyParameters']) > info.ABU_max_len_hist:
                info.ABU_max_len_hist = len(result['historyParameters'])

            info.ABU_estimationHist.append(result['historyParameters'])
            info.ABU_typeProbHistory.append(result['typeProbHistory'])
            info.ABU_trueParameter.append(result['trueParameters'])
            info.ABU_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))

        if result['estimationMode'] == 'MIN':
            if radius != None:
                print radius, result['mainAgentRadius'], radius == result['mainAgentRadius']
                if radius == result['mainAgentRadius']:
                    if len(result['historyParameters']) > info.PF_max_len_hist:
                        info.PF_max_len_hist = len(result['historyParameters'])

                    info.PF_typeProbHistory.append(result['typeProbHistory'])
                    info.PF_estimationHist.append(result['historyParameters'])
                    info.PF_trueParameter.append(result['trueParameters'])
                    info.PF_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))
            else:
                if len(result['historyParameters']) > info.PF_max_len_hist:
                        info.PF_max_len_hist = len(result['historyParameters'])

                info.PF_typeProbHistory.append(result['typeProbHistory'])
                info.PF_estimationHist.append(result['historyParameters'])
                info.PF_trueParameter.append(result['trueParameters'])
                info.PF_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))

    return info

########################################################################################################################
def calculate_error(true_parameter, estimated_parameter_history):

    parameter_errors = list()
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