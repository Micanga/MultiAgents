import ast
import os
import sys
import pickle
from information import Information


def read_files(root_dir,size,nagents, nitems,type_estimation_mode,radius=None):
    print '***** reading the files *****'
    results = list()
    count = 0
    min_time_steps = []
    max_steps = 0
    min_steps = 1000
    for root, dirs, files in os.walk(root_dir):
        if 'pickleResults.txt' in files:
            print root
            with open(os.path.join(root,'pickleResults.txt'),"r") as pickleFile:

                progress = 1 * float(count/1)
                sys.stdout.write("Progress: %.1f%% | file #%d   \r" % (progress,count) )
                sys.stdout.flush()

                dataList = pickle.load(pickleFile)

                # Simulator Information
                systemDetails = dataList[0]
                print systemDetails['parameter_estimation_mode']

                # if systemDetails['round_count'] == 10:
                #     print root
                if 1 == 1:
                    data = dataList[1]
                    for i in range(len(data)):  # Get data for each agent
                        estimationDictionary = {}

                        agentsCounts = systemDetails['agentsCounts']
                        itemsCounts = systemDetails['itemsCounts']

                        generatedDataNumber = systemDetails['generatedDataNumber']

                        estimationDictionary['simWidth'] = systemDetails['simWidth']
                        estimationDictionary['timeSteps'] = systemDetails['timeSteps']
                        estimationDictionary['typeSelectionMode'] = systemDetails['typeSelectionMode']

                        beginTime = systemDetails['beginTime']
                        endTime = systemDetails['endTime']

                        estimationDictionary['computationalTime'] = int(endTime) - int(beginTime)
                        estimationDictionary['parameter_estimation_mode'] = systemDetails['parameter_estimation_mode']



                        agentDictionary = data[i]
                        trueType = agentDictionary['trueType']
                        if trueType in ['l1','l2','f1','f2']:
                            if trueType == 'l1':
                                typeProbHistory = agentDictionary['l1TypeProbHistory']
                                historyParameters = ast.literal_eval(agentDictionary['l1EstimationHistory'])
                            elif trueType == 'l2':
                                typeProbHistory = agentDictionary['l2TypeProbHistory']
                                historyParameters = ast.literal_eval(agentDictionary['l2EstimationHistory'])
                            elif trueType == 'f1':
                                typeProbHistory = agentDictionary['f1TypeProbHistory']
                                historyParameters = ast.literal_eval(agentDictionary['f1EstimationHistory'])
                            elif trueType == 'f2':
                                typeProbHistory = agentDictionary['f2TypeProbHistory']
                                historyParameters = ast.literal_eval(agentDictionary['f2EstimationHistory'])

                            x ={}
                            trueParameters = agentDictionary['trueParameters']
                            estimationDictionary['trueType'] = trueType
                            estimationDictionary['typeProbHistory'] = typeProbHistory
                            estimationDictionary['trueParameters'] = trueParameters
                            estimationDictionary['historyParameters'] = historyParameters
                            estimationDictionary['path'] = root
                            if size == str(systemDetails['simWidth']):

                                if nagents == str(agentsCounts):
                                    if nitems == str(itemsCounts):

                                        if systemDetails['type_estimation_mode'] == type_estimation_mode:

                                            # print estimationDictionary['timeSteps'] , ' ', systemDetails['parameter_estimation_mode']
                                            # print

                                            if systemDetails['parameter_estimation_mode'] == 'AGA':
                                                x['root'] = root
                                                x['step'] = estimationDictionary['timeSteps']
                                                min_time_steps.append(x)
                                                if estimationDictionary['timeSteps'] > max_steps:
                                                    max_steps = estimationDictionary['timeSteps']
                                                if estimationDictionary['timeSteps'] < min_steps:
                                                    min_steps = estimationDictionary['timeSteps']

                                            if root_dir == 'po_outputs':
                                                if radius == str(int(systemDetails['mainAgentRadius'])):
                                                 #   estimationDictionary['mainAgentRadius'] = str(systemDetails['mainAgentRadius'])
                                                    results.append(estimationDictionary)
                                            else:
                                                results.append(estimationDictionary)
            count += 1
    #import ipdb; ipdb.set_trace()
    progress = 1 * float(count/1.0)
    print 'Max: ', max_steps
    print 'Min: ', min_steps
    for m in min_time_steps:
          print m

    sys.stdout.write("Progress: %.1f%% | file #%d      \n" % (progress,count) )
    return results

########################################################################################################################


def extract_information(results,name,radius=None):

    print '***** extracting the information *****'
    info = Information(name)

    for result in results:
        # print len(result['typeProbHistory'])
        # print (len(result['historyParameters']))
        # print info.AGA_max_len_hist
        # print info.ABU_max_len_hist
        # print info.OGE_max_len_hist
        if result['parameter_estimation_mode'] == 'TRUE':
            # print 'true'
            # print 'time step:', result['timeSteps']
            # print result['type_estimation_mode']
            if len(result['typeProbHistory']) > info.TRUE_max_len_hist:
                info.TRUE_max_len_hist = len(result['typeProbHistory'])

            info.TRUE_timeSteps.append(result['timeSteps'])
            # info.TRUE_estimationHist.append(result['historyParameters'])
            # info.TRUE_typeProbHistory.append(result['typeProbHistory'])
            # info.TRUE_trueParameter.append(result['trueParameters'])
            # info.TRUE_errors.append(calculate_error(result['trueParameters'], result['historyParameters']))

        if result['parameter_estimation_mode'] == 'AGA':
            if len(result['typeProbHistory']) > info.AGA_max_len_hist:
                info.AGA_max_len_hist = len(result['typeProbHistory'])

            info.AGA_timeSteps.append(result['timeSteps'])
            info.AGA_estimationHist.append(result['historyParameters'])
            info.AGA_typeProbHistory.append(result['typeProbHistory'])
            info.AGA_trueParameter.append( result['trueParameters'])
            error = calculate_error(result['trueParameters'], result['historyParameters'])
            # print 'AGA', result['path'], error
            info.AGA_errors.append(error)

        if result['parameter_estimation_mode'] == 'ABU':
            if len(result['typeProbHistory']) > info.ABU_max_len_hist:
                info.ABU_max_len_hist = len(result['typeProbHistory'])

            info.ABU_timeSteps.append(result['timeSteps'])
            info.ABU_estimationHist.append(result['historyParameters'])
            info.ABU_typeProbHistory.append(result['typeProbHistory'])
            info.ABU_trueParameter.append(result['trueParameters'])
            error = calculate_error(result['trueParameters'], result['historyParameters'])
            # print 'ABU', result['path'] , error
            info.ABU_errors.append(error)

        if result['parameter_estimation_mode'] == 'MIN':
            if radius != None:
                # print radius, result['mainAgentRadius'], radius == result['mainAgentRadius']
                if radius == result['mainAgentRadius']:
                    if len(result['typeProbHistory']) > info.OGE_max_len_hist:
                        info.OGE_max_len_hist = len(result['typeProbHistory'])

                    info.OGE_timeSteps.append(result['timeSteps'])
                    info.OGE_typeProbHistory.append(result['typeProbHistory'])
                    info.OGE_estimationHist.append(result['historyParameters'])
                    info.OGE_trueParameter.append(result['trueParameters'])
                    error = calculate_error(result['trueParameters'], result['historyParameters'])
                    # print 'OGE', result['path'], error
                    info.OGE_errors.append(error)
            else:
                if len(result['typeProbHistory']) > info.OGE_max_len_hist:
                    info.OGE_max_len_hist = len(result['typeProbHistory'])

                info.OGE_timeSteps.append(result['timeSteps'])
                info.OGE_typeProbHistory.append(result['typeProbHistory'])
                info.OGE_estimationHist.append(result['historyParameters'])
                info.OGE_trueParameter.append(result['trueParameters'])
                error = calculate_error(result['trueParameters'], result['historyParameters'])

                # print 'OGE', result['path'], error
                info.OGE_errors.append(error)

    # print name
    print "number of AGA: ",len(info.AGA_timeSteps)
    print "number of ABU: ", len(info.ABU_timeSteps)
    print "number of OGE: ", len(info.OGE_timeSteps)
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
