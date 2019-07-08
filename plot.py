import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import subprocess
from math import sqrt

import plot_init as init
from plot_statistics import is_constant, calcConfInt

# import matplotlib
# matplotlib.use('Agg')

# 0. Variables
count = 0
fig_count = 0
results = list()
informations = list()

# 1. Defining the Graph Generation Parameters
ROOT_DIRS = ['po_outputs']  # ['AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP']#,'AAMAS_Outputs_POMCP_FO']
# ROOT_DIRS = ['outputs']  # ['AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP']#,'AAMAS_Outputs_POMCP_FO']
# ROOT_DIRS = ['outputs']

# NAMES = ['POMCP']  # ['POMCP','POMCP','POMCP']#,'POMCP_FO']
NAMES = ['MCP']  # ['POMCP','POMCP','POMCP']#,'POMCP_FO']

SIZE = ['10']  # ,'15','20','25']
NAGENTS = ['1']#,'3','5']
NITEMS = ['10']  # ,'15','20','25']
RADIUS = ['7']
experiment_type_set = ['ABU', 'AGA', 'MIN']
type_estimation_mode_set = ['BPTE']


############################################################################################
########  ##		#######   ########  ######
##	  ##  ##	   ##	  ##	 ##	   ##	 ##
##	  ##  ##	   ##	  ##	 ##	   ##
########  ##	   ##	  ##	 ##	    ######
##		  ##	   ##	  ##	 ##		     ##
##		  ##	   ##	  ##	 ##	   ##    ##
##		  ########  #######	     ##	    ######
############################# ###############################################################
def calc_CI(type_hist):
    ci_hist = []
    for th in type_hist:
        ci = []
        for e_h in th:
            ci.append(e_h)
        ci_hist.append(ci)

    conf_int = np.zeros(len(ci_hist[0]))
    ci_hist = np.array(ci_hist)

    for i in range(len(conf_int)):
        if not info.is_constant(ci_hist[:, i]):
            conf_int[i] = info.calcConfInt(ci_hist[:, i])
        else:
            conf_int[i] = 0
    return conf_int

############################# ###############################################################
def plot_type_probability(aga_tp, abu_tp, OGE_tp, threshold, plotname):

    #
    # aga_tp_ci=  calc_CI(aga_tp)
    # abu_tp_ci=  calc_CI(abu_tp)
    # OGE_tp_ci=  calc_CI(OGE_tp)
    aga_tp = np.array(aga_tp)
    abu_tp = np.array(abu_tp)
    OGE_tp = np.array(OGE_tp)

    # 1. Setting the figure
    global fig_count
    fig = plt.figure(fig_count, figsize=(6.4, 2.4))
    fig_count += 1

    # 2. Normalizing TP
    aga_tp = np.array(aga_tp)
    aga_error = aga_tp.mean(axis=0)  # .tolist()
    # for OGE in abu_tp:
    # 	print len(OGE)
    abu_tp = np.array(abu_tp)
    abu_error = abu_tp.mean(axis=0)  # .tolist()
    # for OGE in OGE_tp:
    #     print len(OGE)
    OGE_tp = np.array(OGE_tp)
    OGE_error = OGE_tp.mean(axis=0)  # .tolist()

    aga_error = np.array([1-aga_error[t] for t in range(5,threshold)])
    abu_error = np.array([1-abu_error[t] for t in range(5,threshold)])
    OGE_error = np.array([1-OGE_error[t] for t in range(5,threshold)])
    #
    # plot_aga_ci = np.array([1-aga_tp_ci[t] for t in range(5,threshold)])
    # plot_abu_ci = np.array([1-abu_tp_ci[t] for t in range(5,threshold)])
    # plot_OGE_ci = np.array([1-OGE_tp_ci[t] for t in range(5,threshold)])
    x = [t for t in range(threshold-5)]
    show_confint = True
    if show_confint:
        # delta = (plot_aga_ci - aga_error)
        plt.fill_between(x, aga_error ,#- delta,
                         aga_error, #+ delta,
                         color='#3F5D7D', alpha=.15)
        # delta = (plot_abu_ci - abu_error)
        plt.fill_between(x, abu_error ,#- delta,
                         abu_error, #+ delta,
                         color='#37AA9C', alpha=.15)
        # delta = (plot_OGE_ci - OGE_error)
        plt.fill_between(x, OGE_error ,#- delta,
                         OGE_error ,#+ delta,
                         color='#F66095', alpha=.15)
    # 3. Plotting

    plt.plot(aga_error,
             label='AGA',
             color='#3F5D7D',
             #linestyle='-',
             marker='^',
             markevery=10
             #linewidth=2
              )

    plt.plot(abu_error,
             label='ABU',
             color='#37AA9C',
             #linestyle='--',
             marker='v',
             markevery=10,
             #linewidth=2
             )

    plt.plot(OGE_error,
             label='OGE',
             color='#F66095',
             #linestyle='-.',
             marker ='o',
             markevery=10
             #linewidth=2
             )

    # 4. Saving the result
    axis = plt.gca()
    axis.set_ylabel('Type Error', fontsize='x-large')
    axis.set_xlabel('Number of Iterations', fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    axis.legend(loc="upper center", fontsize='large',
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1,
                fancybox=True, framealpha=0.8, ncol=3)
    plt.savefig("./plots/" + plotname + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# def plot_run_length_bar(true_m, true_s ,aga_m, aga_s, abu_m, abu_s, OGE_m, OGE_s, plotname):
def plot_run_length_bar(aga_m, aga_s, abu_m, abu_s, OGE_m, OGE_s, plotname):
    # 1. Setting the figure
    global fig_count
    fig = plt.figure(fig_count, figsize=(6.4, 2.4))
    fig_count += 1
    bar_w = 0.5

    # 2. Plotting the number of iteration for each run to load items
    # a. defining the main plot
    axis = plt.gca()

    # AGA
    aga = axis.bar(1, height=aga_m, width=bar_w, yerr=aga_s - aga_m, color='#3F5D7D')

    # ABU
    abu = axis.bar(2, height=abu_m, width=bar_w, yerr=abu_s - abu_m, color='#37AA9C')

    # OGE
    OGE = axis.bar(3, height=OGE_m , width=bar_w, yerr=OGE_s - OGE_m, color='#F66095')

    # b. getting the current axis to label
    axis.set_ylabel('Number of Iterations')
    axis.set_xticks([1, 2, 3])
    axis.set_xticklabels(['AGA', 'ABU', 'OGE'])

    # 5. Saving the result
    plt.savefig("./plots/" + plotname + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_run_length_bar_1(aga_m,  abu_m,  OGE_m,  plotname):
    # 1. Setting the figure
    global fig_count
    fig = plt.figure(fig_count, figsize=(6.4, 2.4))
    fig_count += 1
    bar_w = 0.5

    # 2. Plotting the number of iteration for each run to load items
    # a. defining the main plot
    axis = plt.gca()

    # AGA
    aga = axis.bar(1, height=aga_m, width=bar_w, color='#3F5D7D')

    # ABU
    abu = axis.bar(2, height=abu_m, width=bar_w,  color='#37AA9C')

    # OGE
    OGE = axis.bar(3, height=OGE_m , width=bar_w,  color='#F66095')



    # b. getting the current axis to label
    axis.set_ylabel('Number of Iterations')
    axis.set_xticks([1, 2, 3])
    axis.set_xticklabels(['AGA', 'ABU', 'OGE'])

    # 5. Saving the result
    plt.savefig("./plots/" + plotname + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_run_length_bar_true(aga_m,  abu_m,  OGE_m,  plotname):
    # 1. Setting the figure
    global fig_count
    fig = plt.figure(fig_count, figsize=(6.4, 2.4))
    fig_count += 1
    bar_w = 0.5

    # 2. Plotting the number of iteration for each run to load items
    # a. defining the main plot
    axis = plt.gca()

    # AGA
    aga = axis.bar(1, height=aga_m, width=bar_w, color='#3F5D7D')

    # ABU
    abu = axis.bar(2, height=abu_m, width=bar_w,  color='#37AA9C')

    # OGE
    OGE = axis.bar(3, height=OGE_m , width=bar_w,  color='#F66095')

    # TRUE
    true = axis.bar(4, height=true_m, width=bar_w, color='#F66095')

    # b. getting the current axis to label
    axis.set_ylabel('Number of Iterations')
    axis.set_xticks([1, 2, 3,4])
    axis.set_xticklabels(['AGA', 'ABU', 'OGE','TRUE'])

    # 5. Saving the result
    plt.savefig("./plots/" + plotname + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_summarised(aga, aga_std, aga_ci,
                    abu, abu_std, abu_ci, OGE, OGE_std, OGE_ci,
                    threshold, plotname, show_errorbar=False, show_confint=False):

    # 1. Setting the figure
    global fig_count
    fig_w, fig_h = 6.4, 2.4
    fig = plt.figure(fig_count, figsize=(6.4, 2.4))
    fig_count += 1

    # 2. Normalizing for plot
    x = [t for t in range(threshold)]

    plot_aga = np.array([aga[t] for t in range(threshold)])
    plot_abu = np.array([abu[t] for t in range(threshold)])
    plot_OGE = np.array([OGE[t] for t in range(threshold)])

    plot_aga_ci = np.array([aga_ci[t] for t in range(threshold)])
    plot_abu_ci = np.array([abu_ci[t] for t in range(threshold)])
    plot_OGE_ci = np.array([OGE_ci[t] for t in range(threshold)])

    # 3. Plotting the confidence interval
    if show_confint:
        delta = (plot_aga_ci - plot_aga)
        plt.fill_between(x, plot_aga - delta,
                         plot_aga + delta,
                         color='#3F5D7D', alpha=.15)
        delta = (plot_abu_ci - plot_abu)
        plt.fill_between(x, plot_abu - delta,
                         plot_abu + delta,
                         color='#37AA9C', alpha=.15)
        delta = (plot_OGE_ci - plot_OGE)
        plt.fill_between(x, plot_OGE - delta,
                         plot_OGE + delta,
                         color='#F66095', alpha=.15)

    # 4. Plotting the main lines
    plt.rcParams["figure.figsize"] = (fig_w, fig_h)
    plt.plot(plot_aga,
             label='AGA',
             color='#3F5D7D',
             #linestyle='-',
              marker='^',
             markevery=10
             # ,
             #
             #  linewidth=2,
             # clip_on=False
             )

    plt.plot(plot_abu,
             label='ABU',
             color='#37AA9C',
             # linestyle='-',
              marker='v',
             markevery=10,
             # ms=2,
             # linewidth=2,
             # clip_on=False
             #
             )

    plt.plot(plot_OGE,
             label='OGE',
             color='#F66095',
            # linestyle='-',
             marker='o',
             markevery=10

             # linewidth=2,
             )

    # 5. Saving the result
    axis = plt.gca()
    axis.set_ylabel('Error', fontsize='x-large')
    axis.set_xlabel('Number of Iterations', fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    axis.legend(loc="upper center", fontsize='large',
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1,
                fancybox=True, framealpha=0.8, ncol=3)
    plt.savefig("./plots/" + plotname + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


######################################################################
##	   ##	 ###	#### ##	   ##
###   ###   ## ##	 ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ##  ##	 ##  ##  ## ## ##
##	   ## #########  ##  ##  ####
##	   ## ##	 ##  ##  ##   ###
##	   ## ##	 ## #### ##	   ##
######################################################################
# 1. Reading the files and extracting the results
for root in ROOT_DIRS:
    if root == 'po_outputs':
        for sz in SIZE:
            for na in NAGENTS:
                for ni in NITEMS:
                    for tem in type_estimation_mode_set:
                        for ra in RADIUS:
                            filename = 'POMCP_s' + sz + '_a' + na + '_i' + ni + '_t' + tem +'_r' + ra + '_Pickle'
                            if not os.path.exists(filename):
                                results.append(init.read_files(root, sz, na, ni, tem ,ra))
                                info = init.extract_information(results[-1],
                                                                'POMCP_s' + sz + '_a' + na + '_i' + ni + '_t'+tem+ '_r' + ra)

                                info.normalise()
                                info.extract()
                                info.threshold = min([info.AGA_max_len_hist, info.ABU_max_len_hist, info.OGE_max_len_hist])
                                informations.append(info)

                                file = open(filename,'wb')
                                pickle.dump(info, file)
                                file.close()
                            else:
                                file = open(filename, 'r')
                                info = pickle.load(file)
                                informations.append(info)
                                file.close()
    else:
        for sz in SIZE:
            for na in NAGENTS:
                for ni in NITEMS:
                    for tem in type_estimation_mode_set:
                        filename = 'MCTS_s' + sz + '_a' + na + '_i' + ni + '_t' + tem + '_Pickle'
                        if not os.path.exists(filename):
                            results.append(init.read_files(root, sz, na, ni,tem))
                            info = init.extract_information(results[-1], 'MCTS_s' + sz + '_a' + na + '_i' + ni + '_t'+tem)

                            info.normalise()
                            info.extract()
                            info.threshold = min([info.AGA_max_len_hist, info.ABU_max_len_hist, info.OGE_max_len_hist])
                            informations.append(info)

                            file = open(filename, 'wb')
                            pickle.dump(info, file)
                            file.close()
                        else:
                            print filename, 'already exists'
                            file = open(filename, 'r')
                            info = pickle.load(file)
                            informations.append(info)
                            file.close()

# 3. Plotting the Information
print '***** plotting parameters results *****'
# for info in informations:
#     print info.name, 'Level'
#     plot_summarised(info.aga_levels, info.aga_levels_std_dev, info.aga_levels_ci,
#                     info.abu_levels, info.abu_levels_std_dev, info.abu_levels_ci,
#                     info.OGE_levels, info.OGE_levels_std_dev, info.OGE_levels_ci,
#                     info.threshold, info.name + '_Level', False, True)
#     print info.name, 'Radius'
#     plot_summarised(info.aga_radius, info.aga_radius_std_dev, info.aga_radius_ci,
#                     info.abu_radius, info.abu_radius_std_dev, info.abu_radius_ci,
#                     info.OGE_radius, info.OGE_radius_std_dev, info.OGE_radius_ci,
#                     info.threshold, info.name + '_Radius', False, True)
#     print info.name, 'Angle'
#     plot_summarised(info.aga_angles, info.aga_angles_std_dev, info.aga_angles_ci,
#                     info.abu_angles, info.abu_angles_std_dev, info.abu_angles_ci,
#                     info.OGE_angles, info.OGE_angles_std_dev, info.OGE_angles_ci,
#                      info.threshold, info.name + '_Angle', False, True)

print '***** plotting general results *****'
for info in informations:
    general_aga = np.array([info.aga_levels[i] + info.aga_radius[i] \
                            + info.aga_angles[i] for i in range(info.threshold)]) / 3
    general_aga_std_dev = np.array([info.aga_levels_std_dev[i] + info.aga_radius_std_dev[i] \
                                    + info.aga_angles_std_dev[i] for i in range(info.threshold)]) / 3
    general_aga_ci = np.array([info.aga_levels_ci[i] + info.aga_radius_ci[i] \
                               + info.aga_angles_ci[i] for i in range(info.threshold)]) / 3

    general_abu = np.array([info.abu_levels[i] + info.abu_radius[i] \
                            + info.abu_angles[i] for i in range(info.threshold)]) / 3
    general_abu_std_dev = np.array([info.abu_levels_std_dev[i] + info.abu_radius_std_dev[i]\
                                    + info.abu_angles_std_dev[i] for i in range(info.threshold)]) / 3
    general_abu_ci = np.array([info.abu_levels_ci[i] + info.abu_radius_ci[i] \
                               + info.abu_angles_ci[i] for i in range(info.threshold)]) / 3

    general_OGE = np.array([info.OGE_levels[i] + info.OGE_radius[i]\
                           + info.OGE_angles[i] for i in range(info.threshold)]) / 3
    general_OGE_std_dev = np.array([info.OGE_levels_std_dev[i] + info.OGE_radius_std_dev[i]\
                                   + info.OGE_angles_std_dev[i] for i in range(info.threshold)]) / 3
    general_OGE_ci = np.array([info.OGE_levels_ci[i] + info.OGE_radius_ci[i] \
                              + info.OGE_angles_ci[i] for i in range(info.threshold)]) / 3

    plot_summarised(general_aga, general_aga_std_dev, general_aga_ci,
                    general_abu, general_abu_std_dev, general_abu_ci,
                    general_OGE, general_OGE_std_dev, general_OGE_ci,
                    info.threshold, info.name + '_General', False, True)

# 4. Plotting the mean run length
print '***** plotting history len performance *****'
for info in informations:
    aga_m, aga_ci = list(), list()
    abu_m, abu_ci = list(), list()
    OGE_m, OGE_ci = list(), list()

    print 'AGA', info.AGA_mean_len_hist
    print 'ABU', info.ABU_mean_len_hist
    print 'OGE', info.OGE_mean_len_hist

    aga_m.append(info.AGA_mean_len_hist)
    aga_ci.append(info.AGA_ci_len_hist)
    abu_m.append(info.ABU_mean_len_hist)
    abu_ci.append(info.ABU_ci_len_hist)
    OGE_m.append(info.OGE_mean_len_hist)
    OGE_ci.append(info.OGE_ci_len_hist)

    # plot_run_length_bar(info.True_mean_len_hist, info.True_ci_len_hist,
    #                     info.AGA_mean_len_hist, info.AGA_ci_len_hist,
    #                     info.ABU_mean_len_hist, info.ABU_ci_len_hist,
    #                     info.OGE_mean_len_hist, info.OGE_ci_len_hist, info.name+ '_Performance')
    plot_run_length_bar(info.AGA_mean_len_hist, info.AGA_ci_len_hist,
                        info.ABU_mean_len_hist, info.ABU_ci_len_hist,
                        info.OGE_mean_len_hist, info.OGE_ci_len_hist, info.name + '_Performance')

    # plot_run_length_bar_1(
    #                       info.AGA_timeSteps,
    #                       info.ABU_timeSteps,
    #                       info.OGE_timeSteps, info.name + '_Performance')
    # plot_run_length_bar_true(info.TRUE_timeSteps,
    #                     info.AGA_timeSteps,
    #                     info.ABU_timeSteps,
    #                     info.OGE_timeSteps, info.name + '_PerformanceWithTrue')

# 5. Plotting the type probability
print '***** type probability performance *****'
for info in informations:
    plot_type_probability(info.AGA_typeProbHistory,
                          info.ABU_typeProbHistory,
                          info.OGE_typeProbHistory,
                          info.threshold, info.name + 'TypeEstimation')


#print info.significant_difference(info.ABU_timeSteps,info.OGE_timeSteps)