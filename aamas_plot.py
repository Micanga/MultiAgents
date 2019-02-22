import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import subprocess
from math import sqrt

import aamas_plot_init as init
from aamas_plot_statistics import is_constant, calcConfInt

# 0. Variables
count = 0
fig_count = 0
results = list()
informations = list()

# 1. Defining the Graph Generation Parameters
ROOT_DIRS = [
    'po_outputs']  # ['AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP']#,'AAMAS_Outputs_POMCP_FO']
NAMES = ['POMCP']  # ['POMCP','POMCP','POMCP']#,'POMCP_FO']

SIZE = ['10']  # ,'15','20','25']
NAGENTS = ['5']  # ,'2','3','4','5']
NITEMS = ['10']  # ,'15','20','25']
RADIUS = ['3.0', '5.0', '7.0']


############################################################################################
########  ##		#######   ########  ######
##	  ##  ##	   ##	  ##	 ##	   ##	 ##
##	  ##  ##	   ##	  ##	 ##	   ##
########  ##	   ##	  ##	 ##	    ######
##		  ##	   ##	  ##	 ##		     ##
##		  ##	   ##	  ##	 ##	   ##    ##
##		  ########  #######	     ##	    ######
############################# ###############################################################
def plot_type_probability(aga_tp, abu_tp, pf_tp, threshold, plotname):
    aga_tp = np.array(aga_tp)
    abu_tp = np.array(abu_tp)
    pf_tp = np.array(pf_tp)

    # 1. Setting the figure
    global fig_count
    fig = plt.figure(fig_count, figsize=(6.4, 2.4))
    fig_count += 1

    # 2. Normalizing TP
    aga_tp = np.array(aga_tp)
    aga_error = aga_tp.mean(axis=0)  # .tolist()
    # for pf in abu_tp:
    # 	print len(pf)
    abu_tp = np.array(abu_tp)
    abu_error = abu_tp.mean(axis=0)  # .tolist()
    for pf in pf_tp:
        print len(pf)
    pf_tp = np.array(pf_tp)
    pf_error = pf_tp.mean(axis=0)  # .tolist()

    aga_error = np.array([aga_error[t] for t in range(threshold)])
    abu_error = np.array([abu_error[t] for t in range(threshold)])
    pf_error = np.array([pf_error[t]  for t in range(threshold)])

    # 3. Plotting
    plt.plot(aga_error,
             label='AGA',
             color='#3F5D7D',
             linestyle='-',
             linewidth=2)

    plt.plot(abu_error,
             label='ABU',
             color='#37AA9C',
             linestyle='-',
             linewidth=2)

    plt.plot(pf_error,
             label='PF',
             color='#F66095',
             linestyle='-',
             linewidth=2)

    # 4. Saving the result
    axis = plt.gca()
    axis.set_ylabel('True Type Estimation', fontsize='x-large')
    axis.set_xlabel('Number of Iterations', fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    axis.legend(loc="upper center", fontsize='large', \
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1, \
                fancybox=True, framealpha=0.8, ncol=3)
    plt.savefig("./plots/" + plotname + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_run_length_bar(aga_m, aga_s, abu_m, abu_s, pf_m, pf_s, plotname):
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

    # PF
    pf = axis.bar(3, height=pf_m , width=bar_w, yerr=pf_s - pf_m, color='#F66095')

    # b. getting the current axis to label
    axis.set_ylabel('Number of Iterations')
    axis.set_xticks([1, 2, 3])
    axis.set_xticklabels(['AGA', 'ABU', 'PF'])

    # 5. Saving the result
    plt.savefig("./plots/" + plotname + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_summarised(aga, aga_std, aga_ci,
                    abu, abu_std, abu_ci, pf, pf_std, pf_ci,
                    threshold, plotname, show_errorbar=False, show_confint=False):
    # 1. Setting the figure
    global fig_count
    fig_w, fig_h = 6.4, 2.4
    fig = plt.figure(fig_count, figsize=(6.4, 2.4))
    fig_count += 1

    # 2. Normalizing for plot
    x = [t for t in range(threshold)]
    print threshold, len(aga),len(abu),len(pf)

    plot_aga = np.array([aga[t] for t in range(threshold)])
    plot_abu = np.array([abu[t] for t in range(threshold)])
    plot_pf = np.array([pf[t] for t in range(threshold)])

    plot_aga_ci = np.array([aga_ci[t] for t in range(threshold)])
    plot_abu_ci = np.array([abu_ci[t] for t in range(threshold)])
    plot_pf_ci = np.array([pf_ci[t] for t in range(threshold)])

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
        delta = (plot_pf_ci - plot_pf)
        plt.fill_between(x, plot_pf - delta,
                         plot_pf + delta,
                         color='#F66095', alpha=.15)

    # 4. Plotting the main lines
    plt.rcParams["figure.figsize"] = (fig_w, fig_h)
    plt.plot(plot_aga,
             label='AGA',
             color='#3F5D7D',
             linestyle='-',
             linewidth=2,
             clip_on=False)

    plt.plot(plot_abu,
             label='ABU',
             color='#37AA9C',
             linestyle='-',
             linewidth=2,
             clip_on=False)

    plt.plot(plot_pf,
             label='PF',
             color='#F66095',
             linestyle='-',
             linewidth=2,
             clip_on=False)

    # 5. Saving the result
    axis = plt.gca()
    axis.set_ylabel('Error', fontsize='x-large')
    axis.set_xlabel('Number of Iterations', fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    axis.legend(loc="upper center", fontsize='large', \
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1, \
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
                    for ra in RADIUS:
                        filename = 'POMCP_s' + sz + '_a' + na + '_i' + ni + '_r' + ra + '_Pickle'
                        if not os.path.exists(filename):
                            results.append(init.read_files(root, sz, na, ni, ra))
                            info = init.extract_information(results[-1],
                                                            'POMCP_s' + sz + '_a' + na + '_i' + ni + '_r' + ra)
                            print info.name
                            info.normalise()
                            # info.extract()
                            info.threshold = min([info.AGA_max_len_hist, info.ABU_max_len_hist, info.PF_max_len_hist])
                            informations.append(info)

                            file = open(filename, 'wb')
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
                    filename = 'MCTS_s' + sz + '_a' + na + '_i' + ni + '_Pickle'
                    if not os.path.exists(filename):
                        results.append(init.read_files(root, sz, na, ni))
                        info = init.extract_information(results[-1], 'MCTS_s' + sz + '_a' + na + '_i' + ni)
                        print info.name
                        info.normalise()
                        info.extract()
                        info.threshold = min([info.AGA_max_len_hist, info.ABU_max_len_hist, info.PF_max_len_hist])
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
for info in informations:
    print info.name, 'Level'
    plot_summarised(info.aga_levels, info.aga_levels_std_dev, info.aga_levels_ci, \
                    info.abu_levels, info.abu_levels_std_dev, info.abu_levels_ci, \
                    info.pf_levels, info.pf_levels_std_dev, info.pf_levels_ci, \
                    info.threshold, info.name + '_Level', False, True)
    print info.name, 'Radius'
    plot_summarised(info.aga_radius, info.aga_radius_std_dev, info.aga_radius_ci, \
                    info.abu_radius, info.abu_radius_std_dev, info.abu_radius_ci, \
                    info.pf_radius, info.pf_radius_std_dev, info.pf_radius_ci, \
                    info.threshold, info.name + '_Radius', False, True)
    print info.name, 'Angle'
    plot_summarised(info.aga_angles, info.aga_angles_std_dev, info.aga_angles_ci, \
                    info.abu_angles, info.abu_angles_std_dev, info.abu_angles_ci, \
                    info.pf_angles, info.pf_angles_std_dev, info.pf_angles_ci, \
                    info.threshold, info.name + '_Angle', False, True)

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
    general_abu_std_dev = np.array([info.abu_levels_std_dev[i] + info.abu_radius_std_dev[i] \
                                    + info.abu_angles_std_dev[i] for i in range(info.threshold)]) / 3
    general_abu_ci = np.array([info.abu_levels_ci[i] + info.abu_radius_ci[i] \
                               + info.abu_angles_ci[i] for i in range(info.threshold)]) / 3

    general_pf = np.array([info.pf_levels[i] + info.pf_radius[i] \
                           + info.pf_angles[i] for i in range(info.threshold)]) / 3
    general_pf_std_dev = np.array([info.pf_levels_std_dev[i] + info.pf_radius_std_dev[i] \
                                   + info.pf_angles_std_dev[i] for i in range(info.threshold)]) / 3
    general_pf_ci = np.array([info.pf_levels_ci[i] + info.pf_radius_ci[i] \
                              + info.pf_angles_ci[i] for i in range(info.threshold)]) / 3

    plot_summarised(general_aga, general_aga_std_dev, general_aga_ci, \
                    general_abu, general_abu_std_dev, general_abu_ci, \
                    general_pf, general_pf_std_dev, general_pf_ci, \
                    info.threshold, info.name + '_General', False, True)

# 4. Plotting the mean run length
print '***** plotting history len performance *****'
for info in informations:
    aga_m, aga_ci = list(), list()
    abu_m, abu_ci = list(), list()
    pf_m, pf_ci = list(), list()

    print 'AGA', info.AGA_mean_len_hist
    print 'ABU', info.ABU_mean_len_hist
    print 'PF', info.PF_mean_len_hist

    aga_m.append(info.AGA_mean_len_hist)
    aga_ci.append(info.AGA_ci_len_hist)
    abu_m.append(info.ABU_mean_len_hist)
    abu_ci.append(info.ABU_ci_len_hist)
    pf_m.append(info.PF_mean_len_hist)
    pf_ci.append(info.PF_ci_len_hist)

    plot_run_length_bar(info.AGA_mean_len_hist, info.AGA_ci_len_hist, \
                        info.ABU_mean_len_hist, info.ABU_ci_len_hist, \
                        info.PF_mean_len_hist, info.PF_ci_len_hist, 'Perform')

# 5. Plotting the type probability
print '***** type probability performance *****'
for info in informations:
    plot_type_probability(info.AGA_typeProbHistory, \
                          info.ABU_typeProbHistory, \
                          info.PF_typeProbHistory, \
                          info.threshold, info.name + 'TypePerformance')