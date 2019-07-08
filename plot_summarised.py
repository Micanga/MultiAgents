import numpy as np
import pickle
import matplotlib.pyplot as plt

ROOT_DIRS = ['outputs']  # ['AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP']#,'AAMAS_Outputs_POMCP_FO']
NAMES = ['MCTS']  # ['POMCP','POMCP','POMCP']#,'POMCP_FO']

SIZE = ['30']  # ,'15','20','25']
NAGENTS = ['3' ]#,'2']#,'3','5']
NITEMS = ['30']  # ,'15','20','25']
RADIUS = ['3']
parameter_estimation_mode_set = ['AGA', 'ABU', 'MIN']
type_estimation_mode_set = ['BPTE']#'LPTE',
information = []

algorithms = ["aga", "abu", "oge"]
algorithms_labels = ["AGA", "ABU", "OGE"]
algorithms_symbol = ["^", "v", "o"]
algorithm_colors=["#3F5D7D","#37AA9C","#F66095"]

n_agents = ['1','3','5','7']#, '3', '5']
n_size = ['10']
size = '10'
agent = '5'
error_mean = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))
error_confident_interval = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))

type_mean = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))
type_confident_interval = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))

performance_mean = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))
performance_confident_interval = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))

def calc_CI(type_hist):
    ci_hist = []
    for th in type_hist:
        ci = []
        # for e_h in th:
        #     ci.append(e_h)
        ci_hist.append(th)

    conf_int = np.zeros(len(ci_hist))
    ci_hist = np.array(ci_hist)


    conf_int = info.calcConfInt(ci_hist)

    return conf_int

def mean(a):
    return sum(a)/len(a)

for root in ROOT_DIRS:
    if root == 'po_outputs':
        for sz in n_size:
            for na in n_agents:
                for ni in NITEMS:
                    for ra in RADIUS:
                        filename = 'POMCP_s' + sz + '_a' + na + '_i' + ni + '_r' + ra + '_Pickle'
                        file = open(filename, 'r')
                        info = pickle.load(file)
                        information.append(info)
                        file.close()
    else:
        for sz_i in range(len(n_size)):
            for na_i in range(len(n_agents)):
                for tem_i in range(len(type_estimation_mode_set)):
                    sz = n_size[sz_i]
                    na = n_agents[na_i]
                    tem = type_estimation_mode_set[tem_i]
                    ni = sz
                    filename = 'MCTS_s' + sz + '_a' + na + '_i' + ni + '_t' + tem + '_Pickle'
                    file = open(filename, 'r')
                    info = pickle.load(file)
                    info_item = {}
                    info_item['observation_type'] = 'FO'
                    info_item['size']= sz
                    info_item['agent_numbers'] = na
                    info_item['item_numbers'] = ni
                    info_item['type_estimation_mode'] = tem

                    info_item['aga_error_mean'] = (mean(info.aga_levels) +
                                                   mean(info.aga_radius) +
                                                   mean(info.aga_angles))/3
                    info_item['aga_error_ci'] = (mean(info.aga_levels_ci) +
                                                   mean(info.aga_radius_ci) +
                                                   mean(info.aga_angles_ci)) / 3
                    error_mean[na_i][sz_i][0] = info_item['aga_error_mean']
                    error_confident_interval[na_i][sz_i][0] = info_item['aga_error_ci']

                    aga_tp = np.array(info.AGA_typeProbHistory)
                    aga_tp = np.array(aga_tp)
                    aga_tp = aga_tp.mean(axis=0)
                    info_item['aga_type_probability_error'] = np.array([1-aga_tp[t] for t in range(info.threshold)])
                    type_mean[na_i][sz_i][0] = mean(info_item['aga_type_probability_error'])
                    type_confident_interval[na_i][sz_i][0] = calc_CI(info_item['aga_type_probability_error'])

                    info_item['aga_time_steps'] = np.mean(np.array(info.AGA_timeSteps))
                    performance_mean[na_i][sz_i][0]= info_item['aga_time_steps']
                    performance_confident_interval[na_i][sz_i][0]=  calc_CI(info.AGA_timeSteps)

                    info_item['abu_error_mean'] = (mean(info.abu_levels) + mean(info.abu_radius)
                                                + mean(info.abu_angles))/3

                    info_item['abu_error_ci'] = (mean(info.abu_levels_ci) + mean(info.abu_radius_ci)
                               + mean(info.abu_angles_ci)) / 3

                    error_mean[na_i][sz_i][1] = info_item['abu_error_mean']
                    error_confident_interval[na_i][sz_i][1] = info_item['abu_error_ci']

                    abu_tp = np.array(info.ABU_typeProbHistory)
                    abu_tp = np.array(abu_tp)
                    abu_tp = abu_tp.mean(axis=0)
                    info_item['abu_type_probability_error'] = np.array([1-abu_tp[t] for t in range(info.threshold)])
                    type_mean[na_i][sz_i][1] = mean(info_item['abu_type_probability_error'])
                    type_confident_interval[na_i][sz_i][1] = calc_CI(info_item['abu_type_probability_error'])

                    info_item['abu_time_steps'] = np.mean(np.array(info.ABU_timeSteps))
                    performance_mean[na_i][sz_i][1]= info_item['abu_time_steps']
                    performance_confident_interval[na_i][sz_i][1]=  calc_CI(info.ABU_timeSteps)

                    info_item['oge_error_mean'] = (mean(info.OGE_levels) + mean(info.OGE_radius)
                           + mean(info.OGE_angles))/3

                    info_item['oge_error_ci'] = (mean(info.OGE_levels_ci) + mean(info.OGE_radius_ci)
                              + mean(info.OGE_angles_ci)) / 3
                    error_mean[na_i][sz_i][2] = info_item['oge_error_mean']
                    error_confident_interval[na_i][sz_i][2] = info_item['oge_error_ci']

                    oge_tp = np.array(info.OGE_typeProbHistory)
                    oge_tp = np.array(oge_tp)
                    oge_tp = oge_tp.mean(axis=0)
                    info_item['oge_type_probability_error'] = np.array([1-oge_tp[t] for t in range(info.threshold)])
                    type_mean[na_i][sz_i][2] = mean(info_item['oge_type_probability_error'])
                    type_confident_interval[na_i][sz_i][2]= calc_CI(info_item['oge_type_probability_error'])

                    info_item['oge_time_steps'] = np.mean(np.array(info.OGE_timeSteps))
                    performance_mean[na_i][sz_i][2]= info_item['oge_time_steps']-1
                    performance_confident_interval[na_i][sz_i][2]=  calc_CI(info.OGE_timeSteps)-1

                    file.close()


def plot_multiple_agents_performance():

    plt.figure(figsize=(4, 3.0))

    for a in range(len(parameter_estimation_mode_set)):

        plt.errorbar(n_agents, performance_mean[:,0,a],
                       yerr=[m - n for m, n in zip(performance_confident_interval[:,0, a],
                                                   performance_mean[:,0, a])],
                       label=algorithms_labels[a], marker=algorithms_symbol[a],color = algorithm_colors[a])

    plt.legend(loc=0)
    plt.ylim([17, 78])
    plt.xlim([0,8])
    plt.xlabel("Number of Agents")
    plt.ylabel("Number of Iterations")

    plt.savefig("./summarized_plots/performance_multiple_agents" + size + ".pdf", bbox_inches='tight')

    plt.show()


def plot_multiple_agents_error():

    plt.figure(figsize=(4, 3.0))

    for a in range(len(parameter_estimation_mode_set)):

        plt.errorbar(n_agents , error_mean[:,0,a],
                       yerr=[m - n for m, n in zip(error_confident_interval[:,0, a],
                                                   error_mean[:,0, a])],
                       label=algorithms_labels[a], marker=algorithms_symbol[a],color = algorithm_colors[a])

    plt.legend(loc=0)

    plt.xlim([0,8])
    plt.xlabel("Number of Agents")
    plt.ylabel("Parameters Error ")

    plt.savefig("./summarized_plots/error_multiple_agents" + size + ".pdf", bbox_inches='tight')

    plt.show()


def plot_multiple_agents_type():

    plt.figure(figsize=(4, 3.0))



    for a in range(len(parameter_estimation_mode_set)):

        plt.errorbar(n_agents , type_mean[:,0,a]  ,
                       yerr=[m - n for m, n in zip(type_confident_interval[:,0, a],
                                                   type_mean[:,0, a])],
                       label=algorithms_labels[a], marker=algorithms_symbol[a],color = algorithm_colors[a])


    # plt.legend(loc=0)

    plt.xlim([0,8])
    plt.xlabel("Number of Agents")
    plt.ylabel("Type Error ")

    plt.savefig("./summarized_plots/type_multiple_agents" + size + ".pdf", bbox_inches='tight')

    plt.show()


def plot_multiple_sizes():

    plt.figure(figsize=(4, 3.0))

    for a in range(len(parameter_estimation_mode_set)):

        plt.errorbar(n_size, error_mean[0,:,a],
                       yerr=[m - n for m, n in zip(error_confident_interval[0,:, a],
                                                   error_mean[0,:, a])],
                       label=algorithms_labels[a], marker=algorithms_symbol[a],color = algorithm_colors[a])


    plt.legend(loc=0)

    plt.xlim([0,40])
    plt.xlabel("Size of Environment")
    plt.ylabel("Error Value")

    plt.savefig("./summarized_plots/error_multiple_size" + agent + ".pdf", bbox_inches='tight')

    plt.show()

def plot_multiple_performance():

    plt.figure(figsize=(4, 3.0))

    for a in range(len(parameter_estimation_mode_set)):

        plt.errorbar(n_size, performance_mean[0,:,a],
                       yerr=[m - n for m, n in zip(performance_confident_interval[0,:, a],
                                                   performance_mean[0,:, a])],
                       label=algorithms_labels[a], marker=algorithms_symbol[a],color = algorithm_colors[a])


    plt.legend(loc=0)
    plt.ylim([0, 80])
    plt.xlim([0,40])
    plt.xlabel("Size of Environment")
    plt.ylabel("Performance")

    plt.savefig("./summarized_plots/performance_multiple_size" + agent + ".pdf", bbox_inches='tight')

    plt.show()
# plot_multiple_agents_error()
plot_multiple_agents_performance()
# plot_multiple_agents_type()
# plot_multiple_sizes()
# plot_multiple_performance()