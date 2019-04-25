import numpy as np
import pickle
import matplotlib.pyplot as plt

ROOT_DIRS = ['outputs']  # ['AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP']#,'AAMAS_Outputs_POMCP_FO']
NAMES = ['MCTS']  # ['POMCP','POMCP','POMCP']#,'POMCP_FO']

SIZE = ['10']  # ,'15','20','25']
NAGENTS = ['1' ]#,'2']#,'3','5']
NITEMS = ['10']  # ,'15','20','25']
RADIUS = ['3']
parameter_estimation_mode_set = ['AGA', 'ABU', 'MIN']
type_estimation_mode_set = ['LPTE']#'LPTE',
information = []

algorithms = ["aga", "abu", "oge"]
algorithms_labels = ["AGA", "ABU", "OGE"]
algorithms_symbol = ["v", "v", "o"]

n_agents = ['1', '2']#, '3', '5']
n_size = ['10']#, '20', '30']

error_mean = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))
error_confident_interval = np.zeros((len(n_agents),(len(n_size)), len(parameter_estimation_mode_set)))

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

                    info_item['aga_error_mean'] = (info.aga_level_error_mean +
                                                   info.aga_angle_error_mean +
                                                   info.aga_radius_error_mean)/3

                    info_item['aga_error_ci'] = (info.aga_level_error_ci +
                                                   info.aga_angle_error_ci +
                                                   info.aga_radius_error_ci) / 3

                    info_item['aga_type_probability_mean'] = info.aga_type_probability_mean
                    info_item['aga_time_steps'] = np.mean(np.array(info.AGA_timeSteps))


                    error_mean[na_i][sz_i][0] = info_item['aga_error_mean']
                    error_confident_interval[na_i][sz_i][0] = info_item['aga_error_ci']

                    info_item['abu_error_mean'] = (info.abu_level_error_mean +
                                                   info.abu_angle_error_mean +
                                                   info.abu_radius_error_mean)/3

                    info_item['abu_error_ci'] = (info.abu_level_error_ci +
                                                   info.abu_angle_error_ci +
                                                   info.abu_radius_error_ci) / 3

                    info_item['abu_type_probability_mean'] = info.abu_type_probability_mean
                    info_item['abu_time_steps'] = np.mean(np.array(info.ABU_timeSteps))

                    error_mean[na_i][sz_i][1] = info_item['abu_error_mean']
                    error_confident_interval[na_i][sz_i][1] = info_item['abu_error_ci']

                    info_item['oge_error_mean'] = (info.oge_level_error_mean +
                                                   info.oge_angle_error_mean +
                                                   info.oge_radius_error_mean)/3

                    info_item['oge_error_ci'] = (info.oge_level_error_ci +
                                                 info.oge_angle_error_ci +
                                                 info.oge_radius_error_ci) / 3

                    info_item['oge_type_probability_mean'] = info.oge_type_probability_mean
                    info_item['oge_time_steps'] = np.mean(np.array(info.OGE_timeSteps))

                    error_mean[na_i][sz_i][2] = info_item['oge_error_mean']
                    error_confident_interval[na_i][sz_i][2] = info_item['oge_error_ci']

                    file.close()


def plot_multiple_agents():
    print error_mean
   # print  error_mean[ 0,0,0]
    #rint  error_mean[:, 1]

    plt.figure(figsize=(4, 3.0))

    for a in range(len(parameter_estimation_mode_set)):
         print error_confident_interval[:,0, a]
         print error_mean[:,0, a]
         plt.errorbar(n_agents, error_mean[:,0,a]  ,
                       yerr=[m - n for m, n in zip(error_confident_interval[:,0, a], error_mean[:,0, a])],
                       label=algorithms_labels[a], marker=algorithms_symbol[a])
         plt.errorbar(n_agents,error_mean[:,0,a], label=algorithms_labels[a])

    plt.legend(loc=1)

    plt.xlim([0, 5])
    plt.xlabel("Number of Agents")
    plt.ylabel("Number of Iterations")

    plt.savefig("nAgents-15.pdf", bbox_inches='tight')


    plt.show()

plot_multiple_agents()