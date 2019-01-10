import os
import pickle
from collections import defaultdict
config_path = None

# for dirs in os.listdir("outputs"):
#
#     for files in os.walk("outputs/" + str(dirs)):
#         result_path = str(files[0]) + '/results.txt'
#         with open(result_path) as info_read:
#             dataList = pickle.load(info_read)
for root, dirs, files in os.walk('outputs'):
    if 'pickleResults.txt' in files:
        with open(os.path.join(root,'pickleResults.txt'),"r") as pickleFile:
            dataList = pickle.load(pickleFile)
            print dataList
            data = dataList[1]
            systemDetails = dataList[0]
            agentDictionary = data[0]

        # info = defaultdict(list)
        # with open(result_path) as info_read:
        #     for line in info_read:
        #         data = line.strip().split('\n')
        #         data = str(data).strip().split(':')
        #         key, val = data[0], data[6AGA_O_2:]
        #         info[key].append(val)
        #
        # for k, v in info.items():
        #
        #
        #     if 'type selection mode' in k:
        #         type_selection_mode = str(v[0][0]).strip()
        #
        #     if 'parameter estimation mode' in k:
        #         parameter_estimation_mode = str(v[0][0]).strip()
        #
        #     if 'generated data number' in k:
        #         generated_data_number = int(v[0][0])
        #
        #     if 'reuseTree' in k:
        #         reuseTree = v[0][0]
        #
        #     if 'iteration max' in k:
        #         iteration_max = int(v[0][0])
        #
        #     if 'max depth' in k:
        #         max_depth = int(v[0][0])
        #
        #
        #     if 'mcts mode' in k:
        #         mcts_mode = str(v[0][0]).strip()
        #
        #
        # print type_selection_mode
        # print(parameter_estimation_mode)
        # print generated_data_number
        # # os.system('python run_world.py ' + config_path)
