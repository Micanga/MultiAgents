
import sys
import os
import time

config_path = None

# for dirs in os.listdir("inputs/test"):
#
#     for files in os.walk(""
#                          ""
#                          "inputs/test/" + str(dirs)):
#         config_path = str(files[0]) + '/'
#
#         print config_path
#         os.system('python run_world.py ' + config_path)
#                       # + str(files[2][0])

number_of_tests = 5

test_number = 0
while test_number < number_of_tests:

    # b. openning the target dir
    print '- Starting the process'
    experiment_dir = 'inputs/test/'
    filename = 'test_output/'
    experiment_run = 'python run_world.py '+ experiment_dir + ' ' + filename
    print experiment_run

    os.system(experiment_run)
    time.sleep(5)

    test_number += 1
