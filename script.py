#!/usr/bin/python
import os
import sys
import subprocess
import time
from numpy import pi

# 0. General Settings
map_count = 0
number_of_tests = 1

if len(sys.argv) > 1:
    size = sys.argv[1]
else:
    size = '10'
 
if len(sys.argv) > 2:
    nagents = sys.argv[2]
else:
    nagents = ""

if len(sys.argv) > 3:
    nitems = sys.argv[3]
else:
    nitems = '10'


# 1. Defining the experiment type# 1. Defining the experiment type
experiment_type_set = ['MIN','ABU', 'AGA']
type_estimation_mode_set = ['BPTE']

# 2. Starting the experiment
for test_muber in range (number_of_tests):
    
# a. generating random scenarios
    print '- Generating Scenario'
    scenario_generator = 'LD_PRELOAD=/usr/shared_apps/packages/anaconda2-2.5.0/lib/libmkl_core.so /usr/shared_apps/packages/anaconda2-2.5.0/bin/python scenario_generator.py ' +\
        ' ' + size + ' ' + nagents + ' ' + nitems + ' ' + str(map_count) + ' ' + tem
    experiment_dir = os.system(scenario_generator)
    map_count += 1
    time.sleep(1)

    for experiment in experiment_type_set:
        print '----- STARTING TEST ',test_number,' -----'
        print '| Experiment: ',experiment
        print '| Size: ',size
        print '| N Agents: ',nagents
        print '| N Items: ',nitems
        print '---------'

        # b. openning the target dir
        print '- Starting the process'
        sub_dir = 'FO_O_' + experiment
        experiment_dir = "inputs/" + sub_dir +'/'
        filename = 'outputs/'
        experiment_run = 'LD_PRELOAD=/usr/shared_apps/packages/anaconda2-2.5.0/lib/libmkl_core.so /usr/shared_apps/packages/anaconda2-2.5.0/bin/python run_world.py '+ experiment_dir + ' ' + filename
 
        os.system(experiment_run)
        time.sleep(5)

