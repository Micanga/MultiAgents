#!/usr/bin/python
import os
import sys
import subprocess
import time
import gc
from numpy import pi

# 0. General Settings
map_count = 0
number_of_tests = 50


square_grid_size = ['10']
number_of_agents = ['1']
number_of_items = ['10']#,'20','25']

# 1. Defining the experiment type# 1. Defining the experiment type
experiment_type_set = ['MIN','ABU', 'AGA']

type_estimation_mode_set = ['BPTE']
# 2. Starting the experiment
test_number = 0
while test_number < number_of_tests:
    for size in square_grid_size:
        for nagents in number_of_agents:
            for nitems in number_of_items:
                for tem in type_estimation_mode_set:
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
                        print experiment_run
                        gc.collect()

                        os.system(experiment_run)
                        time.sleep(5)

    test_number += 1
