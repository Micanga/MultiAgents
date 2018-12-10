#!/usr/bin/python
import os
import sys
import subprocess
import time

from numpy import pi

# 0. General Settings
number_of_tests = 100

square_grid_size    = ['10','15','20','25']
number_of_agents    = ['1','2','3','4','5']
number_of_items     = ['10','15','20','25']

# 1. Defining the experiment type
experiment_type_set = ['ABU', 'AGA', 'MIN']

# 2. Starting the experiment
test_number = 0
while test_number < number_of_tests:
    for experiment in experiment_type_set:
        for size in square_grid_size:
            for nagents in number_of_agents:
                for nitems in number_of_items:
                    print '******* STARTING TEST ',test_number,' *******'
                    print '| Experiment: ',experiment
                    print '| Size: ',size
                    print '| N Agents: ',nagents
                    print '| N Items: ',nitems
                    print '**********'

                    # a. generating random scenarios
                    print '| (1) Generating Scenario'
                    scenario_generator = 'python scenario_generator.py ' +\
                        experiment + ' ' + size + ' ' + nagents + ' ' + nitems
                    os.system(scenario_generator)
                    time.sleep(1)

                    # b. openning the target dir
                    print '| (2) Starting the process'
                    filename = 'sim.csv'
                    sub_dir = 'FO_O_' + experiment
                    experiment_dir = "inputs/" + sub_dir + '/'
                    experiment_run = 'python run_world.py '+ experiment_dir

                    print '| >',experiment_run
                    os.system(experiment_run)
                    time.sleep(5)

    test_number += 1
