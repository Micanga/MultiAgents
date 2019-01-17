#!/usr/bin/python
import os
import sys
import subprocess
import time

from numpy import pi

# 0. General Settings
map_count       = 0
number_of_tests = 1

square_grid_size    = ['10']
number_of_agents    = ['2']#'2','3','5','7','10'
number_of_items     = ['10']#,'20','25']

radius_set          = ['3','5','7']
angle_set           = [2*pi]

# 1. Defining the experiment type
experiment_type_set = ['ABU', 'AGA', 'MIN']

# 2. Starting the experiment
test_number = 0
while test_number < number_of_tests:
    for size in square_grid_size:
        for nagents in number_of_agents:
            for nitems in number_of_items:
                for radius in radius_set:
                    for angle in angle_set:
                        # a. generating random scenarios
                        print '- Generating Scenario'
                        scenario_generator = 'python po_scenario_generator.py ' +\
                            ' ' + size + ' ' + nagents + ' ' + nitems + ' '\
                            + radius + ' ' + str(angle) + ' ' + str(map_count)
                        experiment_dir = os.system(scenario_generator)
                        map_count += 1
                        time.sleep(1)

                        for experiment in experiment_type_set:
                            print '----- STARTING TEST ',test_number,' -----'
                            print '| Experiment: ',experiment
                            print '| Size: ',size
                            print '| N Agents: ',nagents
                            print '| N Items: ',nitems
                            print '| Radius: ',radius
                            print '| Angle: ', angle
                            print '---------'

                            # b. openning the target dir
                            print '- Starting the process'
                            sub_dir = 'PO_O_' + experiment
                            experiment_dir = "po_inputs/" + sub_dir + '/'
                            filename = 'posim.csv'
                            experiment_run = 'python po_run_world.py '+ experiment_dir
                            print experiment_run
                            
                            os.system(experiment_run)
                            time.sleep(5)

    test_number += 1
