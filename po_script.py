#!/usr/bin/python
import os
import sys
import subprocess
import time

from numpy import pi

# 0. General Settings
number_of_tests = 100

square_grid_size    = ['10','15']#,'20','25']
number_of_agents    = ['1','2','3']#,'4','5']
number_of_items     = ['10','15']#,'20','25']

radius_set = ['3','5','7']
angle_set = [2*pi]

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
                            ' ' + size + ' ' + nagents + ' ' +\
                            nitems + ' ' + radius + ' ' + str(angle)
                        experiment_dir = os.system(scenario_generator)
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
                            experiment = 'python po_run_world.py '+ experiment_dir

                            print experiment
                            os.system(experiment)
                            time.sleep(5)

    test_number += 1
