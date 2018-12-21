#!/usr/bin/python
import os
import sys
import subprocess
import time

# 0. General Settings
RADIUS = ['3','5','7']
ANGLES = '2pi'

number_of_tests = 25
interface_gui = False

type_dict = {'PO_M_ABU':0,'PO_M_AGA':1,'PO_M_MIN':2,'PO_O_ABU':3,'PO_O_AGA':4,'PO_O_MIN':5}

# 1. Defining the experiment type
if len(sys.argv) != 2:
    print ':: usage: python po_script.py [experiment type]'
    print ':: argv = ',sys.argv
    print ':: available types: \"PO_M_ABU\", \"PO_M_AGA\", \"PO_M_MIN\"'
    print ':: \"PO_O_ABU\", \"PO_O_AGA\", \"PO_O_MIN\"'
    exit(0)

experiment_type = sys.argv[1]

# 2. Starting the experiment
test_number = 0
while test_number < number_of_tests:
    print '***** STARTING TEST',test_number,'*****'
    # a. generating random scenarios
    print '- Generating Scenario'
    scenario_generator = 'python po_scenario_generator.py ' + RADIUS[test_number % 3] + ' ' + ANGLES
    os.system(scenario_generator)
    time.sleep(1)

    # b. openning the target dir
    print '- Starting the process'
    experiment_dir = "po_inputs/" + experiment_type + '/'
    filename = 'posim' + '.csv'
    experiment = 'python po_run_world.py '+ experiment_dir + ' ' + filename
    print experiment
    os.system(experiment)
    time.sleep(5)

    test_number += 1
