#!/usr/bin/python
import os
import sys
import subprocess
import time

from numpy import pi

square_grid_size = ['10']
number_of_agents = ['1']
number_of_items = ['10']

# 2. Starting the experiment

for size in square_grid_size:
    for nagents in number_of_agents:
    	for nitems in number_of_items:
	 
	    submit_jub = 'source /etc/profile module add anaconda2/2.5.0 qsub -l h_vmem=10G script.py  ' + size + ' ' + nagents + ' ' + nitems 
	    experiment_dir = os.system(submit_jub)	    
	    time.sleep(1)

          



