import pickle
import matplotlib.pyplot as plt
import ast
import os
import numpy as np
import subprocess

from copy import deepcopy
from math import sqrt

LEVEL = 0
ANGLE = 1
RADIUS = 2

class Information:

	def __init__(self,name):
		self.name = name
		self.threshold = 0

		self.AGA_max_len_hist = 0
		self.ABU_max_len_hist = 0
		self.PF_max_len_hist = 0

		self.AGA_errors = list()
		self.ABU_errors = list()
		self.PF_errors = list()

		self.AGA_mean_len_hist = 0
		self.ABU_mean_len_hist = 0
		self.PF_mean_len_hist = 0

		self.AGA_std_len_hist = 0
		self.ABU_std_len_hist = 0
		self.PF_std_len_hist = 0

		self.AGA_ci_len_hist = 0
		self.AGA_ci_len_hist = 0
		self.AGA_ci_len_hist = 0

		self.AGA_timeSteps = list()
		self.ABU_timeSteps = list()
		self.PF_timeSteps = list()

		self.AGA_estimationHist = list()
		self.ABU_estimationHist = list()
		self.PF_estimationHist = list()

		self.AGA_typeProbHistory= list()
		self.ABU_typeProbHistory= list()
		self.PF_typeProbHistory= list()

		self.AGA_trueParameter = list()
		self.ABU_trueParameter = list()
		self.PF_trueParameter = list()

		self.aga_levels, self.aga_levels_std_dev, self.aga_levels_ci = list(), list(), list()
		self.aga_radius, self.aga_radius_std_dev, self.aga_radius_ci = list(), list(), list()
		self.aga_angles, self.aga_angles_std_dev, self.aga_angles_ci = list(), list(), list()
		
		self.abu_levels, self.abu_levels_std_dev, self.abu_levels_ci = list(), list(), list()
		self.abu_radius, self.abu_radius_std_dev, self.abu_radius_ci = list(), list(), list()
		self.abu_angles, self.abu_angles_std_dev, self.abu_angles_ci = list(), list(), list()
		
		self.pf_levels, self.pf_levels_std_dev, self.pf_levels_ci = list(), list(), list()
		self.pf_radius, self.pf_radius_std_dev, self.pf_radius_ci = list(), list(), list()
		self.pf_angles, self.pf_angles_std_dev, self.pf_angles_ci = list(), list(), list()
		
	@staticmethod
	def calcConfInt(p):
		f = open("tmp.R","w")
		f.write("#!/usr/bin/Rscript\n")

		listStr = ""

		for n in p:
			listStr = listStr + str(n) + ","

		f.write("print(t.test(c("+listStr[:-1]+"),conf.level=0.90))")

		f.close()

		# os.system("chmod +x ./tmp.R")
		output = subprocess.check_output(['Rscript', 'tmp.R'], stderr=subprocess.STDOUT, shell=False)

		# output = subprocess.check_output("./tmp.R",stderr=subprocess.STDOUT,shell=True)
		output = output.split()
		#print 'end of function', float(output[-7])
		return float(output[-7])

	@staticmethod
	def is_constant(array):
		for i in range(0,len(array)-2):
			if array[i+1] - array[i] != 0:
				return False
		return True

	def normalise(self):
		self.AGA_mean_len_hist, self.AGA_std_len_hist, self.AGA_ci_len_hist = self.calc_mean_len_hist(self.AGA_errors,'AGA')
		self.AGA_errors = self.normalise_arrays(self.AGA_max_len_hist,self.AGA_errors)
		self.AGA_typeProbHistory = self.normalise_arrays(self.AGA_max_len_hist,self.AGA_typeProbHistory)
		print '*** AGA data = ',len(self.AGA_errors),'/AGA avg len = ', self.AGA_mean_len_hist,' ***'

		self.ABU_mean_len_hist, self.ABU_std_len_hist, self.ABU_ci_len_hist = self.calc_mean_len_hist(self.ABU_errors,'ABU')
		self.ABU_errors = self.normalise_arrays(self.ABU_max_len_hist,self.ABU_errors)
		self.ABU_typeProbHistory = self.normalise_arrays(self.ABU_max_len_hist,self.ABU_typeProbHistory)
		print '*** ABU data = ',len(self.ABU_errors),'/ABU avg len = ', self.ABU_mean_len_hist, " ***"

		self.PF_mean_len_hist, self.PF_std_len_hist, self.PF_ci_len_hist = self.calc_mean_len_hist(self.PF_errors,'PF')
		self.PF_errors = self.normalise_arrays(self.PF_max_len_hist,self.PF_errors)
		self.PF_typeProbHistory  = self.normalise_arrays(self.PF_max_len_hist,self.PF_typeProbHistory)
		print '*** PF data  = ',len(self.PF_errors),'/PF avg len  = ', self.PF_mean_len_hist,' ***'

	def calc_mean_len_hist(self,errors_list,te):
		lens = []
		for e_l in errors_list:
			lens.append(len(e_l))
		file = open(self.name+te+'Pickle','wb')
		pickle.dump(lens,file)
		file.close() 

		len_sum = 0
		for e_l in errors_list:
			len_sum += len(e_l)
		mean = len_sum/len(errors_list)

		sum_ = 0
		for e_l in errors_list:
			sum_ += (mean-len(e_l))**2
		std_dev = sqrt(sum_/len(errors_list))

		ci_list = []
		for e_l in errors_list:
			ci_list.append(len(e_l))
			if not self.is_constant(ci_list):
				ci = self.calcConfInt(ci_list)
			else:
				ci = 0

		return mean,std_dev,ci

	def normalise_arrays(self, max_value , errors_list):
		for e_l in errors_list:
			last_value = e_l[ - 1]
			diff = max_value - len(e_l)
			for i in range(diff):
				e_l.append(last_value)
		return errors_list

	def extract(self):	
		global LEVEL, RADIUS, ANGLE
		print '*** AGA - extracting level, radius and angle info ***'
		self.aga_levels, self.aga_levels_std_dev, self.aga_levels_ci = self.extract_parameter_errors(self.AGA_errors,LEVEL)
		print 'AGA - levels OK'
		self.aga_radius, self.aga_radius_std_dev, self.aga_radius_ci = self.extract_parameter_errors(self.AGA_errors,RADIUS)
		print 'AGA - radius OK'
		self.aga_angles, self.aga_angles_std_dev, self.aga_angles_ci = self.extract_parameter_errors(self.AGA_errors,ANGLE)
		print 'AGA - angles OK'

		print '*** ABU - extracting level, radius and angle info ***'
		self.abu_levels, self.abu_levels_std_dev, self.abu_levels_ci = self.extract_parameter_errors(self.ABU_errors,LEVEL)
		print 'ABU - levels OK'
		self.abu_radius, self.abu_radius_std_dev, self.abu_radius_ci = self.extract_parameter_errors (self.ABU_errors,RADIUS)
		print 'ABU - radius OK'
		self.abu_angles, self.abu_angles_std_dev, self.abu_angles_ci = self.extract_parameter_errors(self.ABU_errors,ANGLE)
		print 'ABU - angles OK'

		print '*** PF - extracting level, radius and angle info ***'
		self.pf_levels, self.pf_levels_std_dev, self.pf_levels_ci = self.extract_parameter_errors(self.PF_errors,LEVEL)
		print 'PF - levels OK'
		self.pf_radius, self.pf_radius_std_dev, self.pf_radius_ci = self.extract_parameter_errors(self.PF_errors,RADIUS)
		print 'PF - radius OK'
		self.pf_angles, self.pf_angles_std_dev, self.pf_angles_ci = self.extract_parameter_errors(self.PF_errors,ANGLE)
		print 'PF - angles OK'

	def extract_parameter_errors(self,main_error,parameter):
		# 1. Errors
		error_histories = deepcopy(main_error)
		level_error_hist = []
		for error_history in error_histories:
			level_error = []
			for e_h in error_history:
				level_error.append(e_h[ parameter ])
			level_error_hist.append(level_error)

		errors=np.array(level_error_hist)
		errors=errors.mean(axis=0).tolist()

		# 2. Standard Deviation
		level_std_dev_hist = []
		for error_history in error_histories:
			levels_std_dev = []
			for i in range(len(error_history)):
				levels_std_dev.append((errors[i]-error_history[i][ parameter ])**2)
			level_std_dev_hist.append(levels_std_dev)

		std_dev=np.array(level_std_dev_hist)
		std_dev=std_dev.mean(axis=0).tolist()

		for i in range(len(std_dev)):
			std_dev[i] = sqrt(std_dev[i])

		# 3. Confidence Interval
			ci_hist = []
		for error_history in error_histories:
			ci = []
			for e_h in error_history:
				ci.append(e_h[ parameter ])
			ci_hist.append(ci)

		conf_int = np.zeros(len(ci_hist[0]))
		ci_hist=np.array(ci_hist)

		for i in range(len(conf_int)):
			if not self.is_constant(ci_hist[:,i]):
				conf_int[i] = self.calcConfInt(ci_hist[:,i])
			else:
				conf_int[i] = 0

		return errors, std_dev, conf_int