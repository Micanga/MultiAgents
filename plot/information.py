import pickle
import numpy as np
import subprocess

from copy import deepcopy
from math import sqrt
from scipy.stats import sem, t
from scipy import mean

LEVEL = 0
ANGLE = 1
RADIUS = 2

class Information:

	def __init__(self,name):
		self.name = name
		self.threshold = 0
		self.plots_dir = ""


		self.TRUE_max_len_hist = 0
		self.AGA_max_len_hist = 0
		self.ABU_max_len_hist = 0
		self.OGE_max_len_hist = 0

		self.AGA_errors = list()
		self.ABU_errors = list()
		self.OGE_errors = list()

		self.AGA_mean_len_hist = 0
		self.ABU_mean_len_hist = 0
		self.OGE_mean_len_hist = 0

		self.AGA_std_len_hist = 0
		self.ABU_std_len_hist = 0
		self.OGE_std_len_hist = 0

		self.AGA_ci_len_hist = 0
		self.ABU_ci_len_hist = 0
		self.OGE_ci_len_hist = 0

		self.TRUE_timeSteps = list()
		self.AGA_timeSteps = list()
		self.ABU_timeSteps = list()
		self.OGE_timeSteps = list()

		self.AGA_estimationHist = list()
		self.ABU_estimationHist = list()
		self.OGE_estimationHist = list()

		self.AGA_typeProbHistory= list()
		self.ABU_typeProbHistory= list()
		self.OGE_typeProbHistory= list()

		self.AGA_trueParameter = list()
		self.ABU_trueParameter = list()
		self.OGE_trueParameter = list()

		self.aga_levels, self.aga_levels_std_dev, self.aga_levels_ci = list(), list(), list()
		self.aga_radius, self.aga_radius_std_dev, self.aga_radius_ci = list(), list(), list()
		self.aga_angles, self.aga_angles_std_dev, self.aga_angles_ci = list(), list(), list()
		
		self.abu_levels, self.abu_levels_std_dev, self.abu_levels_ci = list(), list(), list()
		self.abu_radius, self.abu_radius_std_dev, self.abu_radius_ci = list(), list(), list()
		self.abu_angles, self.abu_angles_std_dev, self.abu_angles_ci = list(), list(), list()
		
		self.OGE_levels, self.OGE_levels_std_dev, self.OGE_levels_ci = list(), list(), list()
		self.OGE_radius, self.OGE_radius_std_dev, self.OGE_radius_ci = list(), list(), list()
		self.OGE_angles, self.OGE_angles_std_dev, self.OGE_angles_ci = list(), list(), list()
		
	@staticmethod
	def calcConfInt(p):

		confidence = 0.99
		data = p

		n = len(data)
		m = mean(data)
		std_err = sem(data)
		h = std_err * t.ppf((1 + confidence) / 2, n - 1)
		return h+m



	@staticmethod
	def is_constant(array):
		for i in range(0,len(array)-2):
			if array[i+1] - array[i] != 0:
				return False
		return True

	def normalise(self):
		max_len = max(self.AGA_max_len_hist,self.ABU_max_len_hist,self.OGE_max_len_hist,self.TRUE_max_len_hist)

		# print 'max_len', max_len

		self.AGA_errors = self.normalise_arrays(max_len,self.AGA_errors)
		self.AGA_typeProbHistory = self.normalise_arrays(max_len,self.AGA_typeProbHistory)

		self.ABU_errors = self.normalise_arrays(max_len,self.ABU_errors)
		self.ABU_typeProbHistory = self.normalise_arrays(max_len,self.ABU_typeProbHistory)

		self.OGE_errors = self.normalise_arrays(max_len,self.OGE_errors)
		self.OGE_typeProbHistory  = self.normalise_arrays(max_len,self.OGE_typeProbHistory)



	def normalise_arrays(self, max_value , errors_list):
		for e_l in errors_list:
			last_value = e_l[-1]
			diff = max_value - len(e_l)
			for i in range(0,diff):
				e_l.append(last_value)
		return errors_list

	def extract(self):	
		global LEVEL, RADIUS, ANGLE
		# print '*** AGA - extracting level, radius and angle info ***'
		self.aga_levels, self.aga_levels_std_dev, self.aga_levels_ci = self.extract_parameter_errors(self.AGA_errors,LEVEL)
		# print 'AGA - levels OK'
		self.aga_radius, self.aga_radius_std_dev, self.aga_radius_ci = self.extract_parameter_errors(self.AGA_errors,RADIUS)
		# print 'AGA - radius OK'
		self.aga_angles, self.aga_angles_std_dev, self.aga_angles_ci = self.extract_parameter_errors(self.AGA_errors,ANGLE)
		# print 'AGA - angles OK'
		self.aga_level_error_mean = np.mean(np.array(self.aga_levels))
		self.aga_angle_error_mean = np.mean(np.array(self.aga_angles))
		self.aga_radius_error_mean = np.mean(np.array(self.aga_radius))
		self.aga_type_probability_mean = np.mean(np.array(self.AGA_typeProbHistory))


		self.aga_level_error_ci = np.std(np.array(self.aga_levels))
		self.aga_angle_error_ci = np.std(np.array(self.aga_angles))
		self.aga_radius_error_ci = np.std(np.array(self.aga_radius))
		self.aga_type_probability_ci = np.std(np.array(self.AGA_typeProbHistory))

		# print '*** ABU - extracting level, radius and angle info ***'
		self.abu_levels, self.abu_levels_std_dev, self.abu_levels_ci = self.extract_parameter_errors(self.ABU_errors,LEVEL)
		# print 'ABU - levels OK'
		self.abu_radius, self.abu_radius_std_dev, self.abu_radius_ci = self.extract_parameter_errors (self.ABU_errors,RADIUS)
		# print 'ABU - radius OK'
		self.abu_angles, self.abu_angles_std_dev, self.abu_angles_ci = self.extract_parameter_errors(self.ABU_errors,ANGLE)
		# print 'ABU - angles OK'


		self.abu_level_error_mean = np.mean(np.array(self.abu_levels))
		self.abu_angle_error_mean = np.mean(np.array(self.abu_angles))
		self.abu_radius_error_mean = np.mean(np.array(self.abu_radius))
		self.abu_type_probability_mean = np.mean(np.array(self.ABU_typeProbHistory))


		self.abu_level_error_ci = np.std(np.array(self.abu_levels))
		self.abu_angle_error_ci= np.std(np.array(self.abu_angles))
		self.abu_radius_error_ci = np.std(np.array(self.abu_radius))
		self.abu_type_probability_ci = np.std(np.array(self.ABU_typeProbHistory))


		# print '*** OGE - extracting level, radius and angle info ***'
		self.OGE_levels, self.OGE_levels_std_dev, self.OGE_levels_ci = self.extract_parameter_errors(self.OGE_errors,LEVEL)
		# print 'OGE - levels OK'
		self.OGE_radius, self.OGE_radius_std_dev, self.OGE_radius_ci = self.extract_parameter_errors(self.OGE_errors,RADIUS)
		# print 'OGE - radius OK'
		self.OGE_angles, self.OGE_angles_std_dev, self.OGE_angles_ci = self.extract_parameter_errors(self.OGE_errors,ANGLE)
		# print 'OGE - angles OK'

		self.oge_level_error_mean = np.mean(np.array(self.OGE_levels))
		self.oge_angle_error_mean = np.mean(np.array(self.OGE_angles))
		self.oge_radius_error_mean = np.mean(np.array(self.OGE_radius))
		self.oge_type_probability_mean = np.mean(np.array(self.OGE_typeProbHistory))
		self.oge_level_error_ci = np.std(np.array(self.OGE_levels))
		self.oge_angle_error_ci = np.std(np.array(self.OGE_angles))
		self.oge_radius_error_ci = np.std(np.array(self.OGE_radius))
		self.oge_type_probability_ci = np.std(np.array(self.OGE_typeProbHistory))
		

	def extract_parameter_errors(self,main_error,parameter):
		error_histories = deepcopy(main_error)

		# 1. Errors
		# a. extracting the parameter history
		parameter_history = []
		for error_history in error_histories:
			error = []
			for e in error_history:
				error.append(e[parameter])
			parameter_history.append(error)

		# b. normalizing
		errors=np.array(parameter_history)
		errors=errors.mean(axis=0).tolist()

		# # 2. Standard Deviation
		# # b. extracting the std dev
		std_dev_hist = []
		for error_history in error_histories:
			error = []
			for i in range(len(error_history)):
				error.append((errors[i]-error_history[i][ parameter ])**2)
			std_dev_hist.append(error)
		#
		std_dev=np.array(std_dev_hist)
		std_dev=std_dev.mean(axis=0).tolist()

		for i in range(len(std_dev)):
			std_dev[i] = sqrt(std_dev[i])

		# # 3. Confidence Interval
		ci_hist = []
		for error_history in error_histories:
			ci = []
			for e_h in error_history:
				ci.append(e_h[ parameter ])
			ci_hist.append(ci)
		#
		conf_int = np.zeros(len(ci_hist[0]))
		ci_hist=np.array(ci_hist)

		for i in range(len(conf_int)):
			if not self.is_constant(ci_hist[:,i]):
				conf_int[i] = self.calcConfInt(ci_hist[:,i])
			else:
				conf_int[i] = 0

		return errors, std_dev, conf_int

	def significant_difference(self,p,q):
		f = open("tmp.R", "w")
		f.write("#!/usr/bin/Rscript\n")

		listStr = ""

		for n in p:
			listStr = listStr + str(n) + ","

		listStr1 = ""
		for n in q:
			listStr1 = listStr1 + str(n) + ","

		f.write("print(t.test(c(" + listStr[:-1] + "),c(" + listStr1[:-1] + ")))")


		f.close()

		# os.system("chmod +x ./tmp.R")
		output = subprocess.check_output(['Rscript', 'tmp.R'], stderr=subprocess.STDOUT, shell=False)


		print output