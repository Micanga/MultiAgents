import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import subprocess
from math import sqrt

import aamas_plot_init as init
from aamas_plot_statistics import is_constant, calcConfInt

# 0. Variables
count 		 = 0
fig_count 	 = 0
results 	 = list()
informations = list()

# 1. Defining the Graph Generation Parameters
ROOT_DIRS 	= ['Outputs_MCTS']#['AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP','AAMAS_Outputs_POMCP']#,'AAMAS_Outputs_POMCP_FO']
NAMES 		= ['MCTS']#['POMCP','POMCP','POMCP']#,'POMCP_FO']
 
SIZE 	    = ['10']#,'15','20','25']
NAGENTS     = ['2']#,'2','3','4','5']
NITEMS      = ['10']#,'15','20','25']
RADIUS 		= ['3.0','5.0','7.0']

############################################################################################
			########  ##		#######   ########  ######  
			##	  ##  ##	   ##	  ##	 ##	   ##	 ## 
			##	  ##  ##	   ##	  ##	 ##	   ##	   
			########  ##	   ##	  ##	 ##	    ######  
			##		  ##	   ##	  ##	 ##		     ## 
			##		  ##	   ##	  ##	 ##	   ##    ## 
			##		  ########  #######	     ##	    ######  
############################################################################################
def plot_errors(levels,lerr,lci,angles,aerr,aci,radius,rerr,rci,plot_type):
	# 0. Layout Settings
	err_marker = 'o'
	err_marker_size = 1
	err_color = 'lightblue'
	err_offset = 5
	err_gap = 7

	# 1. Setting the figure
	fig = plt.figure(1)

	# 2. Plotting the Level error
	# a. defining the main plot
	plt.subplot(3, 1, 1)

	# + Confiance Interval
	x = [i for i in range(len(levels))]
	plt.fill_between(x, levels+(lci/2), levels-(lci/2),
					 color='b', alpha=.25)

	plt.plot(levels,
			 label=plot_type,
			 color='cornflowerblue',
			 linewidth=1)

	# + Standard Deviation
	#for i in range(len(levels)):
	#	if (i+err_offset) % err_gap == 0:
	#		plt.errorbar(x=i,y=levels[i],yerr=lerr[i],\
	#			marker=err_marker,markersize=err_marker_size,\
	#			color=err_color)

	# b. getting the current axis to label
	axis = plt.gca()
	axis.set_ylabel('Level Error')
	axis.legend(loc="upper right", shadow=True, fontsize='x-large')

	# 3. Plotting the Angle error
	# a. defining the main plot
	plt.subplot(3, 1, 2)

	# + Confiance Interval
	x = [i for i in range(len(angles))]
	plt.fill_between(x, angles+(aci/2), angles-(aci/2),
					 color='b', alpha=.25)

	plt.plot(angles, 
			 label='Angle', 
			 linestyle='-', 
			 color='cornflowerblue',
			 linewidth=1)

	# + Standard Deviation
	#for i in range(len(angles)):
	#	if (i+err_offset) % err_gap == 0:
	#		plt.errorbar(x=i,y=angles[i],yerr=aerr[i],\
	#			marker=err_marker,markersize=err_marker_size,\
	#			color=err_color)

	# b. getting the current axis to label
	axis = plt.gca()
	axis.set_ylabel('Angle Error')

	# 4. Plotting the Radius error
	# a. defining the main plot
	plt.subplot(3, 1, 3)

	# + Confiance Interval
	x = [i for i in range(len(radius))]
	plt.fill_between(x, radius+(rci/2), radius-(rci/2),
					 color='b', alpha=.25)

	plt.plot(radius, 
			 label='Radius', 
			 linestyle='-', 
			 color='cornflowerblue',
			 linewidth=1)

	#for i in range(len(radius)):
	#	if (i+err_offset) % err_gap == 0:
	#		plt.errorbar(x=i,y=radius[i],yerr=rerr[i],\
	#			marker=err_marker,markersize=err_marker_size,\
	#			color=err_color)

	# b. getting the current axis to label
	axis = plt.gca()
	axis.set_ylabel('Radius Error')

	axis.set_xlabel('Number of Iterations')

	# 5. Showing the result
	# fig.savefig("./plots/dataset_history_based.jpg")
	plt.show()
	plt.close(fig)

def plot_type_probability(aga_tp, abu_tp, pf_tp, threshold, plotname):
	aga_tp = np.array(aga_tp)
	abu_tp = np.array(abu_tp)
	pf_tp = np.array(pf_tp)

	# 1. Setting the figure
	global fig_count
	fig = plt.figure(fig_count,figsize=(6.4,2.4))
	fig_count += 1

	# 2. Normalizing TP
	# AGA
	aga_error= list()
	for t in range(threshold):
		er =[]
		for i in range(len(aga_tp)):
			if t < len(aga_tp[i]):
				er.append(aga_tp[i][t])
		aga_error.append(np.array(er).mean())
	aga_error = np.array(aga_error)

	ci_list = []
	for e_l in aga_tp:
		ci_list.append(len(e_l))
		if not is_constant(ci_list):
			aga_ci = calcConfInt(ci_list)
		else:
			aga_ci = 0

	# ABU
	abu_error= list()
	for t in range(threshold):
		er =[]
		for i in range(len(abu_tp)):
			if t < len(abu_tp[i]):
				er.append(abu_tp[i][t])
		abu_error.append(np.array(er).mean())
	abu_error = np.array(abu_error)

	ci_list = []
	for e_l in abu_tp:
		ci_list.append(len(e_l))
		if not is_constant(ci_list):
			abu_ci = calcConfInt(ci_list)
		else:
			abu_ci = 0

	# PF
	pf_error= list()
	for t in range(threshold):
		er =[]
		for i in range(len(pf_tp)):
			if t < len(pf_tp[i]):
				er.append(pf_tp[i][t])
		pf_error.append(np.array(er).mean())
	pf_error = np.array(pf_error)

	ci_list = []
	for e_l in pf_tp:
		ci_list.append(len(e_l))
		if not is_constant(ci_list):
			pf_ci = calcConfInt(ci_list)
		else:
			pf_ci = 0


	# 3. Plotting
	#x = [t for t in range(threshold)]
	#print aga_error
	#plt.fill_between(x, aga_error-(aga_ci/2), 
	#					aga_error+(aga_ci/2),
	#				 color='b', alpha=.25)

	#plt.fill_between(x, abu_error-(abu_ci/2), 
	#					abu_error+(abu_ci/2),
	#				 color='g', alpha=.15)

	#plt.fill_between(x, pf_error-(pf_ci/2), 
	#					pf_error+(pf_ci/2),
	#				 color='r', alpha=.15)


	# b. Len Mean
	plt.plot(aga_error,
			 label='AGA',
			 color='b',
			 linestyle=':',			 
			 linewidth=2)

	plt.plot(abu_error,
			 label='ABU',
			 color='g',
			 linestyle='-.',   
			 linewidth=2)

	plt.plot(pf_error,
			 label='PF',
			 color='r',
			 linestyle='-',   
			 linewidth=2)

	# 4. Showing Results
	axis = plt.gca()
	axis.set_ylabel('True Type Estimation',fontsize='x-large')
	axis.set_xlabel('Number of Iterations',fontsize='x-large')
	axis.xaxis.set_tick_params(labelsize=14)
	axis.yaxis.set_tick_params(labelsize=14)
	axis.legend(loc="upper center", fontsize='large',\
				borderaxespad=0.1,borderpad=0.1,handletextpad=0.1,\
				fancybox=True,framealpha=0.8,ncol=3)

	# 3. Showing the result
	plt.savefig(plotname+'.pdf', bbox_inches = 'tight',pad_inches = 0)
	#plt.show()
	plt.close(fig)

def plot_run_length_bar(aga_m,aga_s,abu_m,abu_s,pf_m,pf_s,plotname):
    # 1. Setting the figure
    global fig_count
    fig = plt.figure(fig_count,figsize=(6.4,2.4))
    fig_count += 1

    bar_w = 0.5

    # 2. Plotting the number of iteration for each run to load items
    # a. defining the main plot
    axis = plt.gca()

    # AGA
    aga=axis.bar(0.75,height=aga_m,width= bar_w,yerr=aga_s-aga_m, color='b')

    # ABU
    abu=axis.bar(1.75,height=abu_m,width= bar_w,yerr=abu_s-abu_m, color='g')

    # PF
    pf=axis.bar(2.75,height = pf_m,width= bar_w,yerr = pf_s-pf_m, color='r')

    # b. getting the current axis to label
    axis.set_ylabel('Number of Iterations')
    axis.set_xticks([0,1,2])
    axis.set_xticklabels(['AGA','ABU','PF'])

    # 5. Showing the result
    plt.savefig(plotname+'.pdf', bbox_inches = 'tight',pad_inches = 0)
    #plt.show()
    plt.close(fig)

def plot_run_length(aga_m,aga_ci,abu_m,abu_ci,pf_m,pf_ci,plotname):
	aga_m = np.array(aga_m)
	aga_ci = np.array(aga_ci)
	abu_m = np.array(abu_m)
	abu_ci = np.array(abu_ci)
	pf_m = np.array(pf_m)
	pf_ci = np.array(pf_ci)

	# 1. Setting the figure
	global fig_count
	fig = plt.figure(fig_count,figsize=(6.4,2.4))
	fig_count += 1

	# 2. Plotting
	#x = [t for t in range(len(aga_m))]
	x = np.array([3,5,7])

	# a. Len CI
	delta = (aga_ci-aga_m)
	plt.fill_between(x, aga_m-delta, 
						aga_m+delta,
					 color='b', alpha=.15)
   	delta = (abu_ci-abu_m)
	plt.fill_between(x, abu_m-delta,
					 abu_m+delta, 
					 color='g', alpha=.15)
	delta = (pf_ci-pf_m)
	plt.fill_between(x, pf_m-delta,
					 pf_m+delta, 
					 color='r', alpha=.15)

	# b. Len Mean
	#import ipdb; ipdb.set_trace()
	#plt.plot(x,aga_m)
	plt.plot(x,aga_m,
			 label='AGA',
			 color='b',
			 linestyle=':',			 
			 linewidth=2,
			 clip_on=False)

	plt.plot(x,abu_m,
			 label='ABU',
			 color='g',
			 linestyle='-.',   
			 linewidth=2,
			 clip_on=False)

	plt.plot(x,pf_m,
			 label='PF',
			 color='r',
			 linestyle='-',   
			 linewidth=2,
			 clip_on=False)

	# c. Formating
	axis = plt.gca()
	axis.set_ylabel('Mean Iterations',fontsize='x-large')
	axis.set_xlabel('Visibility Radius',fontsize='x-large')
	axis.xaxis.set_tick_params(labelsize=14)
	axis.yaxis.set_tick_params(labelsize=14)
	axis.legend(loc="upper center", fontsize='large',\
				borderaxespad=0.1,borderpad=0.1,handletextpad=0.1,\
				fancybox=True,framealpha=0.8,ncol=3)

	# 5. Showing the result
	plt.savefig(plotname+'.pdf', bbox_inches = 'tight',pad_inches = 0)
	#plt.show()
	plt.close(fig)

def plot_summarised(aga,aga_std,aga_ci, 
	abu,abu_std,abu_ci, pf,pf_std,pf_ci,
	threshold, plotname, show_errorbar=False, show_confint=False):
	# 0. Layout Settings
	fig_w, fig_h = 6.4, 2.4

	err_marker = 'o'
	err_marker_size = 1
	err_color = 'lightblue'
	err_offset = 0
	err_gap = 1

	# 1. Setting the figure
	global fig_count
	fig = plt.figure(fig_count,figsize=(6.4,2.4))
	fig_count += 1

	# 2. Plotting
	x = [t for t in range(threshold)]
	#import ipdb; ipdb.set_trace()

	plot_aga = np.array([aga[t] for t in range(threshold)])
	plot_abu = np.array([abu[t] for t in range(threshold)])
	plot_pf  = np.array([pf[t]  for t in range(threshold)])

	plot_aga_ci = np.array([aga_ci[t] for t in range(threshold)])
	plot_abu_ci = np.array([abu_ci[t] for t in range(threshold)])
	plot_pf_ci  = np.array([pf_ci[t]  for t in range(threshold)])

	if show_confint:
		#import ipdb; ipdb.set_trace()
		delta = (plot_aga_ci-plot_aga)
		plt.fill_between(x, plot_aga-delta, 
							plot_aga+delta,
						 color='b', alpha=.15)
		delta = (plot_abu_ci-plot_abu)
		plt.fill_between(x, plot_abu-delta,
						 plot_abu+delta, 
						 color='g', alpha=.15)
		delta = (plot_pf_ci-plot_pf)
		plt.fill_between(x, plot_pf-delta,
						 plot_pf+delta, 
						 color='r', alpha=.15)


	plt.rcParams["figure.figsize"] = (fig_w,fig_h)
	plt.plot(plot_aga,
			 label='AGA',
			 color='b',
			 linestyle=':',			 
			 linewidth=2,
			 clip_on=False)

	plt.plot(plot_abu,
			 label='ABU',
			 color='g',
			 linestyle='-.',   
			 linewidth=2,
			 clip_on=False)

	plt.plot(plot_pf,
			 label='PF',
			 color='r',
			 linestyle='-',   
			 linewidth=2,
			 clip_on=False)

	if show_errorbar:
		delta = (plot_aga_ci-plot_aga)
		for i in range(len(plot_aga)):
			if (i+err_offset) % err_gap == 0:
				plt.errorbar(x=i,y=plot_aga[i],yerr=delta[i],\
					marker=err_marker,markersize=err_marker_size,\
					color='b',alpha=0.15)

		delta = (plot_abu_ci-plot_abu)
		for i in range(len(plot_aga)):
			if (i+err_offset) % err_gap == 0:
				plt.errorbar(x=i,y=plot_abu[i],yerr=delta[i],\
					marker=err_marker,markersize=err_marker_size,\
					color='g',alpha=0.15)

		delta = (plot_pf_ci-plot_pf)
		for i in range(len(plot_aga)):
			if (i+err_offset) % err_gap == 0:
				plt.errorbar(x=i,y=plot_pf[i],yerr=delta[i],\
					marker=err_marker,markersize=err_marker_size,\
					color='r',alpha=0.15)

	axis = plt.gca()
	axis.set_ylabel('Error',fontsize='x-large')
	axis.set_xlabel('Number of Iterations',fontsize='x-large')
	axis.xaxis.set_tick_params(labelsize=14)
	axis.yaxis.set_tick_params(labelsize=14)
	axis.legend(loc="upper center", fontsize='large',\
				borderaxespad=0.1,borderpad=0.1,handletextpad=0.1,\
				fancybox=True,framealpha=0.8,ncol=3)

	# 3. Showing the result
	plt.savefig(plotname+'.pdf', bbox_inches = 'tight',pad_inches = 0)
	#plt.show()
	plt.close(fig)

######################################################################
		##	   ##	 ###	#### ##	   ## 
		###   ###   ## ##	 ##  ###   ## 
		#### ####  ##   ##   ##  ####  ## 
		## ### ##  ##	 ##  ##  ## ## ## 
		##	   ## #########  ##  ##  #### 
		##	   ## ##	 ##  ##  ##   ### 
		##	   ## ##	 ## #### ##	   ## 
######################################################################
# 1. Reading the files and extracting the results
for root in ROOT_DIRS:
	if root == 'Outputs_POMCP':
		for sz in SIZE:
			for na in NAGENTS:
				for ni in NITEMS:
					for ra in RADIUS:
						filename = 'POMCP_s'+sz+'_a'+na+'_i'+ni+'_r'+ra+'_Pickle'
						if not os.path.exists(filename):
							results.append(init.read_files(root,sz,na,ni,ra))
							info = init.extract_information(results[-1],'POMCP_s'+sz+'_a'+na+'_i'+ni+'_r'+ra)
							print info.name
							info.normalise()
							info.extract()
							info.threshold = min([info.AGA_max_len_hist,info.ABU_max_len_hist,info.PF_max_len_hist])
							informations.append(info)

							file = open(filename)
							pickle.dump(info,file)
							file.close() 
						else:
							file = open(filename,'r')
							info = pickle.load(file)
							informations.append(info)
							file.close()
	else:
		for sz in SIZE:
			for na in NAGENTS:
				for ni in NITEMS:
					filename = 'MCTS_s'+sz+'_a'+na+'_i'+ni+'_Pickle'
					if not os.path.exists(filename):
						results.append(init.read_files(root,sz,na,ni))
						info = init.extract_information(results[-1],'MCTS_s'+sz+'_a'+na+'_i'+ni)
						print info.name
						info.normalise()
						info.extract()
						info.threshold = min([info.AGA_max_len_hist,info.ABU_max_len_hist,info.PF_max_len_hist])
						informations.append(info)

						file = open(filename,'wb')
						pickle.dump(info,file)
						file.close() 
					else:
						print filename,'already exists'
						file = open(filename,'r')
						info = pickle.load(file)
						informations.append(info)
						file.close()

# 3. Plotting the Information
print '***** plotting *****'
for info in informations:
	print info.name,'Level'
	plot_summarised(info.aga_levels,info.aga_levels_std_dev,info.aga_levels_ci,\
		info.abu_levels, info.abu_levels_std_dev, info.abu_levels_ci,\
		info.pf_levels, info.pf_levels_std_dev, info.pf_levels_ci,\
		info.threshold, info.name+'_Level',False, True)
	print info.name,'Radius'
	plot_summarised(info.aga_radius, info.aga_radius_std_dev, info.aga_radius_ci,\
		info.abu_radius, info.abu_radius_std_dev, info.abu_radius_ci,\
		info.pf_radius, info.pf_radius_std_dev, info.pf_radius_ci,\
		info.threshold, info.name+'_Radius',False, True)
	print info.name,'Angle'
	plot_summarised(info.aga_angles, info.aga_angles_std_dev, info.aga_angles_ci,\
		info.abu_angles, info.abu_angles_std_dev, info.abu_angles_ci,\
		info.pf_angles, info.pf_angles_std_dev, info.pf_angles_ci,\
		info.threshold, info.name+'_Angle',False, True)

print '***** general results *****'
for info in informations:
	general_aga = np.array([info.aga_levels[i]+info.aga_radius[i]\
							+info.aga_angles[i] for i in range(info.threshold)])/3
	general_aga_std_dev = np.array([info.aga_levels_std_dev[i]+info.aga_radius_std_dev[i]\
							+info.aga_angles_std_dev[i] for i in range(info.threshold)])/3
	general_aga_ci = np.array([info.aga_levels_ci[i]+info.aga_radius_ci[i]\
							+info.aga_angles_ci[i] for i in range(info.threshold)])/3

	general_abu = np.array([info.abu_levels[i]+info.abu_radius[i]\
							+info.abu_angles[i] for i in range(info.threshold)])/3
	general_abu_std_dev = np.array([info.abu_levels_std_dev[i]+info.abu_radius_std_dev[i]\
							+info.abu_angles_std_dev[i] for i in range(info.threshold)])/3
	general_abu_ci = np.array([info.abu_levels_ci[i]+info.abu_radius_ci[i]\
							+info.abu_angles_ci[i] for i in range(info.threshold)])/3

	general_pf = np.array([info.pf_levels[i]+info.pf_radius[i]\
							+info.pf_angles[i] for i in range(info.threshold)])/3
	general_pf_std_dev = np.array([info.pf_levels_std_dev[i]+info.pf_radius_std_dev[i]\
							+info.pf_angles_std_dev[i] for i in range(info.threshold)])/3
	general_pf_ci = np.array([info.pf_levels_ci[i]+info.pf_radius_ci[i]\
							+info.pf_angles_ci[i] for i in range(info.threshold)])/3

	plot_summarised(general_aga, general_aga_std_dev, general_aga_ci,\
		general_abu, general_abu_std_dev, general_abu_ci,\
		general_pf, general_pf_std_dev, general_pf_ci,\
		info.threshold, info.name + '_General',False, True)

# 4. Plotting the mean run length
print '***** history len performance *****'
aga_m, aga_ci = list(), list()
abu_m, abu_ci = list(), list()
pf_m , pf_ci  = list(), list()
for info in informations:
	print '*******'
	print 'AGA',info.AGA_mean_len_hist
	print 'ABU',info.ABU_mean_len_hist
	print 'PF',info.PF_mean_len_hist
	aga_m.append(info.AGA_mean_len_hist)
	aga_ci.append(info.AGA_ci_len_hist)
	abu_m.append(info.ABU_mean_len_hist)
	abu_ci.append(info.ABU_ci_len_hist)
	pf_m.append( info.PF_mean_len_hist )
	pf_ci.append( info.PF_ci_len_hist )

	plot_run_length_bar(info.AGA_mean_len_hist,info.AGA_ci_len_hist,\
						info.ABU_mean_len_hist,info.ABU_ci_len_hist,\
						info.PF_mean_len_hist,info.PF_ci_len_hist,'Perform')
#plot_run_length(aga_m,aga_ci,abu_m,abu_ci,pf_m,pf_ci,'Visibility')

# 5. Plotting the type probability
print '***** type probability performance *****'
for info in informations:
	plot_type_probability(info.AGA_typeProbHistory,\
					info.ABU_typeProbHistory,\
					info.PF_typeProbHistory ,\
					info.threshold, info.name+'TypePerformance')