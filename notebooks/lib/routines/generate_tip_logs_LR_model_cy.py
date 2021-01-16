#! /usr/bin/env python
# Generate Birth Death Rates for Given Initial Conditions
# Tim Tyree
# 1.13.2021
from .. import * 
from ..measure.measure import *  
#^this import is not the best practice. however, minimalism reigns the pythonic.
import numpy as np, pandas as pd, matplotlib.pyplot as plt


from ..my_initialization import *
from ..controller.controller_LR import *#get_one_step_explicit_synchronous_splitting as get_one_step
from ..model.LR_model import *
from ..utils.utils_traj import *
from ..utils.stack_txt_LR import stack_txt, unstack_txt
from ..routines.bdrates import *
from ..measure.utils_measure_tips_cpu import *
from ..viewer import *
import trackpy

#automate the boring stuff
# from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'here_dir' in globals():
	here_dir = os.getcwd()


# @njit
# def comp_transient_gating_variable(var, tau, varinfty):
# 	return (varinfty - var)/tau


def generate_tip_logs_from_ic(initial_condition_dir, h, tmax,
	V_threshold,
	tmin_early_stopping, save_every_n_frames, round_output_decimals, 
	timing, printing, logging, asserting, beeping, saving, 
	data_dir_log, completed_ic_dir, print_log_dir, 
	Ca_i_initial = 2*10**-4, Vmax = 45., Vmin = -75.,
	**kwargs):
	'''generates a log of tip locations on 2D grid with periodic boundary conditions.
	default key word arguments are returned by lib.routines.kwargs.get_kwargs(initial_condition_dir).'''
	level1 = V_threshold
	level2 = 0.

	# if logging, change the print statements to a .log file unique to ic
	if logging:
		log = open(print_log_dir, "a")
		sys.stdout = log

	if printing:
		print(f'loading initial conditions from: \n\t{initial_condition_dir}.')

	# os.chdir(here_dir)
	# txt = load_buffer_LR(initial_condition_dir)
	# width, height = txt.shape[:2]
	# zero_txt = txt.copy()*0.
	# width, height, channel_no = txt.shape


	#reinitialize records
	time_start = 0.  #eval(buffer_fn[buffer_fn.find('time_')+len('time_'):-4])
	if asserting:
		assert (float(time_start) is not None)
	tip_state_lst = []
	t = time_start
	dict_out_lst = []  
	num_steps = int(np.around((tmax-t)/h))


	#initialize simulation
	txt=load_buffer_LR(initial_condition_dir, Ca_i_initial = Ca_i_initial, Vmax = Vmax, Vmin = Vmin)
	width, height, channel_no = txt.shape
	kwargs.update({
		'width':width,
		'height':height
		})
	#allocate memory
	inVc=txt[...,(0,-1)].copy()
	inmhjdfx=txt[...,1:-1].copy()
	outVc=inVc.copy()
	dVcdt=inVc.copy()
	outmhjdfx=inmhjdfx.copy()
	#reformate texture
	txt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)

	#precompute anything that needs precomputing
	dt=h/100.
	#get one_step method
	dt, one_step_map = get_one_step_map(nb_dir,dt)
	# txt=one_step_map(txt)

	if printing:
		#print(f"sigma is {sigma}, threshold is {threshold}.")
		#print(f"pad is {pad}, rejection_distance is edge_tolerance is {edge_tolerance}.")
		print(f"integrating to time t={tmin_early_stopping:.3f} ms without recording with dt={dt:.3f} ms.")
	while (t<tmin_early_stopping):
		txt=one_step_map(txt)
		t+=dt

	#precompute anything that needs precomputing
	dt=h
	#get one_step method
	dt, one_step_map = get_one_step_map(nb_dir,dt)
	# txt=one_step_map(txt)

	if printing:
		#print(f"sigma is {sigma}, threshold is {threshold}.")
		#print(f"pad is {pad}, rejection_distance is edge_tolerance is {edge_tolerance}.")
		print(f"integrating to no later than time t={tmax:.3f} milliseconds. ms with recording with dt={dt:.3f} ms.")
	if timing:
		start = time.time()
	##########################################
	#run the simulation, measuring regularly
	##########################################
	step_count = 0
	n_tips=1 #to initialize loop invarient (n_tips > 0) to True
	while (t<tmax) & (n_tips > 0):
		if step_count%save_every_n_frames == 0:
			#compute tip locations in dict_out
			#update texture namespace
			inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
			# txt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
			img=inVc[...,0]
			dimgdt=dVcdt[...,0]
			dict_out=compute_all_spiral_tips(t,img,dimgdt,level1,level2)#,width=width,height=height)

			#save tip data
			n_tips=dict_out['n']
			# n_tips_lst.append(n_tips)
			# t_lst.append(t)
			dict_out_lst.append(dict_out)

			#update progress bar after each measurement
			if not logging:
				if printing:
					printProgressBar(step_count, num_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

		#forward Euler integration in time
		txt=one_step_map(txt)
		#advance time by one step
		t   += dt
		step_count += 1
	# if not logging:
	# 	if printing:
	# 		printProgressBar(step_count, num_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

	if printing:
		#report the bottom line up front
		if timing:
			print(f"\ntime integration complete. run time was {time.time()-start:.2f} seconds in realtime")
		print(f"\ncurrent time is {t:.1f} ms in simulation time.")

		print(f"number of nan pixel voltages is {np.max(sum(np.isnan(txt[...,0])))}.")
		# print(f"current max voltage is {np.nanmax(txt[...,0]):.4f}.")
		# print(f"current max fast variable is {np.nanmax(txt[...,1]):.4f}.")
		# print(f"current max slow variable is {np.nanmax(txt[...,2]):.4f}.")
		print(f"number of tips is = {n_tips}.") 
		if n_tips==0:
			print(f"zero tips remaining at time t = {t:.1f} ms.")
	if beeping:
		beep(1)
	if printing:
		if t >= tmax:
			print( f"Caution! max_time was reached! Termination time not reached!  Consider rerunning with greater max_time!")
	if saving:
		df = pd.concat([pd.DataFrame(dict_out) for dict_out in dict_out_lst])
		df.reset_index(inplace=True, drop=True)
		#if the end of AF was indeed reachded, append a row recording this
		if n_tips==0:
			next_id = df.index.values[-1]+1
			df = pd.concat([df,pd.DataFrame({'t': float(save_every_n_frames*h+t),'n': int(n_tips)}, index = [next_id])])
		#save the recorded data
		df.round(round_output_decimals).to_csv(data_dir_log, index=False)
		if printing:
			print('saved to:')
			print(data_dir_log)

	#move the completed file to ic-out
	os.rename(initial_condition_dir,completed_ic_dir)
	#input ic moved to output
	if logging:
		if not log.closed:
			log.close()		

	return kwargs

if __name__=='__main__':
	from lib.routines.kwargs_LR_model_cy import get_kwargs
	for ic in sys.argv[1:]:
		kwargs = get_kwargs(ic)
		kwargs = generate_tip_logs_from_ic(ic, **kwargs)
		print(f"completed birth_death_rates_from_ic: {ic}")
		print(f"csv of spiral tip data stored in: {kwargs['completed_ic_dir']}")
