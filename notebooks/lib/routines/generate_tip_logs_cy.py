#! /usr/bin/env python
# Generate Birth Death Rates for Given Initial Conditions
# Tim Tyree
# 10.28.2020
from .. import *
from ..measure.measure import *
#^this import is not the best practice. however, minimalism reigns the pythonic.
import numpy as np, pandas as pd, matplotlib.pyplot as plt

#automate the boring stuff
# from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'here_dir' in globals():
	here_dir = os.getcwd()
def generate_tip_logs_from_ic(initial_condition_dir, h, tmax,
	V_threshold, theta_threshold,dsdpixel,
	tmin_early_stopping, save_every_n_frames, round_output_decimals,
	timing, printing, logging, asserting, beeping, saving,
	data_dir_log, completed_ic_dir, print_log_dir, param_fn, **kwargs):
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
	txt = load_buffer(initial_condition_dir)
	width, height = txt.shape[:2]
	zero_txt = txt.copy()*0.
	# width, height, channel_no = txt.shape

	kwargs.update({
		'width':width,
		'height':height
		})
	#reinitialize records
	time_start = 0.  #eval(buffer_fn[buffer_fn.find('time_')+len('time_'):-4])
	if asserting:
		assert (float(time_start) is not None)
	tip_state_lst = []
	t = time_start
	dict_out_lst = []
	num_steps = int(np.around((tmax-t)/h))

	#precompute anything that needs precomputing
	if printing:
		#print(f"sigma is {sigma}, threshold is {threshold}.")
		#print(f"pad is {pad}, rejection_distance is edge_tolerance is {edge_tolerance}.")
		print(f"starting simulation.  integrating no further than time {tmax:.3f} milliseconds.")
	if timing:
		start = time.time()

	param_dir = os.path.join(here_dir,'lib/model')
	param_dict = json.load(open(os.path.join(param_dir,param_fn)))
	get_time_step=fetch_get_time_step(width,height,DX=dsdpixel,DY=dsdpixel,**param_dict)
	time_step=fetch_time_step(width,height,DX=dsdpixel,DY=dsdpixel,**param_dict)


	##########################################
	#run the simulation, measuring regularly
	##########################################
	step_count = 0
	n_tips=1 #to initialize loop invarient (n_tips > 0) to True
	while (t<tmax) & (n_tips > 0):
		if step_count%save_every_n_frames != 0:
			#forward Euler integration in time
			time_step(txt, h, zero_txt)
		else:
			#take measurements once every n frames
			#compute as discrete flow map dtxt_dt
			dtxt_dt = zero_txt.copy()
			get_time_step(txt, dtxt_dt)

			#compute the images used to find isosurfaces
			img    = txt[...,0]
			dimgdt = dtxt_dt[...,0]

			# find_intersections
			retval = find_intersections(img,dimgdt,level1,level2)
			lst_values_x,lst_values_y,lst_values_theta, lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy = retval
			x_values = np.array(lst_values_x)
			y_values = np.array(lst_values_y)
			# EP states given by bilinear interpolation with periodic boundary conditions
			v_lst, f_lst, s_lst = interpolate_states(x_values,y_values,width,height,txt)
			dvdt_lst, dfdt_lst, dsdt_lst = interpolate_states(x_values,y_values,width,height,dtxt_dt)

			n_tips = x_values.size
			dict_out = {
				't': float(t),
				'n': int(n_tips),
				'x': tuple(lst_values_x),
				'y': tuple(lst_values_y),
				'theta': tuple(lst_values_theta),
				'grad_ux': tuple(lst_values_grad_ux),
				'grad_uy': tuple(lst_values_grad_uy),
				'grad_vx': tuple(lst_values_grad_vx),
				'grad_vy': tuple(lst_values_grad_vy),
				'v':v_lst,
				'f':f_lst,
				's':s_lst,
				'dvdt':dvdt_lst,
				'dfdt':dfdt_lst,
				'dsdt':dsdt_lst,
			}
			dict_out_lst.append(dict_out)

			# #compute both families of contours
			# contours1 = find_contours(img,    level = 0.8)
			# contours2 = find_contours(dimgdt, level = 0.0)

			# #find_tips and measure tip topological/EP state
			# s1_list, s2_list, x_lst, y_lst, v_lst, f_lst, s_lst = measure_system(contours1, contours2, width, height, txt,
			# 																	 jump_threshold = jump_threshold,
			# 																	 size_threshold = size_threshold,
			# 																	 pad=pad, decimals=decimals)
			# n_tips = x_lst.size
			# dict_out = {
			# 	't': float(t),
			# 	'n': int(n_tips),
			# 	'x': tuple(x_lst),
			# 	'y': tuple(y_lst),
			# 	'n1': tuple(s1_list),
			# 	'n2': tuple(s2_list),
			# 	'v':v_lst,
			# 	'f':f_lst,
			# 	's':s_lst,
			# }

			#record data for current time
			dict_out_lst.append(dict_out)

			#forward Euler integration in time
			txt += h*dtxt_dt

			#update progress bar after each measurement
			if not logging:
				if printing:
					printProgressBar(step_count, num_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
		#advance time by one step
		t   += h
		step_count += 1
	if not logging:
		if printing:
			printProgressBar(step_count, num_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

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
	for ic in sys.argv[1:]:
		kwargs = get_kwargs(ic)
		kwargs = generate_tip_logs_from_ic(ic, **kwargs)
		print(f"completed generate_tip_logs_from_ic: {ic}")
		print(f"csv of spiral tip data stored in: {kwargs['completed_ic_dir']}")
