#! /usr/bin/env python
# Generate Birth Death Rates for Given Initial Conditions
# Tim Tyree
# 8.11.2020
#TODO: test this works from an .ipynb
# - TODO: make kwargs accessable from the main function call
# - TODO: learn how to send different ic to each node on the OSG
# - TODO: make positions/ folder and separate Log/ from positions/.
# - TODO: consider switching to a different log file for each routine function call
# - DONE: remove the redundant beginning of track_tip_trajectories
# - DONE: generate more test cases for  birth_death_rates_from_ic
# - DONE: clean up these sections.  remove uneeded comments, redundant code
# - DONE: save birth death rates to a file named according to all of the relevant parameters in a special folder. don't delete the last entry! it gives the termination time!
# - DONE: move all parameters to the beginning of the notebook
# - DONE: move notebook contents to sublimetext and call on a list of functions!
# - DONE: make EP fields spread out into their own fields in the tip locations
# - DONE: parallelize birth_death_rates_from_ic for a list of initial conditions using dask.  (develop the dask part first in an .ipynb, then apply the dask part by calling the functionally in birth_death_rates_from_ic.py)
# - DONE: make the filename handling sensible
# - TODO: change the print statements to print to a log file
# - TODO: smooth out the filenames in sublimeText
# - DONE: make asserting=True by default. (optional) make beeping=True by default
# - TODO(this shouldn't be needed): make ext incrementing work again.  keep it simple, stupid!  Hint: try using operari.get_trailing_number?
# - DONE: make parameters passed by kwargs so dask doesn't run into memory access issues!
# - TODO: fix (parallel dask) logger.  first get a simple case to work with dask


#pylab
# %matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
# from pylab import imshow, show

#tracking
import numba, trackpy

#automate the boring stuff
# from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
	nb_dir = os.getcwd()

#are these even used?
from numba import njit, jit, vectorize
# import skimage as sk
from skimage import measure, filters
# from PIL import Image
# import imageio

#load the libraries
from lib import *
from lib.dist_func import *

#load the libraries
from lib.operari import *
from lib.ProgressBar import *
from lib.minimal_model import *
from lib.TexturePlot import *
from lib.get_tips import *
from lib.minimal_model import *
from lib.intersection import *
from lib.tracking import link

#load the measrue library for robust, simplified, fast tip detection
from lib.measure import find_contours
from lib.measure._utils_find_contours import *
from lib.measure._utils_find_tips import *
from lib.measure._find_tips import *

# %autocall 1
# %load_ext autoreload
# %autoreload 2

# #the given initial conditions
# sub_id = 31
# initial_condition_dir = nb_dir + f'/Data/initial-conditions/ic_200x200.101.{sub_id}.npz'
# # search_for_file()


############################################
###### START PARAMETER COMMENTS ############
############################################
#the given parameters
# #printing/testing parameters
# beeping   = True
# asserting = True
# printing  = True
# plotting  = False

# #define parameters for tip detection
# sigma       = 1.5
# threshold   = 0.6
# V_threshold = 0.5

# #define other parameters that don't appear to affect the results
# edge_tolerance = 3
# pad = 5
# atol = 1e-11
# color_values = None

# h = 0.007
# nsteps = 1*10**6 # int(1.8*10**5)#0#*10**4
# save_every_n_frames = 100#nsteps#100#nsteps#10#50
# max_time = h*nsteps


# # time_sig_figs = 4  #you need to change this manually below for pretty printing only!
# max_buffers_to_save = 0
# buffers_saved = 0
# start_saving_buffers_at_step = 0#10**4 # nsteps/4
# # tip_states = None
# timing = False
# recording_if_odd = True
# recording = True
# descrip = f'sigma_{sigma}_threshold_{threshold}'

# #this saving overwrites the current save_file.  I check ahead of time that this won't happen with a "Caution! ... " print statement
# save = True

# #define parameters for tip tracking
# mem = 2
# sr  = 40 #?works sampling every 100 frames
# # sr  = 1 #works sampling every frame

# #trajectory filtering parameters
# tmin = 50#100
# LT_thresh = 0#2 #14 = 0.007*2000

# #input/output filenames
# base_save_dir = f"{nb_dir}/Data/ds_5_param_set_8"
# save_folder = f'{nb_dir}/Data/ds_5_param_set_8' #no '/Log'

# tip_log_fn = f'Log/{sub_id}_ds_5_sigma_{sigma}_threshold_{threshold}_.csv'
# tip_log_dir = get_unique_file_name(os.path.join(save_folder, tip_log_fn), ext_string='.csv');
# tip_position_dir = tip_log_dir.replace('.csv','_processed.csv')

# data_dir_traj = base_save_dir+"/trajectories"
# data_fn_traj = os.path.basename(tip_position_dir).replace('_processed.csv', f'_traj_sr_{sr}_mem_{mem}.csv')

# save_folder_traj = '/'+os.path.join(*initial_condition_dir.split('/')[:-2])+'/ds_5_param_set_8/trajectories/'
# birth_death_dir = '/'.join(save_folder_traj.split('/')[:-2])+'/birth-death-rates'
# data_fn_bdrates = data_fn_traj.replace('.csv','_bdrates.csv')

# #set the ending extension explicitely
# ending_str = '_007.csv'
# tip_log_dir = '_'.join ( tip_log_dir.split('_')[:-1] ) + ending_str
# if printing:
#     print(f"tip_log_dir is: \n\t{tip_log_dir}.")
#     if (os.path.exists(tip_log_dir)):
#         print(f"Caution! This tip_log_dir already exists!")

# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')
	return True

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__
	return True

####################################
######## GENERATE TIP LOGS  ########
####################################
def generate_tip_logs(initial_condition_dir, **kwargs):
	'''if logging, change the print statements to a .log file unique to ic'''
	# if kwargs['logging']:
	# 	log = open(kwargs['print_log_dir'], "a")
	# 	sys.stdout = log
	logging = kwargs['logging']
	printing = kwargs['printing']
	if printing:
		print(f'loading initial conditions from: \n\t{initial_condition_dir}.')
	os.chdir(nb_dir)
	txt = load_buffer(initial_condition_dir)
	width, height, channel_no = txt.shape
	kwargs.update({'width':width,'height':height})
	#reinitialize records
	time_start = 0.  #eval(buffer_fn[buffer_fn.find('time_')+len('time_'):-4])
	asserting = kwargs['asserting']
	if asserting:
		assert (float(time_start) is not None)
	tip_state_lst = []
	tme = time_start
	#change directory to where results are logged
	save_folder = kwargs['data_folder_log']
	os.chdir(save_folder)
	printing = kwargs['printing']
	if printing:
		print(f"changed directory to save_folder: \n\t{save_folder}.")
	#precompute the _padded_ mesh coordinates
	pad = kwargs['pad']
	ycoord_mesh, xcoord_mesh = np.meshgrid(np.arange(0,txt.shape[0]+2*(pad)),np.arange(0,txt.shape[0]+2*pad))
	nanstate = [np.nan,np.nan,np.nan]
	# channel_no = 3
	# # jump forward a number of steps to see what happens to the buffer
	# for j in range(500):
	#     #integrate explicitely in time
	#     time_step(txt, h=h, zero_txt=zero_txt) #up to twice as fast as for separated calls
	#     tme += h
	#############################################
	####This section may be skipped##############
	#############################################
	# check all the functions work and compile the needed functions just in time
	zero_txt = txt.copy()*0.
	h = kwargs['h']
	time_step(txt, h=h, zero_txt=zero_txt)
	width, height, channel_no = txt.shape
	zero_txt = np.zeros((width, height, channel_no), dtype=np.float64)
	dtexture_dt = zero_txt.copy()
	get_time_step(txt, dtexture_dt)
	sigma = kwargs['sigma']
	V_threshold= kwargs['V_threshold']
	threshold  = kwargs['threshold']
	edge_tolerance = kwargs['edge_tolerance']
	atol = kwargs['atol']
	#lewiner marching squares with periodic boundary conditions fed into curve intersection
	# #compute as discrete flow map dtexture_dt
	# dtexture_dt = zero_txt.copy()
	# get_time_step(txt, dtexture_dt)
	#     img    = pad_mat_by_1x1(txt[...,0])
	#     dimgdt = pad_mat_by_1x1(dtexture_dt[...,0])
	#compute the images to find isosurfaces of
	img = txt[...,0]
	dimgdt = dtexture_dt[...,0]
	#compute both families of contours
	contours1 = find_contours(img, level = 0.5)
	contours2 = find_contours(dimgdt, level = 0.0)
	#find_tips
	tips_mapped = contours_to_simple_tips_pbc(contours1,contours2,width, height, jump_threshold = 2, size_threshold = 6)
	n_old = count_tips(tips_mapped[2])
	#extract local EP field values for each tip
	# states_EP = get_states(tips_mapped, txt, pad, nanstate, xcoord_mesh, ycoord_mesh, channel_no = channel_no)
	# tips_mapped = add_states(tips_mapped, states_EP)
	# color_values = None
	# n_lst, x_lst, y_lst = contours_to_simple_tips_pbc(contours1,contours2,width, height, jump_threshold = 2, size_threshold = 6)
	# n_tips = count_tips(x_lst)
	# #record data for current time
	# t_list.append(t)
	# n_list.append(n_tips)
	# #forward Euler integration in time
	# txt += h*dtexture_dt
	# t   += h
	# #calculate contours and tips after enforcing pbcs
	# img_nxt = txt[..., 0]#padded_txt#
	# img_inc = ifilter(dtexture_dt[..., 0])# ifilter(dpadded_txt_dt) #  #mask of instantaneously increasing voltages
	# img_inc = filters.gaussian(img_inc,sigma=sigma , mode='wrap')
	# img_nxt_unpadded = img_nxt.copy()
	# img_inc_unpadded = img_inc.copy()
	# img_nxt, img_inc = matrices_to_padded_matrices(img_nxt_unpadded, img_inc_unpadded,pad=pad)
	# txt_padded, dtexture_dt_padded = matrices_to_padded_matrices(txt, dtexture_dt,pad=pad)
	# contours_raw = measure.find_contours(img_nxt, level=V_threshold,fully_connected='low',positive_orientation='low')
	# contours_inc = measure.find_contours(img_inc, level=threshold)
	# tips  = get_tips(contours_raw, contours_inc)
	# # tips_mapped = map_pbc_tips_back(tips=tips, pad=pad, width=width, height=height,
	# # 				  edge_tolerance=edge_tolerance, atol = atol)
	# n_old = count_tips(tips_mapped[2])
	# #extract local EP field values for each tip
	# states_EP = get_states(tips_mapped, txt, pad, nanstate, xcoord_mesh, ycoord_mesh, channel_no = channel_no)
	# tips_mapped = add_states(tips_mapped, states_EP)
	# color_values = None
	# # color by state value
	# states_nearest, states_interpolated_linear, states_interpolated_cubic = states_EP
	# chnl = 2
	# # typs = 'nearest'
	# # typs = 'linearly interpolated'
	# typs = 'cubic-spline interpolated'
	# state_value = np.array(states_interpolated_linear)[:,chnl]
	# color_values = state_value
	# print(f'plotting channel {chnl} \nof the {typs} states.')
	# if printing:
	# #bluf
	# print('max value of change for each channel is {:.4f} , {:.4f}, {:.4f}.'.format(*tuple(np.max(txt,axis=(0,1)))))
	# print('max rate of change for each channel is {:.4f} , {:.4f}, {:.4f}.'.format(*tuple(np.max(dtexture_dt,axis=(0,1)))))
	# print(f"\n number of type 1 contour = {len(contours_raw)},\n number of type 2 contour = {len(contours_inc)},")
	# print(f"the number of tips are {n_old}. time is {tme:.1f} ms.")
	# print(f"note for time <100ms, I'm saying nothing about the accuracy of tip detection.")
	# print(f"""the topological tip state:{tips[0]}""")
	# print(f"""x position of tips: {tips[1]}""")
	# print(f"""y position of tips: {tips[2]}""")
	# plotting = kwargs['plotting']
	# if plotting:
	# 	#plot texture contours and tips. oh my!
	# 	# img_nxt_unpadded = img_nxt[pad:-pad,pad:-pad]
	# 	# img_inc_unpadded = img_inc[pad:-pad,pad:-pad]
	# 	contours_raw_unpadded = contours1 # measure.find_contours(img_nxt_unpadded, level=V_threshold,fully_connected='low',positive_orientation='low')
	# 	contours_inc_unpadded = contours2 # measure.find_contours(img_inc_unpadded, level=threshold)
	# 	# #print texture information
	# 	# describe(txt)
	# 	fig = plot_buffer(img, dimgdt, contours1, contours2, tips_mapped,
	# 					  figsize=(5,5),max_marker_size=400, lw=1, color_values=color_values);
	# 	# fig = plot_buffer(img_nxt_unpadded, img_inc_unpadded, contours_raw_unpadded, contours_inc_unpadded, tips_mapped,
	# 	# 				  figsize=(5,5),max_marker_size=400, lw=1, color_values=color_values);
	# 	plt.show()
	# 	# plt.close()
	#integrate explicitely in time
	state = np.zeros((txt.shape[0],txt.shape[1],4),dtype=np.float64)
	nsteps = kwargs['nsteps']
	save_every_n_frames = kwargs['save_every_n_frames']
	recording_if_odd = kwargs['recording_if_odd']
	if printing:
		print(f"sigma is {sigma}, threshold is {threshold}.")
		print(f"pad is {pad}, rejection_distance is edge_tolerance is {edge_tolerance}.")
		print(f"starting simulation.  integrating no further than time {h*nsteps+tme:.3f} milliseconds.")
	start = time.time()
	for step in range(nsteps):
		recording = step%save_every_n_frames==0
		if not recording:
			#integrate explicitely in time
			time_step(txt, h=h, zero_txt=zero_txt) #up to twice as fast as for separated calls
			tme += h
		if recording:
			#calculate discrete flow map
			dtexture_dt = zero_txt.copy()
			get_time_step(txt, dtexture_dt)

			#pad texture for saving view
			# padded_txt, dpadded_txt_dt = textures_to_padded_textures(txt, dtexture_dt,pad=pad)

			#compute the images to find isosurfaces of
			img    = txt[...,0]
			dimgdt = dtexture_dt[...,0]

			#compute both families of contours
			contours1 = find_contours(img,    level = 0.5)
			contours2 = find_contours(dimgdt, level = 0.0)

			#find_tips
			tips_mapped = contours_to_simple_tips_pbc(contours1,contours2,width, height, jump_threshold = 2, size_threshold = 6)

			# #extract local EP field values for each tip
			# states_EP = get_states(tips_mapped, txt, pad, nanstate, xcoord_mesh, ycoord_mesh, channel_no = channel_no)
			# tips_mapped = add_states(tips_mapped, states_EP)

			#integrate explicitely in time by the forward euler method
			txt += h*dtexture_dt
			tme += h
			#
			# #calculate contours and tips after enforcing pbcs
			# img_nxt = txt[..., 0]#padded_txt#
			# img_inc = dtexture_dt[..., 0].copy()# ifilter(dpadded_txt_dt) #  #mask of instantaneously increasing voltages
			# # img_inc = ifilter(dtexture_dt[..., 0])# ifilter(dpadded_txt_dt) #  #mask of instantaneously increasing voltages
			# # img_inc = filters.gaussian(img_inc,sigma=sigma, mode='wrap')
			# img_nxt_unpadded = img_nxt.copy()
			# img_inc_unpadded = img_inc.copy()
			# img_nxt, img_inc = matrices_to_padded_matrices(img_nxt_unpadded, img_inc_unpadded,pad=pad)
			# contours_raw = measure.find_contours(img_nxt, level=V_threshold,fully_connected='low',positive_orientation='low')
			# # contours_inc = measure.find_contours(img_inc, level=threshold)
			# contours_inc = measure.find_contours(img_inc, level=0.)
			# tips  = get_tips(contours_raw, contours_inc)
			# tips_mapped = map_pbc_tips_back(tips=tips, pad=pad, width=width, height=height,
			# 				  edge_tolerance=edge_tolerance, atol = atol)
			#
			# #extract local EP field values for each tip
			# states_EP = get_states(tips_mapped, txt, pad, nanstate, xcoord_mesh, ycoord_mesh, channel_no = channel_no)
			# tips_mapped = add_states(tips_mapped, states_EP)

						#record spiral tip locations
			n_lst, x_lst, y_lst = tips_mapped
			tip_state_lst.append({
						't': float(tme),
						'x': tuple(x_lst),
						'y': tuple(y_lst)
			})

			# #record spiral tip locations
			# s1_lst, s2_lst, x_lst, y_lst, states_nearest, states_interpolated_linear, states_interpolated_cubic = tips_mapped
			# tip_state_lst.append({
			# 			't': float(tme),
			# 			'x': tuple(x_lst),
			# 			'y': tuple(y_lst),
			# 			's1': tuple(s1_lst),
			# 			's2': tuple(s2_lst),
			# 	'states_interpolated_linear': tuple(states_interpolated_linear)
			# })
			#         #record spiral tip locations with all three types of EP_states computed
			#         s1_lst, s2_lst, x_lst, y_lst, states_nearest, states_interpolated_linear, states_interpolated_cubic = tips_mapped
			#         tip_state_lst.append({
			#                     't': float(np.around(tme, time_sig_figs)),
			#                     'x': tuple(x_lst),
			#                     'y': tuple(y_lst),
			#                     's1': tuple(s1_lst),
			#                     's2': tuple(s2_lst),
			#             'states_nearest': tuple(states_nearest),
			#             'states_interpolated_linear': tuple(states_interpolated_linear),
			#             'states_interpolated_cubic': tuple(states_interpolated_cubic),
			#         })

			#determine if an odd number of tips were born
			n = count_tips(tips_mapped[2]) #counts the number of '.' in the nested list of x positions or just a normal list
			dn = n - n_old
			n_old = n

			# #save the state if save_state is True
			# #save_state = recording_if_odd & odd_event & odd_tip_number # ==> odd birth/death event has just occurred
			# save_state = recording_if_odd & (dn%2!=0) & (n%2!=0)
			# if save_state:
			# 	#plot texture contours and tips. oh my!
			# 	#img_nxt_unpadded = img_nxt[pad:-pad,pad:-pad]
			# 	#img_inc_unpadded = img_inc[pad:-pad,pad:-pad]
			# 	# contours_raw_unpadded = measure.find_contours(img_nxt_unpadded, level=V_threshold,fully_connected='low',positive_orientation='low')
			# 	# contours_inc_unpadded = measure.find_contours(img_inc_unpadded, level=0.)
			# 	# contours_inc_unpadded = measure.find_contours(img_inc_unpadded, level=threshold)
			# 	if printing:
			# 		print(f'odd tip spotted at time {tme:.3f}! dn={dn} and n={n}...')
			# 	fig = plot_buffer(img, dimgdt, contours1, contours2, tips_mapped,
			# 					  figsize=(5,5), max_marker_size=200, lw=1, color_values = None);
			# 	# fig = plot_buffer(img_nxt_unpadded, img_inc_unpadded, contours_raw_unpadded, contours_inc_unpadded, tips_mapped,
			# 	# 				  figsize=(5,5), max_marker_size=200, lw=1, color_values = None);
			# 	fig.savefig(f'plot_of_n_{n}_dn_{dn}_for_{descrip}_at_time_{tme:.1f}.pdf', bbox_inches='tight',pad_inches=0);
			# 	plt.close();
			# save_state = recording_if_odd & (dn%2!=0) & (n%2!=0)
			# if save_state:
			# 	#save texture as an .npy file as desired
			# 	if step>start_saving_buffers_at_step:
			# 		if buffers_saved<max_buffers_to_save:
			# 			buffers_saved += 1
			# 			np.save(f'buffer_of_n_{n}_dn_{dn}_for_{descrip}_at_time_{tme:.1f}.npy', txt)
			#early stopping when spirals die out
			stop_early = (n==0) & (step>40000) #np.max(txt[...,0])<0.1
			if stop_early:
				if printing:
					print(f'\nmax voltage is {np.max(txt[...,0]):.4f}.')
					print(f"tip number = {n}.  stopping simulation at time t={tme:.3f}. please record domain size.")
				break
		if not logging:
			if printing:
				printProgressBar(step + 1, nsteps, prefix = 'Progress:', suffix = 'Complete', length = 50)

	if printing:
		#report the bottom line up front
		print(f"\ntime integration complete. run time was {time.time()-start:.2f} seconds in realtime")
		print(f"current time is {tme:.1f} ms in simulation time.")
		print(f"number of nan pixel voltages is {np.max(sum(np.isnan(txt[...,0])))}.")
		# print(f"current max voltage is {np.nanmax(txt[...,0]):.4f}.")
		# print(f"current max fast variable is {np.nanmax(txt[...,1]):.4f}.")
		# print(f"current max slow variable is {np.nanmax(txt[...,2]):.4f}.")
		# n_lst, x_lst, y_lst = get_tips(contours_raw, contours_inc)
		# tip_states = {'n': n_lst, 'x': x_lst, 'y': y_lst}
		# print(f"tip_states are {tip_states}.")
		# print(f'current tip state is {tip_states}')
		# if len(lst)~=0:
		# print(f"number of tips is = {set([len(q) for q in lst_x[-1]])}.") #most recent number of tips
		if recording:
			print(f"\n number of type 1 contour = {len(contours1)},\tnumber of type 2 contour = {len(contours2)},")
			print(f"the number of tips are {count_tips(tips_mapped[2])}.")
			#     print(f"""the topological tip state is the following:{tips[0]}""")
	beeping = kwargs['beeping']
	if beeping:
		beep(1)
	max_time = kwargs['max_time']
	if printing:
		if tme >= max_time:
			print( f"Caution! max_time was reached! Termination time not reached!  Consider rerunning with greater n_steps!")
	tip_log_dir = kwargs['data_dir_log']
	save = kwargs['save']
	if save:
		df = pd.DataFrame(tip_state_lst)
		df.to_csv(tip_log_dir, index=False)
	#     df.to_csv(f'{nb_dir}/Data/tip_log_{descrip}_at_time_{tme:.1f}.csv', index=False)
	if printing:
		print('saved to:')
		print(tip_log_dir)
	# print(f'Data/tip_log__{descrip}_at_time_{tme:.1f}.csv')
	return tip_log_dir, kwargs
##########################################################
######## POSTPROCESSING TIP LOGS TO TIP LOCATIONS ########
##########################################################
def postprocess_tip_logs(tip_log_dir, **kwargs):
	# if kwargs['logging']:
	# 	log = open(kwargs['print_log_dir'], "a")
	# 	sys.stdout = log
	save_folder = kwargs['data_folder_log']
	os.chdir(save_folder)
	printing = kwargs['printing']
	if printing:
		print(str(os.path.exists(tip_log_dir))+" it is that the file to be post processed exists,")

	#save the tip positions expanded into rows
	df_output = process_tip_log_file(tip_log_dir, include_EP=False, include_nonlinear_EP=False)

	# #expand the EP data into its own columns
	# df_output = unwrap_EP(df_output,
	# 			   EP_col_name = 'states_interpolated_linear',
	# 			   drop_original_column=False).copy()

	#save the tip positions to csv
	tip_position_dir= kwargs['data_dir_tips']
	df_output.to_csv(tip_position_dir, index=False)
	if printing:
		print(f"and the resulting \"_processed.csv\" was supplanted herein:\n\t{tip_position_dir}")

	return tip_position_dir

############################################################
######## TRACKING TIP LOCATIONS TO TIP TRAJECTORIES ########
############################################################
def track_tip_trajectories(tip_position_dir, **kwargs):
	# if kwargs['logging']:
	# 	log = open(kwargs['print_log_dir'], "a")
	# 	sys.stdout = log
	printing = kwargs['printing']
	if printing:
		print(f"loading .csv of size [?? {2*sys.getsizeof(tip_position_dir)} KB ??] from \n\t{tip_position_dir}")

	#load processed df with tips as rows
	df = pd.read_csv(tip_position_dir)
	data_fn_tips = kwargs['data_fn_tips']
	fn = data_fn_tips # tip_position_dir.split('/')[-1]
	# # descrip = fn[:fn.find('_processed.csv')]
	# threshold = eval(fn[fn.find('threshold_')+len('threshold_'):].split('_')[0])
	# ds = eval(fn[fn.find('ds_')+len('ds_'):].split('_')[0])
	# sigma = eval(fn[fn.find('sigma_')+len('sigma_'):].split('_')[0])
	# if printing:
	# 	print(f"params inferred from filename: (ds,sigma,threshold) = {(ds,sigma,threshold)}")

	#import tip positions
	save_folder_traj = kwargs['data_folder_traj']
	os.chdir(save_folder_traj)
	if printing:
		print(f"files will be saved in the folder: \n\t{save_folder_traj}")

	# test data has no odd spiral tips since the data has periodic boundary conditions
	no_odd_spiral_tips_exist = not (np.array(list(set(df.n.values)))%2==1).any()
	asserting = kwargs['asserting']
	if asserting:
		assert (no_odd_spiral_tips_exist)
	if printing:
		if not no_odd_spiral_tips_exist:
			print('Caution! odd spiral tips exist!')

	#assign each time a unique frame number
	t_list =  sorted(set(df.t.values))
	frameno_list = list(range(len(t_list)))
	df['frame'] = -9999
	for frameno, t in zip(frameno_list,t_list):
		df.loc[df.t==t, 'frame'] = frameno

	if asserting:
		#test that all entries were given a value
		assert ( not (df.frame<0).any() )

	# track tip trajectories
	# width  = txt.shape[0] #0 may be switched with 1 here
	# height = txt.shape[1]
	width=kwargs['width']
	height=kwargs['height']
	sr=kwargs['sr']
	mem=kwargs['mem']
	distance_L2_pbc = get_distance_L2_pbc(width=width,height=height)
	link_kwargs = {
		'neighbor_strategy' : 'BTree',
		'dist_func'         : distance_L2_pbc,
		'search_range': sr,
		'memory': mem,
		'adaptive_stop': 2.0,  #stop decreasing search_range at 2
		'adaptive_step': 0.95  #if subnet overflow error is thrown, retry with a smaller search_range
		}

	#use my version of trackpy.link_df with their terrible logging shut off.
	traj = link(f=df,t_column='frame', verbose=False, **link_kwargs)

	# screw trackpy's logging shit.
	# # if not printing:
	# # 	blockPrint()
	# blockPrint()
	# traj = trackpy.link_df(
	# 	f=df,t_column='frame', **link_kwargs)
	# if kwargs['logging']:
	# 	log = open(kwargs['print_log_dir'], "a")
	# 	sys.stdout = log
	# else:
	# 	enablePrint()


	# save trajectories to csv
	data_fn_traj=kwargs['data_fn_traj']
	traj.to_csv(data_fn_traj, index=False)
	if printing:
		print (f"data_df_traj: {data_fn_traj}")

	if asserting:
		#test that every row has a nonnegative particle number
		assert ( (traj.particle>=0).all() )

	return data_fn_traj

############################################################
######## COMPUTING BIRTH DEATH RATES########################
############################################################
def compute_birth_death_rates(data_fn_trajectories, **kwargs):
	'''returns False if no spiral tips are detected,
	returns file_name saved to if spiral tips are detected but plotting=False
	else returns file_name, fig'''
	#filter trajectories and compute the (filtered) spiral tip number as a function of time. store in df
	data_fn_traj = data_fn_trajectories#kwargs['data_fn_traj']
	#import most recent tip trajectory data and the corresponding raw tips
	printing=kwargs['printing']
	if printing:
		print (f"loading trajectories from data_fn_traj: {data_fn_traj}.")

	#import most recent tip trajectory data and the corresponding raw tips
	data_folder_traj=kwargs['data_folder_traj']
	os.chdir(data_folder_traj)
	df = pd.read_csv(data_fn_traj)

	# select only data after tmin milliseconds
	tmin=kwargs['tmin']
	df = df[df.t>tmin].copy()

	#naive computation of lifetime for a given tip
	def get_lifetime(pid,df):
		mx,mn = df[(df.particle==pid)].t.describe()[['max','min']]
		lifetime = mx-mn #milliseconds
		return lifetime

	#get a DataFrame of only the long lived spiral tips
	pid_values = np.array(list(set(df.particle.values)))
	lifetime_values = np.array([get_lifetime(pid,df) for pid in pid_values])
	LT_thresh = kwargs['LT_thresh']
	boo = (lifetime_values>LT_thresh)
	boo_long = df.particle == None
	for pid in pid_values[boo]:
		boo_long |= (df.particle==pid)
	df = df.loc[boo_long].copy()
	n_series = df['t'].value_counts().sort_index()

	#explicitely append the last death when n becomes zero
	last_two_times = n_series.index.values[-2:]

	#if there were not any tips observed, don't make a .csv in bdrates and return False
	if len(last_two_times)==0:
		if printing:
			print('no birth-death event was detected!')
		return False
	else:
		h = kwargs['h']#np.diff(last_two_times)
		end_time= last_two_times[-1]+h
		n_series = n_series.append(pd.Series([0], index=[float(end_time)]))

		#store as a pandas.DataFrame
		df = pd.DataFrame({"t":n_series.index.values,"n":n_series.values})

		#compute birth death rates
		df['dn'] = df.n.diff().shift(-1)
		df = df.query('dn != 0').copy()
		rates = 1/df['t'].diff().shift(-1).dropna() # birth death rates in unites of 1/ms
		df['rates'] = rates
		# df.dropna(inplace=True) #this gets rid of the termination time datum.  we want that!

		#save birth death rates to a file named according to all of the relevant parameters in a special folder.
		birth_death_dir=kwargs['data_folder_bdrates']
		os.chdir(birth_death_dir)
		df.index.rename('index', inplace=True)
		data_fn_bdrates=kwargs['data_fn_bdrates']
		df.to_csv(data_fn_bdrates)

		if printing:
			print (f"birth death rates successfully saved in: {data_fn_bdrates}")
		beeping=kwargs['beeping']
		if beeping:
			beep(3)
		plotting=kwargs['plotting']
		if plotting:
			log_scale = False
			fontsize=20
			figsize=(6,5)
			fig, ax = plt.subplots(figsize=figsize)

			x_values = df.query('dn==2').n/2
			y_values = df.query('dn==2').rates
			ax.scatter(x=x_values,y=y_values, c='g', label='$W_{+2}$')

			x_values = df.query('dn==-2').n/2
			y_values = df.query('dn==-2').rates
			ax.scatter(x=x_values,y=y_values, c='r', label='$W_{-2}$')

			ax.legend(fontsize=fontsize-8)
			ax.set_xlabel('n/2', fontsize=fontsize)
			ax.set_ylabel('rate (ms$^{-1}$)', fontsize=fontsize)
			return data_fn_bdrates, fig
		return data_fn_bdrates

def _get_kwargs(ic):
	beeping   = False
	asserting = False
	printing  = True
	plotting  = False
	logging   = True
	sigma       = 1.5 #pixels
	threshold   = 0.6 #unitless 0 to 1
	V_threshold = 0.5  #unitless 0 to 1
	edge_tolerance = 6#20#3#6#10#3#10#3
	pad = 0#10#21#5#10#20#5#20#5
	atol = 1e-11#1e-9#1e-11#1e-9#1e-11
	color_values = None
	h = 0.025#0.01 #0.1 for when D=0.0005cm^2/ms, ##0.007) for when D=0.001cm^2/ms, #milliseconds
	nsteps = 1*10**7
	save_every_n_frames = 100#100#10#100
	max_time = h*nsteps  #milliseconds
	max_buffers_to_save = 0
	buffers_saved = 0
	start_saving_buffers_at_step = 0
	timing = False
	recording_if_odd = True
	recording = True
	descrip = f'sigma_{sigma}_threshold_{threshold}'
	save = True
	mem  = 2#2 #frames
	sr   = 10  #pixels
	ds=5 #cm #width of square domain
	tmin = 100#milliseconds
	LT_thresh = 0# this might be the one causeing odd tips 2 #milliseconds

	kwargs = {
		'beeping':beeping,
		'asserting':asserting,
		'printing':printing,
		'plotting':plotting,  #TODO: test when plotting=True
		'logging':logging,
		'sigma':sigma,  #pixels
		'threshold':threshold,  #unitless 0 to 1
		'V_threshold':V_threshold,  #unitless 0 to 1
		'edge_tolerance':edge_tolerance,
		'pad':pad,
		'atol':atol,
		'color_values':color_values,
		'h':h , #0.1 for when D=0.0005cm^2/ms, ##0.007) for when D=0.001cm^2/ms, #milliseconds
		'nsteps':nsteps,
		'save_every_n_frames':save_every_n_frames,
		'max_time':max_time,  #milliseconds
		'max_buffers_to_save':max_buffers_to_save,
		'buffers_saved_counter':buffers_saved,
		'start_saving_buffers_at_step':start_saving_buffers_at_step,
		'timing':timing,
		'recording_if_odd':recording_if_odd,
		'recording':recording,
		'descrip':descrip,
		'save':save,
		'mem':mem, #frames
		'sr':sr,  #pixels
		'ds': ds, #cm #width of square domain
		'tmin':tmin, #milliseconds
		'LT_thresh':LT_thresh#milliseconds
	}

	#(ignore these for now, the file names already encode how the trial was conducted)
	# #TODO: determine if any of the output files will be overwritten
	# if printing:
	#     print(f"tip_log_dir is: \n\t{data_dir_tips}.")
	#     if (os.path.exists(data_dir_tips)):
	#         print(f"Caution! This tip_log_dir already exists!")
	# #TODO: set the ending extension if the current one is in use
	# ending_str = '_008.csv'
	# tip_log_dir = '_'.join ( tip_log_dir.split('_')[:-1] ) + ending_str
	# if printing:
	# 	print(f"tip_log_dir is: \n\t{tip_log_dir}.")
	# 	if (os.path.exists(tip_log_dir)):
	# 		print(f"Caution! This tip_log_dir already exists!")

	#compute filesystem names
	#input/output filenames
	input_fn = ic
	base_dir = '/'.join(os.path.dirname(input_fn).split('/')[:-1])
	base_save_folder_name = 'ds_5_param_set_8'
	base_save_dir = os.path.join(base_dir, base_save_folder_name)

	#define subfolders
	subfolder_list = ('birth-death-rates', 'trajectories', 'Log')
	data_folder_bdrates = os.path.join(base_save_dir,subfolder_list[0])
	data_folder_traj    = os.path.join(base_save_dir,subfolder_list[1])
	data_folder_log     = os.path.join(base_save_dir,subfolder_list[2])

	#define filenames with an order consistent with workflow
	data_fn = os.path.basename(input_fn)
	data_fn_log     = data_fn.replace('.npz', f'_ds_{ds}_sigma_{sigma}_threshold_{threshold}_log.csv')
	data_fn_tips    = data_fn_log.replace('_log.csv', '_processed.csv')
	data_fn_traj    = data_fn_log.replace('_log.csv', f'_sr_{sr}_mem_{mem}_traj.csv')
	data_fn_bdrates = data_fn_traj.replace('_traj.csv', f'_tmin_{tmin}_LT_{LT_thresh}_bdrates.csv')

	data_dir_bdrates = os.path.join(data_folder_bdrates,data_fn_bdrates)
	data_dir_traj    = os.path.join(data_folder_traj,data_fn_traj)
	data_dir_tips    = os.path.join(data_folder_log,data_fn_tips)
	data_dir_log     = os.path.join(data_folder_log,data_fn_log)
	print_log_dir    = os.path.join(data_folder_log,data_fn_log.replace('_log.csv','.log'))

	kwargs_io = {
		'base_dir':base_dir,
		'base_save_dir':base_save_dir,
		'data_folder_bdrates':data_folder_bdrates,
		'data_folder_traj':data_folder_traj,
		'data_folder_log':data_folder_log,
		'data_fn':data_fn,
		'data_fn_log':data_fn_log,
		'data_fn_tips':data_fn_tips,
		'data_fn_traj':data_fn_traj,
		'data_fn_bdrates':data_fn_bdrates,
		'data_dir_bdrates':data_dir_bdrates,
		'data_dir_traj':data_dir_traj,
		'data_dir_tips':data_dir_tips,
		'data_dir_log':data_dir_log,
		'print_log_dir':print_log_dir,
	}
	kwargs.update(kwargs_io)

	#the following caused an io crash of dask :(
	# if printing:
	# 	print( f"Will integrate up to no later than time = {max_time:.2f} milliseconds.")

	return kwargs

def get_kwargs(ic):
	return _get_kwargs(ic)

################################
###### MAIN ROUTINES ############
################################
def birth_death_rates_from_ic(ic):
	#get key word arguments
	kwargs = _get_kwargs(ic)

	#update any of the default kwargs here, such as the domain width,
	# kwargs['ds'] = ??

	# if logging, change the print statements to a .log file unique to ic
	logging = kwargs['logging']
	if logging:
		log = open(kwargs['print_log_dir'], "a")
		sys.stdout = log

	# main routine
	tip_log_dir, kwargs      = generate_tip_logs(initial_condition_dir=ic, **kwargs)
	tip_position_dir = postprocess_tip_logs(tip_log_dir, **kwargs)
	data_fn_trajectories     = track_tip_trajectories(tip_position_dir, **kwargs)
	data_fn_bdrates  = compute_birth_death_rates(data_fn_trajectories, **kwargs)

	#move the completed file to ic-out
	completed_ic_fn = os.path.join(*(kwargs['base_dir'],'ic-out',os.path.basename(ic)))
	os.rename(ic,completed_ic_fn)

	if logging:
		if not log.closed:
			log.close()
	return data_fn_bdrates

# def tip_log_from_ic(ic):
# 	#get key word arguments
# 	kwargs = _get_kwargs(ic)

# 	#update any of the default kwargs here, such as the domain width,
# 	# kwargs['ds'] = ??

# 	# main routine
# 	tip_log_dir, kwargs      = generate_tip_logs(initial_condition_dir=ic, **kwargs)
# 	# tip_position_dir = postprocess_tip_logs(tip_log_dir, **kwargs)
# 	# data_fn_trajectories     = track_tip_trajectories(tip_position_dir, **kwargs)
# 	# data_fn_bdrates  = compute_birth_death_rates(data_fn_trajectories, **kwargs)

# 	#move the completed file to ic-out
# 	completed_ic_fn = os.path.join(*(kwargs['base_dir'],'ic-out',os.path.basename(ic)))
# 	os.rename(ic,completed_ic_fn)
# 	return data_fn_bdrates


if __name__=='__main__':
	for ic in sys.argv[1:]:
		birth_death_rates_from_ic(ic)
		print(f"completed birth_death_rates_from_ic: {ic}")
		print(f"birth-death rates stored in: {data_fn_bdrates}")
