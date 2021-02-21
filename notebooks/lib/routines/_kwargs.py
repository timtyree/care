import os
def get_kwargs(ic):
	'''default parameters for routines'''
	beeping   = False
	asserting = False
	printing  = False
	plotting  = False #works but not with logger
	logging   = False #seems to be failing to give up the GIL
	saving    = True
	timing    = False
	recording = True
	V_threshold = 0.5  #unitless 0 to 1
	h = 0.025#0.01 #0.1 for when D=0.0005cm^2/ms, ##0.007) for when D=0.001cm^2/ms, #milliseconds
	dsdpixel = 0.025  # cm # the distance between two adjacent pixels to 5/200= 0.025 cm
	jump_threshold = 2 #min pixel distance between consec. vertices on a contour to be considered a jump over periodic boundaries
	size_threshold = 6 #min num of vertices for a contour to be considered.  Smaller values yield much slower measurements, but certain guarantees against missing spiral tips for rare edge cases.
	pad = 1 # used in interpolating local state?
	# edge_tolerance = 6#20#3#6#10#3#10#3
	atol = 1e-10#unused by tip logger - 1e-9#1e-11#1e-9#1e-11
	decimals = 10
	tmax_sec = 300 #maximum time to be integrated in seconds
	tmax = tmax_sec*10**3
	# nsteps = 1*10**7
	# max_time = h*nsteps #milliseconds
	# max_buffers_to_save = 0
	# buffers_saved = 0
	# start_saving_buffers_at_step = 0

	# recording_if_odd = False

	#parameters for making tip trajectories from tip logs
	LT_thresh = 0# this might be the one causeing odd tips 2 #milliseconds
	tmin = 100#milliseconds
	mem  = 2  # frames
	sr   = 3  #pixels  #search range for tracking spiral tips in pixels
	save_every_n_frames = 40 # 1 measurement per 1 ms for h=0.025 ms appears reasonable to resolve nearby births
	tmin_early_stopping = 100 # milliseconds earliest time to stop recording in the absense of spiral tips
	round_output_decimals = 10#5

	kwargs = {
		'beeping':beeping,
		'asserting':asserting,
		'printing':printing,
		'plotting':plotting,  #TODO: test when plotting=True
		'logging':logging,
		'saving':saving,
		'V_threshold':V_threshold,  #unitless 0 to 1
		'jump_threshold':jump_threshold,
		'size_threshold':size_threshold,
		'pad':pad,
		'atol':atol,
		# 'color_values':color_values,
		'h':h , #0.1 for when D=0.0005cm^2/ms, ##0.007) for when D=0.001cm^2/ms, #milliseconds
		'dsdpixel':dsdpixel,
		# 'nsteps':nsteps,
		'save_every_n_frames':save_every_n_frames,
		# 'max_buffers_to_save':max_buffers_to_save,
		# 'buffers_saved_counter':buffers_saved,
		# 'start_saving_buffers_at_step':start_saving_buffers_at_step,
		'timing':timing,
		# 'recording_if_odd':recording_if_odd,
		'recording':recording,
		'saving':saving,
		'mem':mem, #frames
		'sr':sr,  #pixels
		# 'ds': ds, #cm #width of square domain
		'tmin_early_stopping':tmin_early_stopping, #milliseconds
		'tmin':tmin, #milliseconds
		'LT_thresh':LT_thresh,#milliseconds
		'round_output_decimals':round_output_decimals,
		'decimals':decimals,
		'tmax':tmax #milliseconds
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
	data_fn = os.path.basename(input_fn)
	base_dir = '/'.join(os.path.dirname(input_fn).split('/')[:-1])
	base_save_folder_name = 'ds_5_param_set_8'
	base_save_dir = os.path.join(base_dir, base_save_folder_name)

	#define subfolders
	subfolder_list = ('birth-death-rates', 'trajectories', 'Log', 'print-log')
	data_folder_bdrates = os.path.join(base_save_dir,subfolder_list[0])
	data_folder_traj    = os.path.join(base_save_dir,subfolder_list[1])
	data_folder_log     = os.path.join(base_save_dir,subfolder_list[2])
	data_folder_print_log     = os.path.join(base_save_dir,subfolder_list[3])

	#define filenames with an order consistent with workflow
	data_fn_log     = data_fn.replace('.npz', f'_log.csv')
	data_fn_tips    = data_fn_log.replace('_log.csv', '_processed.csv')
	data_fn_traj    = data_fn_log.replace('_log.csv', f'_sr_{sr}_mem_{mem}_traj.csv')
	data_fn_bdrates = data_fn_traj.replace('_traj.csv', f'_tmin_{tmin}_LT_{LT_thresh}_bdrates.csv')

	data_dir_bdrates = os.path.join(data_folder_bdrates,data_fn_bdrates)
	data_dir_traj    = os.path.join(data_folder_traj,data_fn_traj)
	data_dir_tips    = os.path.join(data_folder_log,data_fn_tips)
	data_dir_log     = os.path.join(data_folder_log,data_fn_log)
	print_log_dir    = os.path.join(data_folder_log,data_folder_print_log.replace('_log.csv','.log'))
	completed_ic_dir = os.path.join(*(base_dir,'ic-out',data_fn))
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
		'completed_ic_dir':completed_ic_dir
	}
	kwargs.update(kwargs_io)

	#the following caused an io crash of dask :(
	# if printing:
	# 	print( f"Will integrate up to no later than time = {max_time:.2f} milliseconds.")

	return kwargs
