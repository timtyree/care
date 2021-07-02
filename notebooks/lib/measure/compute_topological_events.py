# computes creation/annihilation events given input_fn of .csv trajectory dataframes
# Tim Tyree
# 6.17.2021
import numpy as np, pandas as pd,os
from ..utils.projection_func import get_subtract_pbc
from .relative_phases import *
from .compute_relative_velocities import *
from ..routines.compute_interactions import compute_df_interactions
from .compute_phase_angles import *

def produce_one_csv(list_of_files, file_out, encoding="utf-8"):
   # Consolidate all csv files into one object
   df = pd.concat([pd.read_csv(file).reset_index() for file in list_of_files])
   # df = pd.concat([pd.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   # if df.columns[0]=='Unnamed: 0':
	#    df.drop(columns=['Unnamed: 0'])
   df.to_csv(file_out, index=False, encoding=encoding)
   return os.path.abspath(file_out)

def filter_before(df,tmin=100,t_column='t'):
	'''filter all time earlier than tmin'''
	tmin=100#ms
	boo=df[t_column]<tmin
	df.drop(df[boo],inplace=True)
	return df

##########################################
# Annihilation
##########################################
def get_compute_final_inout_angles(width,height):
	subtract_pbc=get_subtract_pbc(width=width,height=height)
	def compute_final_inout_angles(d1,d2):
		'''computes the unsigned angles between the final velocity of d1 near death for one tip pair.
		Updates d1,d2 with fields.  aligns locations by index.  supposes index is the field, frame
		Example Usage:
		compute_final_inout_angles=get_compute_final_inout_angles(width,height)
		tdeath_values,angle_values=compute_final_inout_angles(d1,d2)
		'''
		#compute displacement of d1 with pbc
		xy_values=np.array(list(zip(d1['x'],d1['y'])))
		dshifted=d1.shift(1).copy()
		# dshifted=d1.shift(-1).copy()
		xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
		dxy1_values=np.zeros_like(xy_values)+np.nan
		# compute displacement unit vector from tip 1 to tip 2
		xy_values=np.array(list(zip(d1['x'],d1['y'])))
		dshifted=d1.shift(1).copy()
		# dshifted=d1.shift(-1).copy()
		xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
		dxy1_values=np.zeros_like(xy_values)+np.nan
		#compute displacements between
		for j in range(dxy1_values.shape[0]):
			dxy1_values[j]=subtract_pbc(xy_next_values[j],xy_values[j])
		d1['dx']=dxy1_values[:,0]
		d1['dy']=dxy1_values[:,1]
		d1['dt']=d1['t'].diff().shift(-1).iloc[1:-1]
		# d1[['dx','dy','dt']]=d1[['x','y','t']].diff().shift(-1).iloc[1:-1]
		d1['displacement']=np.sqrt(d1['dx']**2+d1['dy']**2)
		d1['dx_hat']=d1['dx']/d1['displacement']
		d1['dy_hat']=d1['dy']/d1['displacement']

		#compute COM relative to d1
		d1['x2']=d2['x']
		d1['y2']=d2['y']
		xy1_values=np.array(list(zip(d1['x'],d1['y'])))
		xy2_values=np.array(list(zip(d1['x2'],d1['y2'])))
		dxy12_values=np.zeros_like(xy1_values)+np.nan
		#compute displacements between
		for j in range(dxy12_values.shape[0]):
			dxy12_values[j]=subtract_pbc(xy2_values[j],xy1_values[j])
		d1['comx']=xy1_values[:,0]+dxy12_values[:,0]/2
		d1['comy']=xy1_values[:,1]+dxy12_values[:,1]/2

		#compute the radial unit vector of d1
		d1['rx']=xy1_values[:,0]-d1['comx']
		d1['ry']=xy1_values[:,1]-d1['comy']
		d1['rx_hat']=d1['rx']/np.sqrt(d1['rx']**2+d1['ry']**2)
		d1['ry_hat']=d1['ry']/np.sqrt(d1['rx']**2+d1['ry']**2)

		#compute unsigned angle between velocity and the radial unit vector
		# compute dot product between tip 1 and tip 2
		cosine_series=d1['dx_hat']*d1['rx_hat']+d1['dy_hat']*d1['ry_hat']
		d1['theta']=np.arccos(cosine_series)   #radians
		d1.dropna(inplace=True)

		angle_values=d1['theta'].values
		tdeath_values=d1['t'].values[-1]-d1['t'].values #ms
		return tdeath_values,angle_values
	return compute_final_inout_angles

def compute_annihilation_events(input_fn,
								width,
								height,
								ds,
								range_threshold=1.,
								min_duration=20.,
								min_range=1.,
								round_t_to_n_digits=3,
								use_min_duration=True,
								use_grad_voltage=True,
								printing=True,
								**kwargs):
	'''input_fn is a string locating the directory of a _trajectories .csv file.
	Returns pandas.Dataframe instance.
	Example Usage:
	input_fn=f"/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-fk-200x200/param_set_8_ds_5.0_tmax_10_diffCoef_0.0005/Log/ic200x200.0.3_traj_sr_400_mem_0.csv"
	df_phases=compute_annihilation_events(input_fn,width,height,ds)#,**kwargs)
	'''
	# # original default values that worked for the Fenton-Karma model...
	#     range_threshold=1.#cm
	#     #compute measures near annihilation
	#     #input: df_ordered_interactions
	#     min_duration=20#150 #ms
	#     min_range=1. #cm
	#     use_min_duration=True#broken for false
	#     use_grad_voltage=False
	df=pd.read_csv(input_fn)
	try:
		DT = compute_DT(df, round_t_to_n_digits=round_t_to_n_digits)
		if printing:
			print(f"the time resolution is {DT} ms.")
			
	except AttributeError as e:
		return f'Warning: AttributeError raise for {input_fn}.  frame is probably missing from the input DataFrame instance...'

	dsdpixel = ds / width  #cm per pixel
	DS = dsdpixel
	#compile jit
	compute_final_inout_angles=get_compute_final_inout_angles(width,height)
	# compute_angle_between_final_velocities = get_compute_angle_between_final_velocities(width, height)
	compute_ranges_between = get_compute_ranges_between(width=width,height=height)

	# death_ranges,birth_ranges,DT=return_bd_ranges(input_fn,DS,round_t_to_n_digits=3)
	#compute interactions
	df_interactions = compute_df_interactions(input_fn, DS=DS)
	df_interactions.dropna(inplace=True)
	death_ranges = DS * df_interactions.rT.values
	birth_ranges = DS * df_interactions.r0.values

	#filter any deaths that occur at ranges exceeding range_threshold
	boo = df_interactions.rT * DS < range_threshold
	df_ordered_interactions = df_interactions[boo].sort_values('Tavg',
															   ascending=False)

	#compute the phase time series between pid and pid_deathmate
	#find last zero for when phi1==0
	#compute phi2 at that time using linear interpolation
	#append phi2 to dphi_lst... proceed to the next particle
	pid_queue = list(df_ordered_interactions.pid.values)
	pid_deathmate_dict = dict(
		zip(pid_queue, list(df_ordered_interactions.pid_deathmate.values)))
	df_out_lst = []
	while len(pid_queue) > 0:
		pid = pid_queue.pop(0)
		pid_deathmate = pid_deathmate_dict[pid]
		try:
			pid_queue.remove(pid_deathmate)
			#extract d1,d2
			d1 = df[df.particle == pid].copy()
			d2 = df[df.particle == pid_deathmate].copy()
			d1.index = d1.frame
			d2.index = d2.frame
			#compute ranges between
			range_values = compute_ranges_between(d1, d2) * dsdpixel
			length = range_values.shape[0]
			#compute x,y values
			if use_grad_voltage:
				t_to_death_values, phi1_values, phi2_values, phi_sum_values, phi_diff_values = compute_phase_angles_from_grad_voltage(
					d1, d2)

			else:
				t_vals = d1['t'].values[-length:]
				t_to_death_values = np.max(t_vals) - t_vals

			nobs = range_values[1:-1].shape[0]
			if nobs > 1:
				#compute angle between velocities
				# tdeath_values, theta_values = compute_angle_between_final_velocities(d1, d2)
				tdeath_values, theta_values = compute_final_inout_angles(d1, d2)
				#align range_values and theta_values
				boo = ~np.isnan(theta_values)
				theta_values = theta_values[boo]
				assert (range_values.shape[0] == t_to_death_values.shape[0])
				if use_min_duration:
					#filter by min_duration
					t_values = t_to_death_values[1:-1]
					boo_keep = min_duration <= np.max(t_values) - np.min(
						t_values)
					#filter by min_range
					boo_keep &= min_range <= np.max(range_values)
				else:
					boo_keep = min_range <= np.max(range_values)

				if boo_keep:
					if use_grad_voltage:
						df_out = pd.DataFrame({
							'pid':
							pid + 0 * t_values.astype('int'),
							'pid_deathmate':
							pid_deathmate + 0 * t_values.astype('int'),
							'tdeath':
							np.around(t_values,round_t_to_n_digits),
							'phi1':
							np.abs(phi1_values)[1:-1],
							'phi2':
							np.abs(phi2_values)[1:-1],
							'phi_sum':
							np.abs(phi_sum_values)[1:-1],
							'phi_diff':
							np.abs(phi_diff_values)[1:-1],
							'r':
							range_values[1:-1],
							'theta':
							theta_values,
						})
					else:
						df_out = pd.DataFrame({
							'pid':
							pid + 0 * t_values.astype('int'),
							'pid_deathmate':
							pid_deathmate + 0 * t_values.astype('int'),
							'tdeath':
							np.around(t_values,round_t_to_n_digits),
							'r':
							range_values[1:-1],
							'theta':
							theta_values,
						})

					#append x,y values to list
					df_out_lst.append(df_out)
		except ValueError as e:  #for catching ValueError: list.remove(x): x not in list
			pass
	if printing:
		print(f"the number of trials appended to df_out_lst is {len(df_out_lst)}")
		assert (len(df_out_lst) > 0)
	if not (len(df_out_lst) > 0):
		return None
	# t_to_death_values.shape, range_values.shape, theta_values.shape, d1[
	#     't'].values.shape, d2['t'].values.shape
	df_phases = pd.concat(df_out_lst)
	return df_phases


def save_annihilation_events(input_fn,
							 width,
							 height,
							 ds,
							 save_folder=None,
							 save_fn=None,
							 **kwargs):
	'''
	Example Usage:
	input_fn=f"/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-fk-200x200/param_set_8_ds_5.0_tmax_10_diffCoef_0.0005/Log/ic200x200.0.3_traj_sr_400_mem_0.csv"
	save_fn=save_annihilation_events(input_fn,width,height,ds,save_folder=None,save_fn=None)#,**kwargs)
	'''
	df_phases = compute_annihilation_events(input_fn, width, height, ds, **kwargs)
	if df_phases is None:
		return f"Warning: no annihilation events considered valid for trial located at \n\t {input_fn}"
	if type(df_phases)!=type(pd.DataFrame):
		return df_phases

	if save_folder is None:
		#save df_phases as csv
		save_folder = os.path.dirname(
			os.path.dirname(input_fn)) + '/annihilations'
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)
	os.chdir(save_folder)
	if save_fn is None:
		save_fn = os.path.basename(input_fn).replace('.csv',
													 '_annihilations.csv')
	df_phases.to_csv(save_fn, index=False)
	return os.path.abspath(save_fn)

#DONE: copy daskbag-routine for msd analysis
#TODO: dask-bag routine ^this shiz for a given input_folder that contains trajectory files
def get_routine_traj_to_annihilation(width=200,height=200,ds=5.,
		save_folder=None,save_fn=None,**kwargs):
	'''
	Example Usage:
	routine_traj_to_annihilation=get_routine_traj_to_annihilation(width=200,height=200,ds=5.)
	'''
	def routine_traj_to_annihilation(input_file_name):
		'''run_routine_log_to_msd returns where it saves .csv of annihilation events
		'''
		# # traj_fn = preprocess_log(fn)# wraps generate_track_tips_pbc
		# traj_fn = generate_track_tips_pbc(fn, save_fn=None)
		# input_file_name=traj_fn
		# output_file_name=input_file_name.replace('.csv',"_unwrap.csv")
		# retval_ignore= unwrap_trajectories(input_file_name, output_file_name)
		output_file_name=save_annihilation_events(input_file_name,width,height,ds,
			save_folder=save_folder,save_fn=save_fn,**kwargs)#,**kwargs)
		return output_file_name
	return routine_traj_to_annihilation

##########################################
# Creation
##########################################
def get_compute_initial_inout_angles(width,height):
	subtract_pbc=get_subtract_pbc(width=width,height=height)
	def compute_initial_inout_angles(d1,d2):
		'''computes the unsigned angles between the initial velocity of d1 near birth for one tip pair.
		Updates d1,d2 with fields.  aligns locations by index.  supposes index is the field, frame
		Example Usage:
		compute_initial_inout_angles=get_compute_initial_inout_angles(width,height)
		tbirth_values,angle_values=compute_initial_inout_angles(d1,d2)
		'''
		#compute displacement of d1 with pbc
		xy_values=np.array(list(zip(d1['x'],d1['y'])))
		dshifted=d1.shift(1).copy()
		# dshifted=d1.shift(-1).copy()
		xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
		dxy1_values=np.zeros_like(xy_values)+np.nan
		# compute displacement unit vector from tip 1 to tip 2
		xy_values=np.array(list(zip(d1['x'],d1['y'])))
		dshifted=d1.shift(1).copy()
		# dshifted=d1.shift(-1).copy()
		xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
		dxy1_values=np.zeros_like(xy_values)+np.nan
		#compute displacements between
		for j in range(dxy1_values.shape[0]):
			dxy1_values[j]=subtract_pbc(xy_next_values[j],xy_values[j])
		d1['dx']=dxy1_values[:,0]
		d1['dy']=dxy1_values[:,1]
		d1['dt']=d1['t'].diff().shift(-1).iloc[1:-1]
		# d1[['dx','dy','dt']]=d1[['x','y','t']].diff().shift(-1).iloc[1:-1]
		d1['displacement']=np.sqrt(d1['dx']**2+d1['dy']**2)
		d1['dx_hat']=d1['dx']/d1['displacement']
		d1['dy_hat']=d1['dy']/d1['displacement']

		#compute COM relative to d1
		d1['x2']=d2['x']
		d1['y2']=d2['y']
		xy1_values=np.array(list(zip(d1['x'],d1['y'])))
		xy2_values=np.array(list(zip(d1['x2'],d1['y2'])))
		dxy12_values=np.zeros_like(xy1_values)+np.nan
		#compute displacements between
		for j in range(dxy12_values.shape[0]):
			dxy12_values[j]=subtract_pbc(xy2_values[j],xy1_values[j])
		d1['comx']=xy1_values[:,0]+dxy12_values[:,0]/2
		d1['comy']=xy1_values[:,1]+dxy12_values[:,1]/2

		#compute the radial unit vector of d1
		d1['rx']=xy1_values[:,0]-d1['comx']
		d1['ry']=xy1_values[:,1]-d1['comy']
		d1['rx_hat']=d1['rx']/np.sqrt(d1['rx']**2+d1['ry']**2)
		d1['ry_hat']=d1['ry']/np.sqrt(d1['rx']**2+d1['ry']**2)

		#compute unsigned angle between velocity and the radial unit vector
		# compute dot product between tip 1 and tip 2
		cosine_series=d1['dx_hat']*d1['rx_hat']+d1['dy_hat']*d1['ry_hat']
		d1['theta']=np.arccos(cosine_series)   #radians
		d1.dropna(inplace=True)

		angle_values=d1['theta'].values
		tbirth_values=d1['t'].values-d1['t'].values[0] #ms
		return tbirth_values,angle_values
	return compute_initial_inout_angles

def compute_creation_events(input_fn,
								width,
								height,
								ds,
								range_threshold=1.,
								min_duration=20.,
								min_range=1.,
								round_t_to_n_digits=3,
								use_min_duration=True,
								use_grad_voltage=False,
								printing=True,
								**kwargs):
	'''input_fn is a string locating the directory of a _trajectories .csv file.
	Returns pandas.Dataframe instance.
	Example Usage:
	input_fn=f"/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-fk-200x200/param_set_8_ds_5.0_tmax_10_diffCoef_0.0005/Log/ic200x200.0.3_traj_sr_400_mem_0.csv"
	df_phases=compute_creation_events(input_fn,width,height,ds)#,**kwargs)
	'''
	# # original default values that worked for the Fenton-Karma model...
	#     range_threshold=1.#cm
	#     #compute measures near creation
	#     #input: df_ordered_interactions
	#     min_duration=20#150 #ms
	#     min_range=1. #cm
	#     use_min_duration=True#broken for false
	#     use_grad_voltage=False
	df=pd.read_csv(input_fn)
	DT = compute_DT(df, round_t_to_n_digits=round_t_to_n_digits)
	if printing:
		print(f"the time resolution is {DT} ms.")

	dsdpixel = ds / width  #cm per pixel
	DS = dsdpixel
	#compile jit
	compute_initial_inout_angles=get_compute_initial_inout_angles(width,height)
	# compute_angle_between_initial_velocities = get_compute_angle_between_initial_velocities(width, height)
	compute_ranges_between = get_compute_ranges_between(width=width,height=height)
	# birth_ranges,birth_ranges,DT=return_bd_ranges(input_fn,DS,round_t_to_n_digits=3)
	#compute interactions
	df_interactions = compute_df_interactions(input_fn, DS=DS)
	df_interactions.dropna(inplace=True)
	birth_ranges = DS * df_interactions.rT.values
	birth_ranges = DS * df_interactions.r0.values

	#filter any births that occur at ranges exceeding range_threshold
	boo = df_interactions.rT * DS < range_threshold
	df_ordered_interactions = df_interactions[boo].sort_values('Tavg',
															   ascending=False)

	#compute the phase time series between pid and pid_birthmate
	#find last zero for when phi1==0
	#compute phi2 at that time using linear interpolation
	#append phi2 to dphi_lst... proceed to the next particle
	pid_queue = list(df_ordered_interactions.pid.values)
	pid_birthmate_dict = dict(
		zip(pid_queue, list(df_ordered_interactions.pid_birthmate.values)))
	df_out_lst = []
	while len(pid_queue) > 0:
		pid = pid_queue.pop(0)
		pid_birthmate = pid_birthmate_dict[pid]
		try:
			pid_queue.remove(pid_birthmate)
			#extract d1,d2
			d1 = df[df.particle == pid].copy()
			d2 = df[df.particle == pid_birthmate].copy()
			d1.index = d1.frame
			d2.index = d2.frame
			#compute ranges between
			range_values = compute_ranges_between(d1, d2) * dsdpixel
			length = range_values.shape[0]
			#compute x,y values
			if use_grad_voltage:
				t_to_birth_values, phi1_values, phi2_values, phi_sum_values, phi_diff_values = compute_phase_angles_from_grad_voltage(
					d1, d2)
			else:
				t_vals = d1['t'].values[-length:]
				t_to_birth_values = t_vals - np.min(t_vals)

			nobs = range_values[1:-1].shape[0]
			if nobs > 1:
				#compute angle between velocities
				tbirth_values, theta_values = compute_initial_inout_angles(d1, d2)
				# tbirth_values, theta_values = compute_angle_between_initial_velocities(d1, d2)
				#align range_values and theta_values
				boo = ~np.isnan(theta_values)
				theta_values = theta_values[boo]
				assert (range_values.shape[0] == t_to_birth_values.shape[0])
				if use_min_duration:
					#filter by min_duration
					t_values = t_to_birth_values[1:-1]
					boo_keep = min_duration <= np.max(t_values) - np.min(
						t_values)
					#filter by min_range
					boo_keep &= min_range <= np.max(range_values)
				else:
					boo_keep = min_range <= np.max(range_values)

				if boo_keep:
					if use_grad_voltage:
						df_out = pd.DataFrame({
							'pid':
							pid + 0 * t_values.astype('int'),
							'pid_birthmate':
							pid_birthmate + 0 * t_values.astype('int'),
							'tbirth':
							np.around(t_values,round_t_to_n_digits),
							'phi1':
							np.abs(phi1_values)[1:-1],
							'phi2':
							np.abs(phi2_values)[1:-1],
							'phi_sum':
							np.abs(phi_sum_values)[1:-1],
							'phi_diff':
							np.abs(phi_diff_values)[1:-1],
							'r':
							range_values[1:-1],
							'theta':
							theta_values,
						})
					else:
						df_out = pd.DataFrame({
							'pid':
							pid + 0 * t_values.astype('int'),
							'pid_birthmate':
							pid_birthmate + 0 * t_values.astype('int'),
							'tbirth':
							np.around(t_values,round_t_to_n_digits),
							'r':
							range_values[1:-1],
							'theta':
							theta_values,
						})

					#append x,y values to list
					df_out_lst.append(df_out)
		except ValueError as e:  #for catching ValueError: list.remove(x): x not in list
			pass
	if printing:
		print(f"the number of trials appended to df_out_lst is {len(df_out_lst)}")
		assert (len(df_out_lst) > 0)
	if not (len(df_out_lst) > 0):
		return None
	# t_to_birth_values.shape, range_values.shape, theta_values.shape, d1[
	#     't'].values.shape, d2['t'].values.shape
	df_phases = pd.concat(df_out_lst)
	return df_phases


def save_creation_events(input_fn,
							 width,
							 height,
							 ds,
							 save_folder=None,
							 save_fn=None,
							 **kwargs):
	'''
	Example Usage:
	input_fn=f"/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-fk-200x200/param_set_8_ds_5.0_tmax_10_diffCoef_0.0005/Log/ic200x200.0.3_traj_sr_400_mem_0.csv"
	save_fn=save_creation_events(input_fn,width,height,ds,save_folder=None,save_fn=None)#,**kwargs)
	'''
	df_phases = compute_creation_events(input_fn, width, height, ds,**kwargs)
	if df_phases is None:
		return f"Warning: no annihilation events considered valid for trial located at \n\t {input_fn}"

	if save_folder is None:
		#save df_phases as csv
		save_folder = os.path.dirname(
			os.path.dirname(input_fn)) + '/creations'
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)
	os.chdir(save_folder)
	if save_fn is None:
		save_fn = os.path.basename(input_fn).replace('.csv',
													 '_creations.csv')
	df_phases.to_csv(save_fn, index=False)
	return os.path.abspath(save_fn)

#DONE: copy daskbag-routine for msd analysis
#TODO: dask-bag routine ^this shiz for a given input_folder that contains trajectory files
def get_routine_traj_to_creation(width=200,height=200,ds=5.,
		save_folder=None,save_fn=None,**kwargs):
	'''
	Example Usage:
	routine_traj_to_creation=get_routine_traj_to_creation(width=200,height=200,ds=5.)
	'''
	def routine_traj_to_creation(input_file_name):
		'''run_routine_log_to_msd returns where it saves .csv of creation events
		'''
		# # traj_fn = preprocess_log(fn)# wraps generate_track_tips_pbc
		# traj_fn = generate_track_tips_pbc(fn, save_fn=None)
		# input_file_name=traj_fn
		# output_file_name=input_file_name.replace('.csv',"_unwrap.csv")
		# retval_ignore= unwrap_trajectories(input_file_name, output_file_name)
		output_file_name=save_creation_events(input_file_name,width,height,ds,
			save_folder=save_folder,save_fn=save_fn,**kwargs)#,**kwargs)
		return output_file_name
	return routine_traj_to_creation
