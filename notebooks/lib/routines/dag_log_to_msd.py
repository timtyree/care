from ..my_initialization import *
from .compute_msd import *
# from ..routines.compute_msd import *
# from .compute_diffcoef import *
from ..routines.compute_diffcoef import *
# from ..routines.track_tips import *

def run_routine_log_to_msd(fn):
	'''run_routine_log_to_msd returns where it saves unwrapped trajectories
	fn is a .csv file name of a raw tip log
	TODO: fix function nominclature for run_routine_log_to_msd everywhere/functionally
	'''
	# traj_fn = preprocess_log(fn)# wraps generate_track_tips_pbc
	traj_fn = generate_track_tips_pbc(fn, save_fn=None)
	input_file_name=traj_fn
	output_file_name=input_file_name.replace('.csv',"_unwrap.csv")
	retval_ignore= unwrap_trajectories(input_file_name, output_file_name)
	return output_file_name

def gen_msd_figs(file,n_tips=1,**kwargs):#,V_thresh):
	"""computes mean squared displacement and saves corresponding plots.
	DT is the tim1e between two spiral tip observations in milliseconds.
	file is a string locating in a folder with files ending in _unwrap.csv
	n_tips is the number of tips"""
	return generate_msd_figures_routine(file,n_tips,**kwargs)#, V_thresh=None)

def gen_diffcoeff_figs(input_file_name,trial_folder_name, **kwargs):
	'''file is a string starting with diffcoeff_'''
	return generate_diffcoeff_figures(input_file_name, trial_folder_name,**kwargs)

def produce_one_csv(list_of_files, file_out):
   # Consolidate all csv files into one object
   df = pd.concat([pd.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   if df.columns[0]=='Unnamed: 0':
	   df.drop(columns=['Unnamed: 0'])
   df.to_csv(file_out, index=False, encoding="utf-8")

def gen_diffcoeff_table(input_folder,trial_folder_name_lst=None, tau_min=0.15,tau_max=0.5,**kwargs):
	'''Example input_folder is at {nbdir}/Data/initial-conditions-suite-2'''
	diffcoeff_fn_base=input_folder+f"/ds_5_param_set_4/msd/diffcoeff_emsd_longest_by_trial_tips_ntips_1_Tmin_{tau_min}_Tmax_{tau_max}.csv"
	foo_dfn=lambda trial_folder_name:diffcoeff_fn_base.replace('ds_5_param_set_4',trial_folder_name)
	if trial_folder_name_lst is None:
		#list some collections of trials
		trial_folder_name_lst=[
			'ds_5_param_set_8_fastkernel_V_0.4_archive',
			'ds_5_param_set_8_fastkernel_V_0.5_archive',
			'ds_5_param_set_8_fastkernel_V_0.6_archive',
			'ds_5_param_set_8_og',
			'ds_5_param_set_4',
			'ds_2_param_set_8',
			'ds_1_param_set_8']
	#generate figures
	fn_lst = [foo_dfn(str) for str in trial_folder_name_lst]
	fn_lst_out=[]
	for n,fn in enumerate(fn_lst):
		trial_folder_name=trial_folder_name_lst[n]
		retval=gen_diffcoeff_figs(input_file_name=fn, trial_folder_name=trial_folder_name, **kwargs)
		fn_lst_out.append(retval)

	# #save df_out in initial-conditions-2/
	# input_file_name=fn_lst_out[0]
	# sl=input_file_name.split('/')
	# trial_folder_name=sl[-4]
	# nb_dir='/home/timothytyree/Documents/GitHub/care/notebooks'
	save_folder_table = input_folder#os.path.join(nb_dir,f'/Data/initial-conditions-suite-2')
	file_out=save_folder_table+'/avg-diffcoeff-table.csv'
	produce_one_csv(fn_lst_out, file_out)
	# print(f'\n output csv saved in:\t {os.path.dirname(file_out)}')
	return fn_lst_out

# # #TODO: use the daskbag motif to accelerate the pipeline before generate_msd_figures_routine_for_list
# #all CPU version
# b = db.from_sequence(input_fn_lst, npartitions=9).map(routine)
# start = time.time()
# retval = list(b)
# print(f"run time for generating birth-death rates from file_name_list: {time.time()-start:.2f} seconds.")
# beep(10)

def dag_a_postprocess(emsd_fn,trial_folder_name,dir_out,**kwargs):
	'''returns string locating the diffcoef_summary for trial in trial_folder_name'''
	input_file_name=emsd_fn
	fn2= compute_diffusion_coeffs(input_file_name,**kwargs)
	input_file_name=os.path.abspath(fn2)
	retval= generate_diffcoeff_figures(input_file_name,trial_folder_name,dir_out=dir_out,**kwargs)
	return retval
