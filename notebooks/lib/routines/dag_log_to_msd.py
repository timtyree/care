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

####################
# Finished Routines
####################
def run_routine_log_to_unwrapped_trajectory(input_file_name,sr, mem,L,  use_cache=True, DS=0.025, **kwargs):
	'''ic is a .csv file name of a tip log.'''
	traj_fn = os.path.abspath(input_file_name).replace('/Log','/trajectories').replace('log.csv', f'traj_sr_{sr}_mem_{mem}.csv')
	#save results
	input_fn=input_file_name
	folder_name=os.path.dirname(input_fn)
	dirname = folder_name.split('/')[-1]
	save_folder = folder_name.replace(dirname,'trajectories_unwrap')
	if not os.path.exists(save_folder):
	    os.mkdir(save_folder)
	os.chdir(save_folder)
	output_fn=os.path.basename(input_fn).replace('.csv','_emsd.csv')
	df_msd.to_csv(output_fn, index=False)
	return os.path.abspath(output_fn)

##deprecated
# output_file_name = traj_fn.replace('/trajectories','/trajectories_unwrap').replace('.csv',"_unwrap.csv")
# if not use_cache or not os.path.exists(traj_fn):
#     traj_fn = generate_track_tips_pbc(input_file_name, save_fn=traj_fn,sr=sr,mem=mem, width=L,height=L)#,**kwargs)
# if not use_cache or not os.path.exists(output_file_name):
#     retval_ignore = unwrap_trajectories(traj_fn, output_file_name,width=L,height=L,DS=DS)
# return os.path.abspath(output_file_name)

def run_routine_unwrapped_trajectories_to_diffcoeff_summary(file_name_list,trial_folder_name,L,DS,use_cache=True,n_tips=1,**kwargs):
    '''file_name_list is a list of strings locating files ending in _unwrap.csv'''
    input_file_name=file_name_list[0]
    #determine DT
    df=pd.read_csv(input_file_name)
    DT=df.t.head(2).diff().dropna().values[0]
    dirname = os.path.dirname(input_file_name).split('/')[-1]
    folder_name=os.path.dirname(input_file_name)
    save_folder = folder_name.replace(dirname,'msd')
    emsd_fn = os.path.join(save_folder,f"emsd_longest_by_trial_tips_ntips_{n_tips}.csv")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    os.chdir(save_folder)
    if not use_cache or not os.path.exists(emsd_fn):
        #compute the ensemble mean squared displacements for each trial
        emsd_fn=generate_msd_figures_routine_for_list(file_name_list, n_tips, DT=DT,DS=DS,
                                                               output_file_name=emsd_fn,L=L,                                                       save_folder=save_folder,**kwargs)
    #compute the diffusion coefficient summary
    os.chdir(os.path.dirname(emsd_fn))
    dir_out=os.path.join(save_folder,"diffcoeff_summary_"+os.path.basename(emsd_fn))
    if not use_cache or not os.path.exists(dir_out):
        dir_out = dag_a_postprocess(emsd_fn=emsd_fn,
                                trial_folder_name=trial_folder_name,
                                dir_out=dir_out,**kwargs)
    return dir_out

def workflow_reduce_logs_to_diffcoeff_summary(input_fn_lst, L,DS,use_cache_0=True,use_cache_1=False, npartitions=2, **kwargs):
    print(f"the total number of trials recorded is {len(input_fn_lst)}")
    def routine(fn):
        output_file_name=run_routine_log_to_unwrapped_trajectory(fn, use_cache=use_cache_0,L=L,**kwargs)
        return output_file_name
    b = db.from_sequence(input_fn_lst, npartitions=npartitions).map(routine)
    file_name_list = list(b)
    print(f'reducing {len(file_name_list)} unwrapped trajectories files to a single row describing the diffusion coefficient of spiral tips for that trial...')
    #reduce reduce_logs_to_diffcoeff_summary
    df_out= run_routine_unwrapped_trajectories_to_diffcoeff_summary(file_name_list,use_cache=use_cache_1,DS=DS,L=L,**kwargs)
    return df_out

def get_all_logs(trial_folder,trial_folder_name):
    cwd=os.getcwd()
    file=f"{trial_folder}/Log/ic_200x200.001.12_log.csv"
    trgt='log.csv'
    assert(file[-len(trgt):]==trgt)
    input_fn_lst=get_all_files_matching_pattern(file,trgt)
    # print(len(input_fn_lst))
    os.chdir(cwd)
    return input_fn_lst

def gen_diffcoeff_summary(trial_folder_name,L,DS,ic_suite_fn,**kwargs):
    #find all log files for each trial folder
    trial_folder=f"{ic_suite_fn}/{trial_folder_name}"
    input_fn_lst= get_all_logs(trial_folder,trial_folder_name)
    print(f"{trial_folder_name} contains {len(input_fn_lst)} tip logs")
    #reduce that trial folder to a single diffusion coefficient row
    diffcoeff_summary_fn=workflow_reduce_logs_to_diffcoeff_summary(
        input_fn_lst,L=L,DS=DS,trial_folder_name=trial_folder_name,
        use_cache_0=True,**kwargs)
    return diffcoeff_summary_fn

# def get_all_trial_folders_not_archived(ic_suite_fn):
#     os.chdir(ic_suite_fn)
#     dir_lst=os.listdir()
#     trial_folder_name_lst=[]
#     for dir_run in dir_lst:
#         # dir_run=dir_lst[0]
#         trial_folder_name=dir_run
#         #test if trial_folder_name is not a folder
#         boo  = os.path.isdir(trial_folder_name)
#         #test if trial_folder_name starts with ic
#         boo &=trial_folder_name[:2]!='ic'
#         #test if trial_folder_name contains archiv
#         boo &=trial_folder_name.find('archiv')==-1
#         #if not, append it to the trial_folder_name_lst
#         if boo:
#             trial_folder_name_lst.append(trial_folder_name)
#     return trial_folder_name_lst

# #find all log files for a given trial
# trial_folder_name='ds_5_param_set_3'
# trial_folder=f"{nb_dir}/Data/initial-conditions-suite-2/{trial_folder_name}"
# input_fn_lst= get_all_logs(trial_folder,trial_folder_name)
# print(len(input_fn_lst))

#DONE: test workflow_reduce_logs_to_diffcoeff_summary runtime on trial_folder with cached results
# pipeline to run up to an individual data run (workflow_reduce_logs_to_diffcoeff_summary)

def diffcoeff_table_gener(ic_suite_fn,L,DS,trial_folder_name_lst,dict_kwargs_trial,npartitions=2,**kwargs_table):
    #generate the diffusion table
    for trial_folder_name in trial_folder_name_lst:
        kwargs_trial= dict_kwargs_trial[trial_folder_name]
        kwargs_trial.update(kwargs_table)
        print(f'beginning diffusion analysis of {trial_folder_name}...')
        diffcoeff_summary_fn= gen_diffcoeff_summary(trial_folder_name,L=L,ic_suite_fn=ic_suite_fn,npartitions=npartitions,DS=DS,**kwargs_trial)
    #reduce
    retval= gen_diffcoeff_table(ic_suite_fn,trial_folder_name_lst=trial_folder_name_lst,**kwargs_table)
    return retval

def msd_fig_gener(input_file_name,n_tips=1):
    trgt='_unwrap.csv'
    assert(input_file_name[-len(trgt)]==trgt)
    #Generate msd figures for input_file_name
    dirname = os.path.dirname(input_file_name).split('/')[-1]
    folder_name=os.path.dirname(input_file_name)
    save_folder = folder_name.replace(dirname,'msd')
    output_file_name = f"emsd_longest_by_trial_tips_ntips_{n_tips}.csv"
    #determine DT
    df=pd.read_csv(input_file_name)
    DT=df.t.head(2).diff().dropna().values[0]
    retval=gen_msd_figs(file_out,n_tips, DT)#, V_thresh=0.4)
    return retval
