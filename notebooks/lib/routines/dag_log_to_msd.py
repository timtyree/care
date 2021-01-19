from ..my_initialization import *
from .compute_msd import *
# from ..routines.compute_msd import *
# from .compute_diffcoef import *
from ..routines.compute_diffcoef import *
# from ..routines.track_tips import *

def run_routine_log_to_msd(fn):
	'''ic is a .csv file name of a tip log'''
	# traj_fn = preprocess_log(fn)# wraps generate_track_tips_pbc
	traj_fn = generate_track_tips_pbc(fn, save_fn=None)
	input_file_name=traj_fn
	output_file_name=input_file_name.replace('.csv',"_unwrap.csv")
	retval_ignore= unwrap_trajectories(input_file_name, output_file_name)
	return output_file_name

def gen_msd_figs(file,n_tips=1,DT=1.):#,V_thresh):
	"""computes mean squared displacement and saves corresponding plots.
	DT is the tim1e between two spiral tip observations in milliseconds.
	file is a string locating in a folder with files ending in _unwrap.csv
	n_tips is the number of tips"""
	return generate_msd_figures_routine(file,n_tips, DT=DT, V_thresh=None)

def gen_diffcoeff_figs(input_file_name):
    '''file is a string starting with diffcoeff_'''
    return generate_diffcoeff_figures(input_file_name,tau_min=.15,tau_max=0.5,saving=True,
            R2_thresh=0.94,duration_thresh=2.5,fontsize=22,figsize_2=(15,4.5))

def produce_one_csv(list_of_files, file_out):
   # Consolidate all csv files into one object
   result_obj = pd.concat([pd.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")

def gen_diffcoeff_table(input_folder='/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2'):
    diffcoeff_fn_base=input_folder+"/ds_5_param_set_4/msd/diffcoeff_emsd_longest_by_trial_tips_ntips_1_Tmin_0.15_Tmax_0.5.csv"
    foo_dfn=lambda trial_folder_name:diffcoeff_fn_base.replace('ds_5_param_set_4',trial_folder_name)
    #list all collections of trials
    trial_folder_name_lst=[
        'ds_5_param_set_8_fastkernel_V_0.4_archive',
        'ds_5_param_set_8_fastkernel_V_0.5_archive',
        'ds_5_param_set_8_fastkernel_V_0.6_archive',
        'ds_5_param_set_8_og',
        'ds_5_param_set_4',
        'ds_2_param_set_8',
        'ds_1_param_set_8',
    ]
    #generate figures
    fn_lst = [foo_dfn(str) for str in trial_folder_name_lst]
    fn_lst_out=[]
    for fn in fn_lst:
        retval=gen_diffcoeff_figs(input_file_name=fn)
        fn_lst_out.append(retval)

    # #save df_out in initial-conditions-2/
    # input_file_name=fn_lst_out[0]
    # sl=input_file_name.split('/')
    # trial_folder_name=sl[-4]
    # nb_dir='/home/timothytyree/Documents/GitHub/care/notebooks'
    save_folder_table = input_folder#os.path.join(nb_dir,f'/Data/initial-conditions-suite-2')
    file_out=save_folder_table+'/avg-diffcoeff-table.csv'
    produce_one_csv(fn_lst_out, file_out)
    print(f'\n final csv saved in:
    \t {file_out}')
