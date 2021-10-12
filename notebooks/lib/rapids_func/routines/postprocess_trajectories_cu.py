#unwraps and smooths trajectories on the gpu
#Programmer: Tim Tyree
#Date: 10.6.2021
import numpy as np, cupy as cp, numba.cuda as cuda, cudf, os, re, dask.bag as db, time
from .. import *
from .unwrap_and_smooth_trajectories_cu import *
from ..utils.operari import get_all_files_matching_pattern
from ..measure.annihilations_cu import routine_compute_radial_velocities_pbc_cu

def routine_postprocess_trajectory_folder(input_fn,DT,tavg1, tavg2=24,
                                        max_dtmax_thresh = 0,      #ms
                                        max_Rfinal_thresh = 0.2,   #cm
                                        min_duration_thresh = 40,  #ms
                                        npartitions=None,
                                        width=200,
                                        height=200,
                                        ds=5,
                                        use_drop_shorter_than=True,
                                        drop_shorter_than=50, #ms
                                        tmin=100., #ms
                                        pid_col='particle',
                                        t_col='t',
                                        printing=False,**kwargs):
    if npartitions is None:
        npartitions=os.cpu_count()
    input_fn_lst=get_all_files_matching_pattern(input_fn,trgt='.csv')
    input_fn_lst=[fn for fn in input_fn_lst if fn.find('_unwrap.csv')==-1]
    if printing:
        print(f"running return_moving_average_of_pbc_trajectories_and_save on {len(input_fn_lst)} files...")

    def routine(input_fn):
        try:
            return return_moving_average_of_pbc_trajectories_and_save(
                input_fn, tavg1, pid_col, t_col, DT, width, height,
                use_drop_shorter_than, drop_shorter_than, tmin, printing)
        except Exception as e:
            return None

    #perform moving average on all files in list
    bag = db.from_sequence(input_fn_lst, npartitions=npartitions).map(routine)
    start = time.time()
    retval_lst = list(bag)
    if printing:
        print(f"the run time was {(time.time()-start)/60:.2f} minutes.")
    fn_lst=[fn for fn in retval_lst if fn is not None]
    if printing:
        print(f"the number of successfully smoothed trajectory files was {len(fn_lst)}")

    def routine2(input_fn):
        try:
            return routine_compute_radial_velocities_pbc_cu(input_fn,
                                                         tavg2=tavg2,
                                                         width=width,  #for handling periodic boundary conditions (pbc)
                                                         height=height, #for handling periodic boundary conditions (pbc)
                                                         ds=ds,     #domain size in cm.  set to width to have nulled effect
                                                         max_dtmax_thresh = max_dtmax_thresh,      #ms
                                                         max_Rfinal_thresh = max_Rfinal_thresh,   #cm
                                                         min_duration_thresh = min_duration_thresh,  #ms
                                                         round_t_to_n_digits=7,
                                                         trial_col='event_id_int',  #TODO: generalize kwarg handling of trial_col
                                                         pid_col=pid_col,
                                                         t_col=t_col,
                                                         use_tavg2=True,
                                                         save_df_pairs=True,
                                                         printing=False,
                                                         testing=True)#,**kwargs)
        except Exception as e:
            return e

    #perform annihilation identification and smoothing via a savitzky-golay filtration on all files in list
    bag = db.from_sequence(fn_lst, npartitions=npartitions).map(routine2)
    start = time.time()
    retval_lst = list(bag)
    if printing:
        print(f"the run time was {(time.time()-start)/60:.2f} minutes.")
    fn_lst2=[fn for fn in retval_lst if fn is not None]
    if printing:
        print(f"the number of trajectory files processed to annihilation files was {len(fn_lst2)}")
    return fn_lst,fn_lst2

#for loop implementation has an expected run time of ~4 minutes for LR data with DT=0.5
#for loop implementation has an expected run time of ~4 minutes for FK data with DT=0.4
