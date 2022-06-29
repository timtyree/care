# annihilation_analysis_cu.py
#Programmer: Tim Tyree
#Date: 6.29.2022
import shutil, os, pandas as pd, numpy as np, cudf, cupy
from ...utils.parquetio import load_parquet_by_trial_num
from ...utils.hide_print import HiddenPrints
from ...routines.track_tips import compute_track_tips_pbc
from .unwrap_and_smooth_trajectories_cu import comp_moving_avg_pbc_trajectories_cu
from ..measure.annihilations_cu import compute_radial_velocities_of_annihilations_cu
from ..measure.alignment_cu import align_timeseries_simple
from ..utils.utils_df import comp_lifetimes_by, copy_df_as_pandas

def routine_annihilation_analysis_pbc(trial_num,
    log_folder_parquet='/home/timothytyree/Documents/GitHub/care/notebooks/Data/from_wjr/tippos_per_001_log/',
    DS=0.025, #cm/pixel
    DT=1., #ms/frame
    width=200.,  #pixels
    height=200., #pixels
    tavg1=0.,  #moving average window, in ms
    tavg2=14., #ms #savgol_filter time window performed on R  #12 ms was no longer smooth
    min_termination_time=100.,
    min_duration_thresh=1., #ms #minimum lifetime for a spiral tip position to be considered
    max_Rfinal_thresh=0.5,  #cm
    max_dtmax_thresh= 0.,   #cm # max disagreement between tmax for _self relative to _other
    round_t_to_n_digits=7,
    t_col='t',
    pid_col='particle',
    trial_col='trial_num',
    use_tavg2=True, #unsmoothed R is preserved through R_nosavgol
    testing=True,
    printing=True,**kwargs):
    """routine_annihilation_analysis_pbc loads spiral tip locations from cache,
    computes the trajectories while enforcing periodic boundary conditions,
    optionally smooths trajectories while enforcing periodic boundary conditions, on the gpu,
    solves for annihilation events between pairs of particles subject to periodic boundary conditions, also on the gpu,
    and computes a linear time adjustment so the time before annihilation, t', cooresponds to R=0 when t'=0.
    routine_annihilation_analysis_pbc returns a dictionary containing dataframes at each of these steps, dict_msr.
    the lifetimes of particles are included in dict_msr

    Example Usage:
dict_msr=routine_annihilation_analysis_pbc(trial_num=638,
    log_folder_parquet='/home/timothytyree/Documents/GitHub/care/notebooks/Data/from_wjr/tippos_per_001_log/',
    DS=0.025, #cm/pixel
    DT=1., #ms/frame
    width=200.,  #pixels
    height=200., #pixels
    tavg1=0.,  #moving average window, in ms
    tavg2=14., #ms #savgol_filter time window performed on R  #12 ms was no longer smooth
    min_termination_time=100.,
    min_duration_thresh=1., #ms #minimum lifetime for a spiral tip position to be considered
    max_Rfinal_thresh=0.5,  #cm
    max_dtmax_thresh= 0.,   #cm # max disagreement between tmax for _self relative to _other
    round_t_to_n_digits=7,
    t_col='t',
    pid_col='particle',
    trial_col='trial_num',
    use_tavg2=True, #unsmoothed R is preserved through R_nosavgol
    testing=True,
    printing=True)
    """
    #load spiral tip positions from cache
    g=load_parquet_by_trial_num(trial_num=trial_num,folder_parquet=log_folder_parquet)
    termination_time=g[t_col].max()
    if termination_time<min_termination_time:
        if printing:
            print(f"Warning: {termination_time=} < {min_termination_time=}")
        return None
    ###################################
    # Track trajectories
    ###################################
    #compute spiral tip trajectories, as before
    #DONE: find the mem, sr, that i used before
    ds=DS*width #cm side length for the whole domain
    sr=3*width #search range i(n pixels).  ignore for now.
    mem=0  #memory for filling in missing data (in frames)
    #DONE: generate tip tracks using pbc
    with HiddenPrints():
        traj = compute_track_tips_pbc(g, mem=mem, sr=sr, width=width, height=height)
    if testing:
        #test max times agree
        assert g['t'].max()==traj['t'].max()
    ##DONE(optional): save traj to cache
    # traj_dir=os.path.abspath(input_fn)+'_log.csv'
    # df_loc.to_csv(log_dir,index=False)
    traj.drop_duplicates(inplace=True)
    pid_lst = sorted(set(traj[pid_col].values))

    printing_lifetimes_traj=True
    if printing and printing_lifetimes_traj:
        #compute the lifetimes
        #print summary stats on particle lifetimes for one input folder
        dft=traj.groupby(pid_col)[t_col].describe()
        df_lifetimes=-dft[['max','min']].T.diff().loc['min']
        print(f"termination time was {traj[t_col].max():.2f} ms")
        print(f"printing summary stats on particle lifetimes:")
        print(df_lifetimes.describe())
        # print(df_lifetimes.head(10))
        #print("\nPlease make a manual decision about minimum_lifetime, crop_start_by, and crop_end_by")

    # #DONE: test that termination times still agree
    # assert df_traj['t'].max()==traj['t'].max()
    # # traj['t'].max()
    # # df_traj['n'].max()
    # #DONE: verified that these unwrapped trajectories exhibit no jumps greater than 20 pixels per ms
    # jump_thresh=20
    # # DONT: truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    # df=df_traj.copy()# = pd.concat([unwrap_traj_and_center(traj[traj[pid_col]==pid].copy(), width, height, DS) for pid in pid_lst])
    # #truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    # df_lst = []
    # for pid in  pid_lst:#[2:]:
    #     d = df[(df.particle==pid)].copy()
    #     x_values, y_values = d[['x','y']].values.T
    #     index_values = d.index.values.T
    #     jump_index_array, spd_lst = find_jumps(x_values, y_values,
    #                                            width=width, height=height,
    #                                            DS=DS,DT=DT, jump_thresh=jump_thresh)#, **kwargs)#.25)
    #     if len(jump_index_array)>0:
    #         print(f"jump for {pid=}!  {np.max(spd_lst)=:.4f}")
    # # #         ji = jump_index_array[0]
    # # #         d.drop(index=index_values[ji:], inplace=True)
    # #     df_lst.append(d)
    # # df_traj = pd.concat(df_lst)
    # # df_traj['t'] = df_traj.t.round(round_t_to_n_digits)
    #DONE: found a previous parameter setting (deprecated)
    #i don't think these are connected to anything anymore. maybe their complicated function got pruned...
    #src: http://localhost:8890/notebooks/Processing%20a%20Folder%20of%20Tip%20Logs.ipynb
    # minimum_lifetime=40. #ms
    # crop_start_by=0#40
    # crop_end_by=150#40
    # #was changed to
    # # minimum_lifetime=10.#40. #ms
    # crop_start_by=0#0 #ms
    # crop_end_by=0#150 #ms

    ###################################
    # Smooth trajectories
    ###################################
    navg1=int(tavg1/DT)
    #compute moving average respecting pbc (gpu-accelerated, estimated run time ~1 second for whole termination event)
    df=comp_moving_avg_pbc_trajectories_cu(traj,navg1=navg1,width=width,height=height)#,t_col='t',pid_col='particle',printing=True)#,**kwargs)
    df['event_id_int']= int(trial_num) #int(event_id_int)
    df['event_id']= df[trial_col] + df[pid_col] / (1.+df[pid_col].max())
    df.reset_index(inplace=True)#,drop=True)
    df_smoothed=df.copy()
    df=df.to_pandas().copy()
    # df_smoothed.head()

    ###################################
    # Pair annihilation events
    ###################################
    #previous particle pairing settings
    #src of kwargs: http://localhost:8890/notebooks/fast%20estimation%20of%20particle%20properties.ipynb
    # max_dtmax_thresh = 0      #ms
    # max_Rfinal_thresh = 0.2   #cm
    # min_duration_thresh = 40  #ms
    # DS # as input

    navg2 = int(tavg2 / DT)
    if not (navg2 % 2 == 1):  #odd navg2 is required by savgol_filter
        navg2 += 1
        tavg2 = np.around(navg2 * DT, round_t_to_n_digits)
    if use_tavg2:
        if printing:
            print(f'effective savistzky-golay time window in use: {tavg2} ms')

    #compute radial velocities of annihilations
    df=df_smoothed.copy()
    df.index = df.index.astype(np.int64)
    # df['event_id_int']=df['particle'] #event_id_int is used by compute_radial_velocities_of_annihilations_cu as trial_col
    df['event_id_int']=df[trial_col] #event_id_int is used by compute_radial_velocities_of_annihilations_cu as trial_col
    dfr, df_pairs = compute_radial_velocities_of_annihilations_cu(df.copy(),
        navg2=navg2, #num. frames to average over.  navg2 must be an odd, positive integer.
        DS=DS,    #cm/pixel
        DT=DT,   #ms/frame
        width=width,
        height=height,
        max_dtmax_thresh=max_dtmax_thresh,  #ms
        max_Rfinal_thresh=max_Rfinal_thresh,  #cm
        min_duration_thresh=min_duration_thresh,  #ms
        use_tavg2=use_tavg2,
        pid_col=pid_col,
        trial_col=trial_col,
        use_dask=False,
        printing=printing,
        testing=False)#testing)
    dfr.reset_index(inplace=True)
    # dfr.head()
    #DONE: print the lifetime of all pid_lst_stumps
    printing_lifetimes_pairs=True
    if printing and printing_lifetimes_pairs:
        min_num_obs=4
        print(f"investigating the annihilation pair stumps involving fewer than {min_num_obs=} computable time points")
        print(f"for dfr, the stumps are:")
        df_lifetimes_pairs=df_pairs['duration']
        d=(df_lifetimes_pairs/DT).round()
        id_lst_stumps=[int(x) for x in d[d<min_num_obs].index.values]
        for i,pid in enumerate(id_lst_stumps):
            print(f"apparent minimum lifetime of annihilation pair #{pid}:\t{df_lifetimes.loc[pid]} ms.")


    printing_stumps=True
    #(optionally) investigate any stumps
    if printing and printing_stumps:
        #determine remaining stumps
        min_num_obs=5 #for pid_lst_stumps only
        d=dfr.groupby(by='index_pairs')['t'].count()
        pid_lst_stumps=[int(x) for x in d[d<min_num_obs].index.values]

        #print the lifetime of all pid_lst_stumps
        df_lifetimes=comp_lifetimes_by(df=traj,t_col='t',by='particle',pid_lst=None,printing=False)#,**kwargs)
        for i,pid in enumerate(pid_lst_stumps):
            print(f"apparent lifetime of particle {pid}:\t{df_lifetimes.loc[pid]} ms.")

        dft=df_smoothed.reset_index().groupby(pid_col).describe()[t_col]
        df_lifetimes_smoothed=-dft[['max','min']].T.diff().loc['min']
        #DONE: print the lifetime of all pid_lst_stumps
        for i,pid in enumerate(pid_lst_stumps):
            print(f"apparent lifetime of particle {pid}:\t{df_lifetimes_smoothed.loc[pid]} ms.")

    ###################################
    # Align annihilation events
    ###################################
    #linearly shift the times by tshift
    # df_R=dfr.to_pandas()
    # df_P=df_pairs.to_pandas()
    max_num_obs_align=3
    if printing:
        print(f"considering no more than {DT*max_num_obs_align:.0f} ms leading up to annihilation for adjustment of tdeath")
    df_R,df_P=align_timeseries_simple(dfr,df_pairs,
                                P_col='index_pairs',
                                R_col='R_nosavgol',
                                T_col='tdeath',
                                T_col_out='talign',
                                max_num_obs_align=max_num_obs_align)

    #compute raw lifetimes
    df_lifetimes_traj=comp_lifetimes_by(df=traj,t_col='t',by='particle',pid_lst=None,printing=False)#,**kwargs)

    #format output as dict
    dict_msr=dict(
        trial_num=trial_num,
        df_R=df_R,
        df_pairs=df_P,
        df_smoothed=copy_df_as_pandas(df_smoothed),
        df_traj=traj,
        df_lifetimes_traj=df_lifetimes_traj
    )
    return dict_msr
