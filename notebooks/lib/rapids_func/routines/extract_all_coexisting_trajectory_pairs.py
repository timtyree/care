import dask.bag as db, time
import numpy as np, cudf, cupy as cp, dask_cudf
from .. import *

#DONE: computes the dataframe identifying the largest start/end times for all pairs of particles
#TODO: wrap all this into a function
#TODO(later): add support to parallelize this over dask with npartitions determined locally.  measure any performance boost
def extract_all_trajectory_pairs_cu(df,df_pairs,pid_col='particle',t_col='t',trial_col='event_id_int',npartitions=None,use_dask=None,printing=True,**kwargs):
    '''returns the dataframe containing the timeseries data for pairs of particles in df_pairs with values aligned by t_col
    returns a cudf.DataFrame instance that identifies pairs of particles that coexist along with their start and end times, for all rows uniquely identified by trial_col
    df is a cudf.DataFrame instance with a unique index for each row and a particle id recorded in the column indicated by pid_col
    option to parallelize this gpu accelerated task via dask with npartitions determined locally.
    Example Usage:
    df_traj=extract_all_xy_trajectory_pairs_cu(df,df_pairs,pid_col=pid_col,t_col=t_col, trial_col=trial_col)
    '''
    dff=df
    #TODO: test whether .sort_values is needed here.  if not, remove it
    df_pairs=df_pairs.sort_values([trial_col,'pid_self'], ascending=True).copy()
    #iterate over all events and compute the dataframe identifying the largest start/end times for all pairs of particles
    event_id_int_values=dff[trial_col].drop_duplicates().values
    num_trials=event_id_int_values.get().shape[0]
    #define cpu resource allocation
    if npartitions is None:
        npartitions=np.max((int(os.cpu_count()/2),1))
        if npartitions/2>num_trials:
            npartitions=1
    if use_dask is None:
        use_dask=npartitions>1

    if printing:
        print(f"using {npartitions} cores to extract pairs of particles that coexist over {event_id_int_values.shape[0]} independent trials...")

    #DONE:
    #TODO(later): measure any performance boost
    if not use_dask:
        df_traj_lst=[]
        for event_id_int in event_id_int_values.get():
            dfff=df[df[trial_col]==event_id_int]
            df_traj=extract_xy_trajectory_pairs_cu(dfff,df_pairs,pid_col=pid_col,t_col=t_col)
            #augment df_intersecting_pairs with any columns desired
            df_traj[trial_col]=event_id_int
            #record
            df_traj_lst.append(df_traj)
    else:
        #define routine locally with dff baked
        def routine(event_id_int):
            try:
                dfff=df[df[trial_col]==event_id_int]
                df_traj=extract_trajectory_pairs_cu(dfff,df_pairs,pid_col=pid_col,t_col=t_col)
                #augment df_intersecting_pairs with any columns desired
                df_traj[trial_col]=event_id_int
                return df_traj
            except Exception as e:
                return e

        #feed routine to a daskbag
        bag = db.from_sequence(list(event_id_int_values.get()), npartitions=npartitions).map(routine)
        start = time.time()
        df_traj_lst = list(bag)
        if printing:
            print(f"the run time was {(time.time()-start)/60:.2f} minutes.")
            print(f"the number of successfully processed trials was {len(df_traj_lst)}")

    df_traj_out=cudf.concat(df_traj_lst)
    return df_traj_out


# ###############
# # Example Usage
# ###############
# if __name__=='__main__':
#     #define parameters
#     trial_col='event_id_int'
#     pid_col='particle'
#     t_col='t'
#     width=200
#     height=200
#     tmin=100.
#     printing=True
#
#     tavg2=0.
#     minimum_duration_threshold=tavg2
#     minimum_duration_threshold=25 #ms
#     #recall a couple postprocessed single trials
#     input_fn=f"/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-3-LR/param_qu_tmax_30_Ko_5.4_diffCoef_0.0005_dt_0.5/smoothed_trajectories_navg_8/ic002.11_traj_sr_600_mem_0_smoothed.csv"
#     input_fn2=f"/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-3-LR/param_qu_tmax_30_Ko_5.4_diffCoef_0.0005_dt_0.5/smoothed_trajectories_navg_8/ic002.12_traj_sr_600_mem_0_smoothed.csv"
#     input_fn_lst=[input_fn,input_fn2]
#     df=dask_cudf.read_csv(input_fn_lst).compute()
#     type(df)
#
#     df=df.sort_values([trial_col,pid_col, t_col], ascending=True).copy()
#     DT=get_DT_cu(df,t_col,pid_col)
#     navg2=int(tavg2/DT)
#     if printing:
#         print(f"DT={DT} ms")
#
#     #compute the intermediate dataframe of particle start/end times
#     grouped=df.groupby([trial_col,pid_col])
#     dft=grouped[t_col]
#     dff=cudf.DataFrame({
#         'tmin':dft.min(),
#         'tmax':dft.max(),
#     })
#     dff.reset_index(inplace=True)
#     # dff.head()
#
#     #test that duration is strictly positive
#     assert ( (df_intersecting_pairs_all['duration']>0).all() )
#     if printing:
#         print (f"verified that the duration is strictly positive for all pairs of tips that coexist nontrivially")
#
#     #plot histogram of durations
#     fontsize=20
#     #DONE: histograms sanity check df_intersecting_pairs
#     #DONE: verify that routine gives different values for different event_id_int
#     fig,ax=plt.subplots(figsize=(7,4))
#     yv=np.linspace(0,0.02,10)
#     ax.plot(minimum_duration_threshold+0.*yv,yv,'gray',lw=2, linestyle='dashed', label='threshold')
#     trial_col_lst=sorted(df_intersecting_pairs_all[trial_col].drop_duplicates().values.get())
#     for trial in trial_col_lst[:5]:
#         df_intersecting_pairs_all.query(f"{trial_col} == {trial}")['duration'].to_pandas().hist(density=True,bins=50,ax=ax,label=trial)#,color='event_id_int')
#     format_plot(ax, xlabel='duration of pair coexistance (ms)', ylabel='probability density', fontsize=fontsize, use_loglog=False)
#
#     ax.legend(fontsize=fontsize)
#     plt.tight_layout()
#     plt.show()
#     #DONE: wrap generation of df_intersecting_pairs into a function
#     #DONE: include event_id_int outside ^that function
#     #DONE: accumulate df_intersecting_pairs_all over all event_id_int_values
