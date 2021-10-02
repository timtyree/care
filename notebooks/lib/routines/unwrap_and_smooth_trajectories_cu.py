#unwraps and smooths trajectories on the gpu
#Programmer: Tim Tyree
#Date: 9.29.2021
import numpy as np, cupy as cp, numba.cuda as cuda, cudf, os, re, dask.bag as db, time
from ..measure.unwrap_and_smooth_cu import *
from ..utils.operari import get_all_files_matching_pattern

#DONE(dev smoothing with pbc): dev method that unwraps, keeps track of the unwrapping map, smooths the unwrapped by moving average, and then rewraps
#DONE: unwrap each trajectory
#DONE: compute simple moving average of the x and y coordinates
#DONE: rewrap to smoothed coordinates to the original coordinate system
#DONE: clean one example input_fn
#DONE: smooth one example input_fn with pbc

def return_moving_average_of_pbc_trajectories(input_fn, tavg1, pid_col, t_col, DT,
                                              width=200,
                                              height=200,
                                              use_drop_shorter_than=True,
                                              drop_shorter_than=50,
                                              tmin=100.,
                                              printing=False,
                                              **kwargs):
    '''returns cudf dataframe that results from the routine.
    routine reads input_fn, reversably unwraps trajectories, computes the moving average of x and y, rewraps smoothed coordinates to original coordinates (might not still obey explicit pbc of shape (0,width,0,height))
    tavg1 is the moving average window, and drop_shorter_than is the minimum duration of trajectories to consider in the same units as the time column, t_col.  dt is the time between two frames.  '''
    navg1=int(tavg1/DT)
    height=width
    df=cudf.read_csv(input_fn)
    col_keep_lst=['x','y',pid_col,t_col]
    #(optional) drop all columns that are not immediately relevant
    col_drop_lst=set(df.columns).difference(col_keep_lst)
    df.drop(columns=col_drop_lst,inplace=True)
    if printing:
        print(f"features dropped: {col_drop_lst}")
        print(f"note: ^these can be recovered using the index")
    #sort by particle and then by time
    df=df.sort_values([pid_col, t_col], ascending=True).copy()

    #drop trials that are too brief... also drop rows that occur before tmin
    if use_drop_shorter_than:
        grouped = df.groupby(pid_col)
        dft=grouped[t_col]
        dfkeep=(dft.max()-dft.min() >= drop_shorter_than).copy()
        if printing:
            print(f"the percent of observations dropped for being briefer than {drop_shorter_than} ms is {(1-dfkeep[dfkeep].size/dfkeep.size)*100:.2f}%")
        pid_keep_values=dfkeep[dfkeep].index.values
        pid_keep_lst=list(pid_keep_values.get())
        #batch the query, because f-string parsing has a limit at ~1000 list items and for efficient scalability
        batchsize=500
        num_batches=int(np.ceil(len(pid_keep_lst)/batchsize))
        dfq_lst=[]
        for j in range(num_batches):
            dfq=df.query(f"{pid_col} in {pid_keep_lst[batchsize*j:batchsize*(j+1)]} and t >= {tmin}").copy()
            dfq_lst.append(dfq)
        df=cudf.concat(dfq_lst)
        del dfq_lst, pid_keep_values, dfkeep, grouped

    #the majority of the routine
    apply_unwrap_xy_trajectories_pbc(df,t_col,pid_col,width,height)
    apply_moving_avg_xy_trajectories(df,t_col,pid_col,navg1,x_col='x_unwrap',y_col='y_unwrap')

    #compute rewrapped coordinates
    df['x']=df['x_unwrap']-df['dx_unwrap']
    df['y']=df['y_unwrap']-df['dy_unwrap']
    #CONFIRMED: by increasing navg1, I can decrease the max displacement for all particles.

    #add unique identifier for whole trial that is unique accross different csv files
    #add unique identifier for each particle that is unique accross different csv files
    fn = os.path.basename(input_fn)
    event_id_int=float('1'+(''.join(re.findall(r'-?\d+\d*',fn))))
    df['event_id_int']= int(event_id_int)
    df['event_id']= event_id_int + df[pid_col] / (1.+df[pid_col].max())

    col_keep_lst=['x','y',t_col,pid_col,'event_id_int',"dx_unwrap","dy_unwrap"]
    dff=df[col_keep_lst].copy()

    #DONE: drop anycolumns that are recomputed with a one liner, such as x_unwrap and diffx_unwrap, and speed
    #DONE: collect recomputation of ^those one liners into a compact update_smoothed_trajectories function
    # # #DONE: verified that update_smoothed_trajectories reproduces the full df to machine precision.
    # # df_recalled=update_smoothed_trajectories(dff)
    # # df_recalled.head()
    # #DONE: test that all columns are contained in the dataframe reconstructed from the output dataframe
    # assert((np.sort(df.columns.values)==np.sort(df_recalled.columns.values)).all())
    return dff

def return_moving_average_of_pbc_trajectories_and_save(
        input_fn, tavg1, pid_col, t_col, DT, width, height,
        use_drop_shorter_than, drop_shorter_than, tmin, printing):
    dff = return_moving_average_of_pbc_trajectories(
        input_fn,
        tavg1,
        pid_col,
        t_col,
        DT,
        width=width,
        height=height,
        use_drop_shorter_than=use_drop_shorter_than,
        drop_shorter_than=drop_shorter_than,
        tmin=tmin,
        printing=printing)

    #save as csv in new folder
    navg1=int(tavg1/DT)
    save_folder_ext = f'smoothed_trajectories_tavg_{tavg1}'
    save_folder = os.path.join(os.path.dirname(os.path.dirname(input_fn)),
                               save_folder_ext)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    ext_str = f'_smoothed'
    save_fn = os.path.basename(input_fn).replace('.csv', ext_str + '.csv')

    save_dir = os.path.join(save_folder, save_fn)

    dff.reset_index(inplace=True)
    dff.to_csv(save_dir,index=False)
    if printing:
        if os.path.exists(save_dir):
            print(f"the results were successfully saved to {save_folder}")
        else:
            print(f"the results were unsuccessfully saved to {save_folder}")
    return save_dir

def update_smoothed_trajectories(df,pid_col,t_col):
    event_id_int=df['event_id_int']
    df['event_id']= event_id_int + df[pid_col] / (1.+df[pid_col].max())
    #compute unwrapped coordinates
    df['x_unwrap']=df['x']+df['dx_unwrap']
    df['y_unwrap']=df['y']+df['dy_unwrap']
    #apply rolling difference to x and y after unwrapping
    df['incol']=df['x_unwrap']
    grouped = df.groupby(pid_col)
    mdwargs={'win_size':2}
    df = grouped.apply_grouped(rolling_diff,
                                   incols=['incol'],
                                   outcols=dict(outcol=np.float64), kwargs=
                               mdwargs)
    df['diffx_unwrap']=df['outcol']

    df['incol']=df['y_unwrap']
    grouped = df.groupby(pid_col)
    df = grouped.apply_grouped(rolling_diff,
                                   incols=['incol'],
                                   outcols=dict(outcol=np.float64), kwargs=
                               mdwargs)
    df['diffy_unwrap']=df['outcol']
    #drop data that isn't needed anymore
    df.drop(columns=['incol','outcol'],inplace=True)
    #compute the naive speed of the unwrapped trajectories in pixels per frame
    df['speed']=cp.sqrt(df['diffx_unwrap']**2+df['diffy_unwrap']**2)#pixels per frame
    return df

def load_smoothed_trajectories(input_fn,pid_col,t_col):
    df=cudf.read_csv(input_fn)
    update_smoothed_trajectories(df,pid_col,t_col)
    return df

def routine_postprocess_trajectory_folder(input_fn,DT,tavg1=2, npartitions=None,
                                        width=200,
                                        height=200,
                                        use_drop_shorter_than=True,
                                        drop_shorter_than=50, #ms
                                        tmin=100., #ms
                                        pid_col='particle',
                                        t_col='t',
                                        printing=False,**kwargs):
    if npartitions is None:
        npartitions=os.cpu_count()
    input_fn_lst=get_all_files_matching_pattern(input_fn,trgt='.csv')
    print(f"running return_moving_average_of_pbc_trajectories_and_save on {len(input_fn_lst)} files...")

    def routine(input_fn):
        try:
            return return_moving_average_of_pbc_trajectories_and_save(
                input_fn, tavg1, pid_col, t_col, DT, width, height,
                use_drop_shorter_than, drop_shorter_than, tmin, printing)
        except Exception as e:
            return None

    bag = db.from_sequence(input_fn_lst, npartitions=npartitions).map(routine)
    start = time.time()
    retval_lst = list(bag)
    print(f"the run time was {(time.time()-start)/60:.2f} minutes.")
    print(f"the number of successful files was {len(retval_lst)}")
    return retval_lst

#for loop implementation has an expected run time of ~4 minutes for LR data with DT=0.5
#for loop implementation has an expected run time of ~4 minutes for FK data with DT=0.4
