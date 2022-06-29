import dask.bag as db
import numpy as np, pandas as pd, sys, os, time
from ..rapids_func.measure.annihilations_cu import savgol_filter_try
from ..measure.compute_slope import compute_95CI_ols

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float,axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#TODO: port to lib:
#this block tares ballistically annihilating particles
#TODO: accelerate tshift_tare_routine
#NOTA BENE: the same memory, df_Ri, is being accessed by every worker simultaneously...
#TODO: compare runtimes with cudf.DataFrame(df_Ri)
def tshift_tare_routine(df_R,navg2,max_num_groups=9e9,npartitions=None,R_col='R_nosavgol',printing=True,**kwargs):
    '''
    Nota Bene: the same memory, df_Ri, is being accessed by every worker simultaneously...

    Example Usage:
    retval=tshift_tare_routine(df_R,navg2,max_num_groups=9e9,plotting=False,npartitions=None,R_col='R_nosavgol')
    tshift_align_lst,speed_align_lst,group_name_lst,SR_speed_align_lst,Delta_SR_speed_align_lst=retval
    '''
    navg2=navg2 if navg2%2==1 else navg2+1
    #TODO: accelerate tshift_tare_routine
    #compute time shifts for the first x groups
    #include memory preallocation
    df_R['tdeath_align']=df_R['tdeath'].copy()
    #sort by time
    df_R.sort_values(by='tdeath',ascending=True,inplace=True)
    #group dataframe by annihilation event
    # groups=df_R.sort_values('tdeath',ascending=True).groupby(['event_id_int','pid_self','pid_other'])
    groups=df_R.groupby(['event_id_int','pid_self','pid_other'])
    df_Ri=df_R.set_index(['event_id_int','pid_self','pid_other'])
    # groups=df_fk.sort_values('tdeath',ascending=True).groupby(['event_id_int','pid_self','pid_other'])
    tshift_align_lst=[]
    speed_align_lst=[]
    group_name_lst=[]
    SR_speed_align_lst=[]
    Delta_SR_speed_align_lst=[]
    count=0
    task_lst=[]
    for group_name,g in groups:
        count+=1
        if max_num_groups>=count:
            data=group_name,g
            #TODO: generate task_lst
            task_lst.append(data)

    #DONE: accelerate tshift_tare_routine here
    def eval_routine(data):
        mode='diff'#'ols'
        group_name,g=data
        #to compute tshift_align,
        savgol_kwargs=dict(
                window_length=navg2,
                polyorder=3,
                deriv=1,
                delta=1.0,
                axis=-1,
                mode='interp')
        savgol0_kwargs=dict(
                window_length=navg2,
                polyorder=3,
                deriv=0,
                delta=1.0,
                axis=-1,
                mode='interp')

        #extract the R_values and tdeath_values
        tdeath_values=g['tdeath'].values
        SR_values=savgol_filter_try(g['R_nosavgol'].values,**savgol0_kwargs)**2

        if not (SR_values<0).any():


            #perform ols least squares on the final 8 values
            if mode=='ols':
                y=SR_values[:8]
                x=tdeath_values[:8]
            else:# mode=='diff':
                y=SR_values[:3]
                x=tdeath_values[:3]

            try:
                dict_ols=compute_95CI_ols(x,y)
                m=dict_ols['m']
                SR_speed_align=m
                Delta_SR_speed_align=dict_ols['Delta_m']
                #compute the time shift
                tf=x[0]
                tshift_align=float(tf+dict_ols['b']/m)
                return tshift_align,group_name,SR_speed_align,Delta_SR_speed_align
            except AssertionError as e:
                return 0.,group_name,-9999.,-9999.
        else:
            return 0.,group_name,-9999.,-9999.
    # #         else:
    # #             SR_speed_align=-9999
    # #             Delta_SR_speed_align=-9999
    # #             tshift_align=-9999
    # #             #raise(r'Exception: mode={mode} not implemented!')
    #         #compute tdeath_align from tdeath and the moving average of speed
    #         #tshift trajectories by that value
    #         return tshift_align,group_name,SR_speed_align,Delta_SR_speed_align
    #     else:
    #         return 0.,group_name,-9999.,-9999.
    def routine(data):
        # return eval_routine(data)
        try:
            return eval_routine(data)
        except Exception as e:
            return f"Warning: {e}"

    #TODO: dask bag accelerate tshift_tare_routine
    if npartitions is None:
        npartitions=os.cpu_count()
    #all CPU version
    b = db.from_sequence(task_lst, npartitions=npartitions).map(routine)
    #TODO: evaluate task_lst
    start = time.time()
    retval = list(b)
    if printing:
        print(f"run time aligning trajectories was {time.time()-start:.2f} seconds.")
    return retval
#     tshift_align_lst,group_name_lst,SR_speed_align_lst,Delta_SR_speed_align_lst=retval
#     #TODO(maybe needed?): transpose output retval to lists
#     return tshift_align_lst,group_name_lst,SR_speed_align_lst,Delta_SR_speed_align_lst
