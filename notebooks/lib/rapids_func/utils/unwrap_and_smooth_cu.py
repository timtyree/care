#utils that unwrap and smooth trajectories on the gpu
#Programmer: Tim Tyree
#Date: 9.29.2021
import numpy as np, cupy as cp, numba.cuda as cuda, cudf#, pandas as pd

def unwrapper_pbc(incol, outcol, jump_thresh, width):
    '''
    Example Usage: for cudf.DataFrame instance
    df['incol']=df['x']
    grouped = df.groupby(pid_col)
    df = grouped.apply_grouped(unwrapper_pbc,
                                   incols=['incol'],
                                   outcols=dict(outcol=np.float64), jump_thresh=width/2)
    df['dx_unwrap']=df['outcol']
    df.head()
    '''
    e=incol
    de_unwrap=outcol
    for i in range(cuda.threadIdx.x, len(e), cuda.blockDim.x):
        de_unwrap[i]=0
        if i>0:
            de=e[i]-e[i-1]
            jump_plus=de<-jump_thresh
            jump_minus=de>jump_thresh
            if jump_plus:
                de_unwrap[i]=width
            elif jump_minus:
                de_unwrap[i]=-width

def rolling_avg(incol, outcol, win_size):
    e=incol
    rolling_avg_e=outcol
    for i in range(cuda.threadIdx.x, len(e), cuda.blockDim.x):
        if i < win_size - 1:
            # if there is not enough data to fill the window, take the average to be nan
            rolling_avg_e[i] = np.nan
        else:
            total = 0
            for j in range(i - win_size + 1, i + 1):
                total += e[j]
            rolling_avg_e[i] = total / win_size

def rolling_diff(incol, outcol, win_size=2):
    e=incol
    rolling_diff_e=outcol
    for i in range(cuda.threadIdx.x, len(e), cuda.blockDim.x):
        if i < win_size - 1:
            # if there is not enough data to fill the window, take the average to be nan
            rolling_diff_e[i] = np.nan
        else:
            j=i - win_size + 1
            rolling_diff_e[i] = e[i]-e[j]

def apply_unwrap_xy_trajectories_pbc(df,t_col,pid_col,width,height,**kwargs):
    #now we only have good data...  we can compute moving averages for each particle!
    #allocate memory
    df['dx_unwrap']=0.*df['x']
    df['dy_unwrap']=0.*df['y']
    df['x_unwrap']=df['x']
    df['y_unwrap']=df['y']
    #TODO(optional): reset the index... not needed and ruins reconstruction of dropped columns at the end...  don't do it...
    # df.reset_index(inplace=True)
    #apply unwrapping to x and y
    df['incol']=df['x']
    grouped = df.groupby(pid_col)
    uwargs={'jump_thresh':width/2,"width":width}
    df = grouped.apply_grouped(unwrapper_pbc,
                                   incols=['incol'],
                                   outcols=dict(outcol=np.float64), kwargs=
                               uwargs)
    df['dx_unwrap']=df['outcol']

    df['incol']=df['y']
    grouped = df.groupby(pid_col)
    uwargs={'jump_thresh':height/2,"width":height}
    df = grouped.apply_grouped(unwrapper_pbc,
                                   incols=['incol'],
                                   outcols=dict(outcol=np.float64), kwargs=
                               uwargs)
    df['dy_unwrap']=df['outcol']

    df.drop(columns=['incol','outcol'],inplace=True)

    #DONE: confirmed ^that was nontrivial and reasonable looking
    # (df['dx_unwrap']!=0).any(),(df['dy_unwrap']!=0).any()
    # df[df['dx_unwrap']!=0].head()
    # (True,True)

    #aggregte over jumps
    grouped_unwrap=df.groupby(pid_col)

    #aggregate along a given columns in grouped_unwrap
    result=grouped_unwrap[['dx_unwrap','dy_unwrap']].cumsum()

    #map result back onto df using reindexing ninjitsu
    cp_col_lst=['dx_unwrap','dy_unwrap']
    df.reset_index(inplace=True)
    result.reset_index(inplace=True)
    for col in cp_col_lst:
        df[col]=result[col]
    df.set_index('index',inplace=True)

    #compute unwrapped coordinates
    df['x_unwrap']=df['x']+df['dx_unwrap']
    df['y_unwrap']=df['y']+df['dy_unwrap']
    return df

def apply_moving_avg_xy_trajectories(df,t_col,pid_col,navg1,x_col='x_unwrap',y_col='y_unwrap',**kwargs):
    diffx_col='diff'+x_col
    diffy_col='diff'+y_col
    #apply smoothing to x and y after unwrapping
    df['incol']=df[x_col]
    grouped = df.groupby(pid_col)
    if navg1>0:
        mawargs={'win_size':navg1}
        df = grouped.apply_grouped(rolling_avg,
                                       incols=['incol'],
                                       outcols=dict(outcol=np.float64), kwargs=
                                   mawargs)
        df[x_col]=df['outcol']

        df['incol']=df[y_col]
        grouped = df.groupby(pid_col)
        df = grouped.apply_grouped(rolling_avg,
                                       incols=['incol'],
                                       outcols=dict(outcol=np.float64), kwargs=
                                   mawargs)
        df[y_col]=df['outcol']
    # else:
    #     #perform no moving average if the window is of size zero
    #     pass
    # #drop data that isn't needed anymore
    #DONE: verified that dropping data here doesn't affect the number of final nonnan values
    # df.drop(columns=['incol','outcol'],inplace=True)
    df.dropna(inplace=True)
    # df.head()

    #apply smoothing to x and y after unwrapping
    df['incol']=df[x_col]
    grouped = df.groupby(pid_col)
    mdwargs={'win_size':2}
    df = grouped.apply_grouped(rolling_diff,
                                   incols=['incol'],
                                   outcols=dict(outcol=np.float64), kwargs=
                               mdwargs)
    df[diffx_col]=df['outcol']

    df['incol']=df[y_col]
    grouped = df.groupby(pid_col)
    df = grouped.apply_grouped(rolling_diff,
                                   incols=['incol'],
                                   outcols=dict(outcol=np.float64), kwargs=
                               mdwargs)
    df[diffy_col]=df['outcol']

    #drop data that isn't needed anymore
    df.drop(columns=['incol','outcol'],inplace=True)
    # df.dropna(inplace=True)

    #compute the naive speed of the unwrapped trajectories in pixels per frame
    df['speed']=cp.sqrt(df[diffx_col]**2+df[diffy_col]**2)#pixels per frame
    # df['speed']=cp.sqrt(df['diffx_unwrap']**2+df['diffy_unwrap']**2)#pixels per frame#*DS/DT*10**3 #cm/s

    # #DONE: test and verify that the largest stepsize in the unwrapped xy is reasonable for both x and y
    # max_speed_values=df.groupby(pid_col)['speed'].max().values
    # plt.hist(max_speed_values.get(),bins=30)
    # plt.xlabel('max pixel displacement between two frames')
    # max_speed_warning=20 #pixels per frame
    # assert ((max_speed_values.get()<max_speed_warning).all())
    # plt.show()
    return df
