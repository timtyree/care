#Methods of mappping annihilation .csv folders to dRdt versus R data
import numpy as np, pandas as pd, os, sys, re
from . import *
from scipy.signal import savgol_filter
################################
# Smoothed method
################################
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_average_of_events(df,
                             valid_event_id_lst=None,
                             navg=20,
                             id_col='event_id',
                             t_col='tdeath',
                             **kwargs):
    df.sort_values([id_col, t_col], ascending=False, inplace=True)
    if valid_event_id_lst is None:
        event_id_lst = sorted(set(df[id_col].values))
        valid_event_id_lst=event_id_lst
    #moving average of df for each event_id
    for event_id in valid_event_id_lst:
        bo = df[id_col] == event_id
        df.loc[bo] = df[bo].rolling(navg).mean()
    df.dropna(inplace=True)
    return df

def smooth_derivative_filter_of_events(df,
                                       valid_event_id_lst=None,
                                       DT_sec=0,
                                       x_col='r',
                                       dxdt_col='drdt',
                                       id_col='event_id',
                                       t_col='tdeath',
                                       ascending=False,
                                       navg=21,
                                       polyorder=3,
                                       mode='interp',
                                       **kwargs):
    '''
    navg is an odd integer and is the number of frames to average over.
    polyorder=3 corresponds to cubic.
    mode='interp' is reasonable for nonperiodic 1D arrays
    mode can be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. wrap is for periodic boundary conditions.
    DT=0.000025 is in seconds per frame for the highest resolution spiral tip data set I made.
    '''
    df.sort_values([id_col, t_col], ascending=False, inplace=True)
    if valid_event_id_lst is None:
        event_id_lst = sorted(set(df[id_col].values))
        valid_event_id_lst=event_id_lst
    for event_id in valid_event_id_lst:
        bo = df[id_col] == event_id
        R_values = df[x_col][bo].values
        if R_values.shape[0]>=navg:
            dRdt_values = savgol_filter(x=R_values,
                                        window_length=navg,
                                        polyorder=polyorder,
                                        deriv=1,
                                        delta=1.0,
                                        axis=-1,
                                        mode='interp')
        else:
            dRdt_values=np.nan+0.*R_values
        df.loc[bo, dxdt_col] = dRdt_values/DT_sec
    return df

#DONE: make a get_annihilation_df function that takes navg1,navg2 and kwargs and returns the resulting df
def get_annihilation_df(input_fn,navg1,navg2,
                        rdeath_thresh = 0.7,
                        t_col = 'tdeath',
                        id_col = 'event_id',
                        pid_col = 'pid',
                        x_col = 'r',
                        dxdt_col = 'drdt',
                        DT = 0,
                        DT_sec=0,
                        size_thresh=200,
                        printing = False,**kwargs):
    '''first, get_annihilation_df removes any event_id's that end (t_col takes minimum absolute value) at a range of more than 0.5 cm (r > rdeath_thresh)
    then, get_annihilation_dfcomputes the moving_average_of_events
    Example Usage:
    navg1=41
    navg2=11
    df,valid_event_id_lst=get_annihilation_df(input_fn,navg1,navg2)
    data=df,valid_event_id_lst,navg1,navg2
    '''
    #load data and filter unreasonable events
    df = pd.read_csv(input_fn)
    df.sort_values([id_col, t_col], ascending=False, inplace=True)
    tvals = sorted(set(df[t_col].values))
    DT = tvals[1] - tvals[0] #ms
    DT_sec=DT*0.001 #seconds
    event_id_lst = sorted(set(df[id_col].values))
    valid_event_id_lst=filter_events_ending_at_large_range(df,rdeath_thresh=rdeath_thresh)
    invalid_event_id_lst=find_events_insufficient_size(df,size_thresh=size_thresh)
    valid_event_id_lst=list(set(valid_event_id_lst).difference(set(invalid_event_id_lst)))
    if printing:
        print(f"{len(event_id_lst)} annihilation events are inputed")
        print(f"{len(valid_event_id_lst)} annihilation events end at a range smaller than {rdeath_thresh} cm.")

    #smooth and differentiate data
    df = moving_average_of_events(df, valid_event_id_lst, navg=navg1, id_col=id_col, t_col=t_col)
    valid_event_id_lst=filter_events_ending_at_large_range(df,rdeath_thresh=rdeath_thresh)
    invalid_event_id_lst=find_events_insufficient_size(df,size_thresh=size_thresh)
    valid_event_id_lst=list(set(valid_event_id_lst).difference(set(invalid_event_id_lst)))
    df = smooth_derivative_filter_of_events(df,DT_sec=DT_sec,valid_event_id_lst=valid_event_id_lst,
                                            x_col=x_col,
                                            dxdt_col=dxdt_col,
                                            id_col=id_col,
                                            t_col=t_col,
                                            ascending=False,
                                            navg=navg2,
                                            polyorder=3,
                                            mode='interp')
    df.dropna(inplace=True)#filters any events that had DT be too small
    valid_event_id_lst = filter_events_ending_at_large_range(df, rdeath_thresh=rdeath_thresh)
    return df,valid_event_id_lst


################################
# Old naive method without smoothing
################################
#remove any event_id's that end (t_col takes minimum absolute value) at a range of more than 0.5 cm (r > rdeath_thresh)
def get_annihilation_df_naive(input_fn,
                            rdeath_thresh=0.7,
                            min_dRdt=-1000,
                            max_dRdt=1000,
                            x_col='r',
                            dxdt_col='drdt',
                            t_col='tdeath',
                            id_col='event_id',
                            pid_col='pid',
                            DT=0,
                            DT_sec=0,
                            size_thresh=200,
                            printing=False,**kwargs
                             ):
    '''
    Example Usage:
    df,valid_event_id_lst=get_annihilation_df_naive(input_fn)
    '''
    df=pd.read_csv(input_fn)
    df.sort_values([id_col, t_col], ascending=False, inplace=True)
    event_id_lst=sorted(set(df[id_col].values))
    if printing:
        print(f"{len(event_id_lst)} annihilation events are inputed")
    valid_event_id_lst=filter_events_ending_at_large_range(df,rdeath_thresh=rdeath_thresh)
    invalid_event_id_lst=find_events_insufficient_size(df,size_thresh=size_thresh)
    valid_event_id_lst=list(set(valid_event_id_lst).difference(set(invalid_event_id_lst)))
    if printing:
        print(f"{len(valid_event_id_lst)} annihilation events end at a range smaller than {rdeath_thresh} cm.")

    #TODO: remove all plotting from the following block
    #plot annihilations without filtering directly from csv
    df.sort_values([id_col, t_col], ascending=False, inplace=True)
    df[dxdt_col]=df[x_col].diff()/DT_sec #cm/s
    boo=np.abs(df[x_col].diff())>DT
#     for j,event_id in enumerate(valid_event_id_lst):
#         d=df[df[id_col]==event_id]
#         df.loc[df[id_col]==event_id,dxdt_col]=d[x_col].diff()/DT_sec #cm/s

    #naive filtering of extreme data
    boo|=(df[dxdt_col]>=max_dRdt)|(df[dxdt_col]<=min_dRdt)
    df.loc[boo,dxdt_col]=np.nan
    df.dropna(inplace=True)
    return df,valid_event_id_lst
