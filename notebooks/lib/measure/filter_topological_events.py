# filter_topological_events.py
import pandas as pd,numpy as np
#DONE: make function that identifies any event_id's that are smaller than a threshold size
#DONE: use ^that to remove (set difference) that one event preventing a reasonably large navg2 value
def find_events_insufficient_size(df,size_thresh=200,t_col='tdeath',id_col='event_id'):
    '''
    Example Usage:
    invalid_event_id_lst=find_events_insufficient_size(df,size_thresh=100)
    '''
    boo=df.groupby(id_col)[t_col].count()<size_thresh
    invalid_event_id_lst=list(boo[boo].index.values)
    return invalid_event_id_lst

def filter_events_ending_at_large_range(df,rdeath_thresh=0.7,t_col='tdeath',
    id_col='event_id',x_col = 'r',dxdt_col = 'drdt'):
    '''
    Example Usage:
    valid_event_id_lst=filter_events_ending_at_large_range(df,rdeath_thresh=0.1)
    '''
    event_id_lst=sorted(set(df[id_col].values))
    valid_event_id_lst=[]
    for event_id in event_id_lst:
        d=df[df[id_col]==event_id]
        mint=d[t_col].min()
        rdeath=d[d[t_col]==mint][x_col].min()
        if rdeath<rdeath_thresh:
            valid_event_id_lst.append(event_id)
    return valid_event_id_lst
