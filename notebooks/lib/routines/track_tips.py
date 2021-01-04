# Tim Tyree
# 12.18.2020

import trackpy, sklearn, pandas as pd, numpy as np
from .. import *
from ..utils.dist_func import *
def compute_track_tips_pbc(df, mem=2, sr=400,
                       width=200, height=200, adaptive_step=0.5,
                       adaptive_stop=1e-5, **kwargs):
    '''returns a dataframe of trajectories resulting from the positions
    listed in the .csv, input_file_name using period boundary conditions (pbc).
    sr is the search range, which needs to be bigger than sqrt(max(width,height))
    to work with periodic boundary conditions.'''
    # distance_L2_pbc = get_distance_L2_pbc(width,height)
    # df = pd.read_csv(input_file_name)
    #assign each time a unique frame number
    t_list =  sorted(set(df.t.values))
    frameno_list = list(range(len(t_list)))
    df['frame'] = -9999
    for frameno, t in zip(frameno_list,t_list):
        df.loc[df.t==t, 'frame'] = frameno
    #assert that all entries were given a value
    assert ( not (df.frame<0).any() )
    distance_L2_pbc = get_distance_L2_pbc(width,height)

    link_kwargs = {
        'neighbor_strategy' : 'BTree',
        'adaptive_step':adaptive_step,
        'adaptive_stop': adaptive_stop,
        'dist_func'         : distance_L2_pbc,
        'memory': mem,
        'search_range':sr
    }

    traj = trackpy.link_df(
        f=df.head(-1),t_column='frame',**link_kwargs)
    return traj
def generate_track_tips_pbc(input_file_name, save_fn=None, mem=0, sr=400,
                       width=200, height=200, adaptive_step=0.5,
                       adaptive_stop=1e-5, **kwargs):
    '''performs compute_track_tips_pbc and then saves to csv using
    save_fn, which is generated automatically if not given.'''
    if save_fn is None:
        save_fn = os.path.abspath(input_file_name).replace('/Log','/trajectories').replace('log.csv', f'traj_sr_{sr}_mem_{mem}.csv')
    df = pd.read_csv(input_file_name)
    df.drop_duplicates(subset=['x','y'],keep='first',inplace=True,ignore_index=True)
    # df.dropduplicate(inplace=True)1
    traj = compute_track_tips_pbc(df, **kwargs)
    traj.to_csv(save_fn, index=False)
    return save_fn

