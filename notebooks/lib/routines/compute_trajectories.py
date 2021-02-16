# compute_trajectories.py

import trackpy, pandas as pd, numpy as np
from .. import *
from ..utils.dist_func import *

def compute_trajectories(input_file_name, mem, sr, width, height, **kwargs):
    df = pd.read_csv(input_file_name)
    df.drop_duplicates(subset=['t','x','y'],keep='first',inplace=True,ignore_index=True)
    # df.drop_duplicates(inplace=True, ignore_index=True)
    print(f"tracking spiral tips for {os.path.basename(input_file_name)}...")
    t_list =  sorted(set(df.t.values))
    frameno_list = list(range(len(t_list)))

    df['frame'] = -9999
    for frameno, t in zip(frameno_list,t_list):
        df.loc[df.t==t, 'frame'] = frameno
    #assert that all entries were given a value
    assert ( not (df.frame<0).any() )

    #consider all tip pairs
    # width, height = 200, 200 # txt.shape[:2]
    distance_L2_pbc = get_distance_L2_pbc(width,height)

    link_kwargs = {
        'neighbor_strategy' : 'BTree',
        'adaptive_step':0.5,
        'adaptive_stop':1e-5,
        'dist_func'    :distance_L2_pbc,
        'memory'       :mem,
        'search_range' :sr
    }

    # df['frame'] = np.around(df['t']/h)
    # df = df.astype(dtype={'frame':int}).copy()
    traj = trackpy.link_df(
        f=df.head(-1),t_column='frame',**link_kwargs)
    return traj

def routine_compute_trajectories(input_file_name, mem, sr, width, height, save_folder=None, input_folder=None, **kwargs):
    if input_folder is not None:
        os.chdir(input_folder)
    traj = compute_trajectories(input_file_name, mem = mem, sr  = sr, width = width, height = height)
    if save_folder is None:
        dirname = os.path.dirname(input_file_name).split('/')[-1]
        folder_name=os.path.dirname(input_file_name)
        save_folder = folder_name.replace(dirname,'trajectories')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    os.chdir(save_folder)
    save_fn = input_file_name.replace('log.csv', f'traj_sr_{sr}_mem_{mem}.csv')
    #save results
    traj.to_csv(save_fn, index=False)
    return os.path.abspath(save_fn)
