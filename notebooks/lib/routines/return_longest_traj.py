# from lib import *
# from lib.routines.generate_tip_logs_LR_model_cy import *
# from lib.routines.kwargs_LR_model_cy import get_kwargs
# from lib.utils.get_txt import get_txt
# from lib.routines.dag_log_to_msd import *
# from lib.routines.compute_msd import get_longest_trajectories
# import random
# import os,sys
#
import trackpy, pandas as pd, numpy as np
from .. import *
from .track_tips import *
from ..utils.dist_func import *
from ..utils.utils_traj import *
from .compute_msd import * #unwrap_traj_and_center

def return_unwrapped_trajectory(df, width, height, sr, mem, dsdpixel, DT, round_t_to_n_digits,jump_thresh, **kwargs):
    '''df is a pandas.DataFrame containing the tip log results.'''
    DS=dsdpixel
    # generate_track_tips_pbc
    df.drop_duplicates(subset=['t','x','y'],keep='first',inplace=True)
    #,ignore_index=True)
    df = compute_track_tips_pbc(df, mem, sr, width, height)#,**kwargs)
    # unwrap_trajectories
    pid_lst = sorted(set(df.particle.values))
    df = pd.concat([unwrap_traj_and_center(df[df.particle==pid].copy(), width, height, DS) for pid in pid_lst])
    pid_longest_lst=pid_lst
    #truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    df_lst = []
    for pid in  pid_longest_lst:#[2:]:
        d = df[(df.particle==pid)].copy()
        x_values, y_values = d[['x','y']].values.T
        index_values = d.index.values.T
        jump_index_array, spd_lst = find_jumps(x_values,y_values,width,height, DS=DS,DT=DT, jump_thresh=jump_thresh, **kwargs)#.25)
        if len(jump_index_array)>0:
            ji = jump_index_array[0]
            d.drop(index=index_values[ji:], inplace=True)
        df_lst.append(d)
    df_traj = pd.concat(df_lst)

    df_traj['t'] = df_traj.t.round(round_t_to_n_digits)

    return df


def return_longest_trajectories(df, width, height, dsdpixel, n_tips = 1, DT = 2.,
                                round_t_to_n_digits=0, jump_thresh=20., **kwargs):
    '''df is a pandas.DataFrame of a tip log'''
    mem=0;sr=width*2;DS=dsdpixel
    df=return_unwrapped_trajectory(df, width, height, sr, mem, dsdpixel, DT, round_t_to_n_digits,jump_thresh, **kwargs)
    if n_tips==1:
        df.reset_index(inplace=True)
        try:
            s = df.groupby('particle').t.count()
        except KeyError as e:#("KeyError") as e:
            # print(e)
            print( f"\t trial that failed: {input_file_name.split('/')[-1]}")
            return None
        s = s.sort_values(ascending=True)
        pid_longest_lst = list(s.index.values)#[:n_tips])
        #filter trajectories that do not move explicitely
        pid=pid_longest_lst.pop()
        std_diffx=df[(df.particle==pid)].x.diff().dropna().std()
        boo=False
        if std_diffx:#.diff().dropna().x.std
            if std_diffx>0:
                boo=True
        while not boo:
            pid=pid_longest_lst.pop()
            std_diffx=df[(df.particle==pid)].x.diff().dropna().std()
            boo=False
            if std_diffx:#.diff().dropna().x.std
                if std_diffx>0:
                    boo=True
        pid_longest_lst=[pid]
    else:
        try:
            s = df.groupby('particle').t.count()
        except KeyError as e:#("KeyError") as e:
            # print(e)
            print( f"\t trial that failed: {input_file_name.split('/')[-1]}")
            return None
        s = s.sort_values(ascending=False)
        pid_longest_lst = list(s.index.values[:n_tips])
    df_traj = pd.concat([df[df.particle==pid] for pid in pid_longest_lst])

    #truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    df_lst = []
    for pid in  pid_longest_lst:#[2:]:
        d = df[(df.particle==pid)].copy()
        x_values, y_values = d[['x','y']].values.T
        index_values = d.index.values.T
        jump_index_array, spd_lst = find_jumps(x_values,y_values,width,height, DS=DS,DT=DT, jump_thresh=jump_thresh, **kwargs)#.25)
        if len(jump_index_array)>0:
            ji = jump_index_array[0]
            d.drop(index=index_values[ji:], inplace=True)
        df_lst.append(d)
    df_traj = pd.concat(df_lst)
    # #truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    # df_lst = []
    # for pid in  pid_longest_lst:#[2:]:
    #     d = df[(df.particle==pid)].copy()
    #     # #truncate all info after the first jump
    #     # x_values, y_values = d[['x','y']].values.T
    #     # index_values = d.index.values.T
    #     # jump_index_array, spd_lst = find_jumps(x_values,y_values,width,height, DS=DS,DT=DT, jump_thresh=jump_thresh, **kwargs)#.25)
    #     # if len(jump_index_array)>0:
    #     #     ji = jump_index_array[0]
    #     #     d.drop(index=index_values[ji:], inplace=True)
        # df_lst.append(d)
    # df_traj = pd.concat(df_lst)

    #round trajectory times to remove machine noise from floating point arithmatic
    df_traj['t'] = df_traj.t.round(round_t_to_n_digits)
    # df_traj['x'] = df_traj.x.round(6)
    # df_traj['y'] = df_traj.y.round(6)
    df_traj['grad_ux'] = df_traj.grad_ux.round(5)
    df_traj['grad_uy'] = df_traj.grad_uy.round(5)
    df_traj['grad_vx'] = df_traj.grad_vx.round(5)
    df_traj['grad_vy'] = df_traj.grad_vy.round(5)
    #     assert ( (np.array(sorted(set(df_traj['particle'].values)))==np.array(sorted(pid_longest_lst))).all())
    return df_traj
