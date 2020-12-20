# Tim Tyree
# 12.18.2020

import trackpy, pandas as pd, numpy as np
from .. import *
from .track_tips import *
from ..utils.dist_func import *

def filter_duplicate_trajectory_indices(pid_longest_lst,df_traj):
    '''slow run time. don't use.  duplicates removed earlier in the pipeline much more quickly.'''
    pid_longest_lst_filtered = sorted(pid_longest_lst)
    M = len(pid_longest_lst)
    for n,pid1 in enumerate(pid_longest_lst_filtered):
        x1 = df_traj[df_traj.particle==pid1].x.tail(1).values
        y1 = df_traj[df_traj.particle==pid1].y.tail(1).values
        if n<M+1:
            for pid2 in pid_longest_lst_filtered[n+1:]:
                x2 = df_traj[df_traj.particle==pid2].x.tail(1).values
                y2 = df_traj[df_traj.particle==pid2].y.tail(1).values
                #two tips are the same if their final coordinates are equal to machine precision
                same_tip = bool(x1 == x2) & bool(y1 == y2)
                if same_tip:
                    #pop pid2
                    pid_longest_lst_filtered.remove(pid2)
    return pid_longest_lst_filtered

def find_jumps(x_values,y_values,width=200,height=200, DS=5/200,DT=1, jump_thresh=None):
    '''
    jump_index_array, spd_lst = find_jumps(x_values,y_values,width=200,height=200, DS=5/200,DT=1, jump_thresh=None)
    '''
    #compute the speed of this longest trajectory using pbc
    if jump_thresh is None:
        thresh = np.min((width,height))/2 #threshold displacement to be considered a jump
    else:
        thresh = jump_thresh
    distance_L2_pbc = get_distance_L2_pbc(width,height)
    #     DT = 1.#np.mean(d.t.diff().dropna().values) #ms per frame
    #     DS = 5/200 #cm per pixels
    N = x_values.shape[0]
    spd_lst = []
    spd_lst_naive = []
    for i in range(N-1):
        #compute a speed for i = 0,1,2,...,N-1
        pt_nxt = np.array((x_values[i+1],y_values[i+1]))
        pt_prv = np.array((x_values[i],y_values[i]))
        spd = distance_L2_pbc(pt_nxt,pt_prv)*DS/DT #pixels per ms
        spd_lst.append(spd)
        spd = np.linalg.norm(pt_nxt-pt_prv)
        spd_lst_naive.append(np.linalg.norm(spd))
    boo = (np.array(spd_lst_naive)>thresh)
    #     boo.any()
    jump_index_array = np.argwhere(boo).flatten()
    return jump_index_array, spd_lst


def unwrap_for_each_jump(x_values,y_values,jump_index_array, width=200,height=200):
    '''ux,yv = unwrap_for_each_jump(x_values,y_values,jump_index_array) '''
    yv = y_values.copy()
    xv = x_values.copy()
    for j in  jump_index_array:
        DX = xv[j]-xv[j+1]
        DY = yv[j]-yv[j+1]
        BX = True
        BY = True
        if np.abs(DY)>np.abs(DX):
            #the jump happened over the y boundary
            if DY>0:
                #the jump happend from bottom to top
                if BY:
                    yv[j+1:] = yv[j+1:]+height
#                     BY=False
                else:
                    #taking care of parity
                    BY=True
            else:
                #the jump happened from top to bottom
                if BY:
                    yv[j+1:] = yv[j+1:]-height
#                     BY=False
                else:
                    #taking care of parity
                    BY=True
        else:
            #the jump happened over the x boundary
            if DX>0:
                #the jump happend from left to right
                if BX:
                    xv[j+1:] = xv[j+1:]+width
#                     BX=False
                else:
                    #taking care of parity
                    BX=True

            else:
                #the jump happend from left to right
                if BX:
                    xv[j+1:] = xv[j+1:]-width
#                     BX=False
                else:
                    #taking care of parity
                    BX=True
    return xv,yv


def unwrap_traj_and_center(d):
    '''d is a dataframe of 1 trajectory with pbc.  edits d to have pbc-unwrapped x,y coords and returns d.'''
    if d.t.values.shape[0]<=1:
        return None
    DT = np.mean(d.t.diff().dropna().values) #ms per frame
    DS = 5/200
    x_values = d.x.values.astype('float64')
    y_values = d.y.values.astype('float64')
    jump_index_array, spd_lst = find_jumps(x_values,y_values,DS=DS,DT=DT)
    xv,yv = unwrap_for_each_jump(x_values,y_values,jump_index_array, width=200,height=200)

    #subtract off the initial position for plotting's sake
    xv -= xv[0]
    yv -= yv[0]
    #     return xv,yv

    #store these values in the dataframe
    d = d.copy()
    #store these values in the dataframe
    d.loc[:,'x'] = xv
    d.loc[:,'y'] = yv
    return d

def preprocess_log(input_file_name):
    '''prep and filters raw trajectory
    output_file_name = preprocess_log(input_file_name)
    '''
    #track tips for given input file

    output_file_name = generate_track_tips_pbc(input_file_name, save_fn=None)
    return output_file_name

def unwrap_trajectories(input_file_name, output_file_name):
    # load trajectories
    df = pd.read_csv(input_file_name)
    pid_lst = sorted(set(df.particle.values))
    #(duplicates filtered earlier_ _  _ _ ) filter_duplicate_trajectory_indices is slow (and can probs be accelerated with a sexy pandas one liner)
    pid_lst_filtered = pid_lst#filter_duplicate_trajectory_indices(pid_lst,df)
    # pid_lst_filtered = filter_duplicate_trajectory_indices(pid_lst,df)
    df = pd.concat([unwrap_traj_and_center(df[df.particle==pid]) for pid in pid_lst_filtered])
    df.to_csv(output_file_name,index=False)
    return output_file_name

# ####################################
# # Example Usage
# ####################################
# input_file_name = "/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2/ds_5_param_set_8_fastkernel_V_0.4_archive/Log/ic_200x200.001.11_log.csv"
# output_file_name = preprocess_log(input_file_name)
#
#
# #select the longest n trajectories
# n_tips = 1#15
# s = df.groupby('particle').t.count()
# s = s.sort_values(ascending=False)
# pid_longest_lst = list(s.index.values[:n_tips])
# # d = df[df.particle==pid_longest]
# # print(pid_longest)
# # print(s.head())
# # pid_longest_lst = s.head(n_tips).values
# df_traj = pd.concat([df[df.particle==pid] for pid in pid_longest_lst])
# assert ( (np.array(sorted(set(df_traj['particle'].values)))==np.array(sorted(pid_longest_lst))).all())
#





# #tests
# d = unwrap_traj_and_center(d).copy()
#
# #test that unwrap_traj_and_center removed all jump detections
# x_values = d.x.values.astype('float64')
# y_values = d.y.values.astype('float64')
# jump_index_array, spd_lst = find_jumps(x_values,y_values)
# assert (jump_index_array.size==0)
