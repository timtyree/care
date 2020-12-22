# PlotMSD.py
# Tim Tyree
# 12.19.2020
from ..my_initialization import *
from .. import *
from .dist_func import *


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

def return_longest_n_and_truncate(input_file_name,n_tips = 1,DS = 5/200,DT = 1., round_t_to_n_digits=0):
    #select the longest n trajectories
    df = pd.read_csv(input_file_name)
    df.reset_index(inplace=True)
    s = df.groupby('particle').t.count()
    s = s.sort_values(ascending=False)
    pid_longest_lst = list(s.index.values[:n_tips])
    #     df_traj = pd.concat([df[df.particle==pid] for pid in pid_longest_lst])
    #truncate trajectories to their first apparent jump (pbc jumps should have been removed already)
    df_lst = []
    for pid in  pid_longest_lst:#[2:]:
        d = df[(df.particle==pid)].copy()
        x_values, y_values = d[['x','y']].values.T
        index_values = d.index.values.T
        jump_index_array, spd_lst = find_jumps(x_values,y_values,width=200,height=200, DS=5/200,DT=1, jump_thresh=20.)#.25)
        if len(jump_index_array)>0:
            ji = jump_index_array[0]
            d.drop(index=index_values[ji:], inplace=True)
        df_lst.append(d)
    df_traj = pd.concat(df_lst)
    #round trajectory times to remove machine noise from floating point arithmatic
    df_traj['t'] = df_traj.t.round(round_t_to_n_digits)
    return df_traj
