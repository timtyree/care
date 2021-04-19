# PlotMSD.py
# Tim Tyree
# 12.19.2020
from ..my_initialization import *
from .. import *
from .dist_func import *

def get_all_longer_than(df,DT,T_min=1000):
    '''df is a pandas.DataFrame instance with columns, x, y, t, frame, and particle.
    returns a pandas.DataFrame instance with only the trajectories that last at least T_min, in same time units as DT.
    also removes all trajectories that do not move.'''
    durations=DT*df.groupby('particle').x.count()
    boo=T_min<durations
    #identify any particles that live less time than T_min
    pid_drop_lst=durations[~boo].index.values
    boo_drop=False
    for pid in pid_drop_lst:
        boo_drop|=df.particle==pid
    #identify any particles that do not move
    stdx=df.groupby('particle').x.std()
    boo_drop|=stdx==0
    #drop those marked particles
    df.drop(index=boo_drop[boo_drop].index,inplace=True)
    return df

def find_jumps(x_values,y_values, width, height, DS, DT, jump_thresh=None, distance_L2_pbc=None, **kwargs):
    '''Example Usage:
    jump_index_array, spd_lst = find_jumps(x_values,y_values,width=200,height=200, DS,DT, jump_thresh=None)
    spd_lst is a list of speeds in units of DS/DT.'''
    #compute the speed of this longest trajectory using pbc
    if jump_thresh is None:
        thresh = np.min((width,height))/2 #threshold displacement to be considered a jump
    else:
        thresh = jump_thresh
    if distance_L2_pbc is None:
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

def find_jumps_non_pbc(x_values,y_values,width,height, DS,DT, jump_thresh=None,distance_L2_pbc=None, **kwargs):
    '''
    jump_index_array, spd_lst = find_jumps(x_values,y_values,width=200,height=200, DS=5/200,DT, jump_thresh=None)
    spd_lst is a list of speeds in units of DS/DT.'''
    #compute the speed of this longest trajectory using pbc
    if jump_thresh is None:
        thresh = np.min((width,height))/2 #threshold displacement to be considered a jump
    else:
        thresh = jump_thresh
    if distance_L2_pbc is None:
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
        spd = distance_L2_pbc(pt_nxt,pt_prv)#*DS/DT #pixels per ms
        spd_lst.append(spd)
        spd = np.linalg.norm(pt_nxt-pt_prv)
        spd_lst_naive.append(np.linalg.norm(spd))
    boo = (np.array(spd_lst)>thresh)
    #     boo.any()
    jump_index_array = np.argwhere(boo).flatten()
    return jump_index_array, spd_lst


def return_longest_n_and_truncate(input_file_name,n_tips, DS, DT, round_t_to_n_digits, width,height,**kwargs):
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
        jump_index_array, spd_lst = find_jumps(x_values,y_values,width=width,height=height, DS=DS,DT=DT, jump_thresh=20.,**kwargs)#.25)
        if len(jump_index_array)>0:
            ji = jump_index_array[0]
            d.drop(index=index_values[ji:], inplace=True)
        df_lst.append(d)
    df_traj = pd.concat(df_lst)
    #round trajectory times to remove machine noise from floating point arithmatic
    df_traj['t'] = df_traj.t.round(round_t_to_n_digits)
    return df_traj

def comparable_velocities(d,d_alt,angle_threshold = np.pi/4, num_tail=5,num_head=5):
    '''a comparable_velocity filter for whether d and d_alt have tails and heads have similar directions of mean velocity
    boo = comparable_velocities(d,d_alt,angle_threshold = np.pi/4)'''
    #DONE: compute mean_tail_velocity of tip to be merged into, d
    vx = np.mean(d.tail(num_tail).x.diff().dropna().values)
    vy = np.mean(d.tail(num_tail).y.diff().dropna().values)
    mean_tail_velocity = np.array([vx,vy]).T

    #the tip to be merged is d_alt
    #TODO: compute mean_head_velocity of tip to be merged, d_alt
    vx = np.mean(d_alt.head(num_head).x.diff().dropna().values)
    vy = np.mean(d_alt.head(num_head).y.diff().dropna().values)
    mean_head_velocity = np.array([vx,vy]).T
    #TODO: compute the angle between these two velocities
    a = np.dot(mean_head_velocity,mean_tail_velocity)
    a /= np.linalg.norm(mean_head_velocity)
    a /= np.linalg.norm(mean_tail_velocity)
    angle = np.arccos(a)
    boo = angle<angle_threshold
    return boo

def comp_lifetime_diff_and_avg(d,d_other):
    '''absdiff,avgval=comp_lifetime_diff_and_avg(d,d_other)'''
    tmin_other = d_other.t.min()
    tmax_other = d_other.t.max()
    lifetime_other = tmax_other-tmin_other
    tmin_other = d.t.min()
    tmax_other = d.t.max()
    lifetime_self = tmax_other-tmin_other
    absdiff = np.abs(lifetime_self-lifetime_other)
    avgval = lifetime_self/2+lifetime_other/2
    return absdiff,avgval

def comp_lifetime(d):
    '''lifetime_self=comp_lifetime(d)'''
    tmin_other = d.t.min()
    tmax_other = d.t.max()
    lifetime_self = tmax_other-tmin_other
    #     absdiff = np.abs(lifetime_self-lifetime_other)
    #     avgval = lifetime_self/2+lifetime_other/2
    return lifetime_self

def get_neighboring_tip(xy_self,xy_others, pid_others,distance_L2_pbc):
    """nearest_pid, nearest_dist = get_neighboring_tip(xy_self,xy_others)"""
    nearest_pid  =  pid_others[0]
    nearest_dist = distance_L2_pbc ( xy_others[0], xy_self[0])
    if len(pid_others)>1:
        for j,pid_other in enumerate(pid_others):
            dist = distance_L2_pbc ( xy_others[j], xy_self[0])
            if dist<nearest_dist:
                nearest_dist = dist
                nearest_pid  = int(pid_other)
    return nearest_pid, nearest_dist

def get_tips_in_range(xy_self,xy_others, pid_others, distance_L2_pbc,dist_thresh=10):
    """pid_lst = get_tips_in_range(xy_self,xy_others, pid_others, dist_thresh=10)"""
    pid_lst = []
    if len(pid_others)>0:
        for j,pid_other in enumerate(pid_others):
            dist = distance_L2_pbc ( xy_others[j], xy_self[0])
            if dist<dist_thresh:
                pid_lst.append (  int(pid_other) )
    return pid_lst

def identify_birth_partner(df,cid,distance_L2_pbc,cid_others=None,verbose=False):
    """identify birth mate using set difference.
    Example Usage:
    cid_birthmate, nearest_dist_birth, t_birth = identify_birth_partner(df,cid,distance_L2_pbc,cid_others=None)
    """
    #self
    d = df[df.cid == cid]
    x,y,frm,t = d.head(1)[['x','y','frame','t']].values.T
    frm_birth=frm
    xy_self = np.array((x,y)).T
    #others that died in the same frame
    if cid_others is None:
        cid_others = df[(df.frame==int(frm))&(df.cid!=cid)]['cid'].values.T
    cid_others_nxt = df[(df.frame==int(frm)+1)&(df.cid!=cid)]['cid'].values.T
    cid_others_prv = df[(df.frame==int(frm)-1)&(df.cid!=cid)]['cid'].values.T
    # cid_born_lst=sorted(set(list(cid_others_nxt)).difference(set(list(cid_others))))
    cid_born_lst=sorted(set(list(cid_others)).difference(set(list(cid_others_prv))))
    try:
        assert(len(cid_born_lst)>0)
    except Exception as e:
        if verbose:
            print('birth exception')
            print(sorted(cid_others_prv))
            print(sorted(cid_others))
        # print(e)
        return -9999,np.nan, float(t)
        # assert(len(cid_born_lst)>0)
    #at the time of birth/death, the suspects were...
    cid_others=np.array(cid_born_lst)
    boo = (df.frame!=df.frame)#tautologically False
    for cid_other in cid_others:
        boo |=(df.cid==cid_other)
    boo &= (df.frame==int(frm_birth)) #select only the cid_others in the death frame
    x_others,y_others = df[boo][['x','y']].values.T
    xy_others = np.vstack((x_others,y_others)).T
    cid_birthmate, nearest_dist = get_neighboring_tip(xy_self,xy_others,cid_others,distance_L2_pbc)
    return cid_birthmate, nearest_dist, float(t)


def identify_death_partner(df,cid,distance_L2_pbc,cid_others=None,verbose=False):
    """identify death mate using set difference.
    Example Usage:
    cid_deathmate, nearest_dist_death, t_death = identify_death_partner(df,cid,distance_L2_pbc)
    """
    #self
    d = df[df.cid == cid]
    x,y,frm,t = d.tail(1)[['x','y','frame','t']].values.T
    frm_death=frm
    df['keep']=True
    xy_self = np.array((x,y)).T
    #others that died in the same frame
    if cid_others is None:#&(df.keep)
        cid_others = df[(df.frame==int(frm))&(df.cid!=cid)&(df.keep)]['cid'].values.T
    cid_others_nxt = df[(df.frame==int(frm)+1)&(df.cid!=cid)]['cid'].values.T
    cid_died_lst=sorted(set(list(cid_others)).difference(set(list(cid_others_nxt))))
    try:
        assert(len(cid_died_lst)>0)
    except Exception as e:
        if verbose:
            print('death exception')
            print(sorted(cid_others))
            print(sorted(cid_others_nxt))
        # print(e)
        return -9999,np.nan, float(t)
    #at the time of birth/death, the suspects were...
    cid_others=np.array(cid_died_lst)
    boo = (df.frame!=df.frame)#tautologically False
    for cid_other in cid_others:
        boo |=(df.cid==cid_other)
    boo &= (df.frame==int(frm_death)) #select only the cid_others in the death frame
    x_others,y_others = df[boo][['x','y']].values.T
    xy_others = np.vstack((x_others,y_others)).T
    cid_deathmate, nearest_dist = get_neighboring_tip(xy_self,xy_others,cid_others,distance_L2_pbc)
    return cid_deathmate, nearest_dist, float(t)




# def identify_death_partner(df,cid,distance_L2_pbc):
#     '''Example usage:
#     nearest_cid, nearest_dist, t = identify_death_partner(df,cid,distance_L2_pbc)
#     '''
#     # f = df
#     d = df[df.cid == cid]
#     # pid = sorted(set(d.particle.values))[0]
#     # d = f[f.particle == pid]
#     #identify the death partner
#     x,y,t = d.tail(1)[['x','y','t']].values.T
#     #at the time of birth/death, the suspects were...
#     # x_others,y_others,pid_others = f[(f.t==float(t))&(f.particle!=pid)][['x','y','particle']].values.T
#     x_others,y_others,cid_others = df[(df.t==float(t))&(df.cid!=cid)][['x','y','cid']].values.T
#     xy_others = np.vstack((x_others,y_others)).T
#     xy_self = np.array((x,y)).T
#     nearest_cid, nearest_dist = get_neighboring_tip(xy_self,xy_others,cid_others,distance_L2_pbc)
#     return nearest_cid, nearest_dist, t

# def identify_birth_partner(df,cid,distance_L2_pbc):
#     '''
#     nearest_cid, nearest_dist, t_birth = identify_birth_partner(df,cid,distance_L2_pbc)'''
#     # f = df
#     d = df[df.cid == cid]
#     # pid = sorted(set(d.particle.values))[0]
#     # d = f[f.particle == pid]
#     #identify the death partner
#     x,y,t = d.head(1)[['x','y','t']].values.T
#     #at the time of birth/death, the suspects were...
#     # x_others,y_others,pid_others = f[(f.t==float(t))&(f.particle!=pid)][['x','y','particle']].values.T
#     x_others,y_others,cid_others = df[(df.t==float(t))&(df.cid!=cid)][['x','y','cid']].values.T
#     xy_others = np.vstack((x_others,y_others)).T
#     xy_self = np.array((x,y)).T
#     nearest_cid, nearest_dist = get_neighboring_tip(xy_self,xy_others,cid_others,distance_L2_pbc)
#     return nearest_cid, nearest_dist, t
