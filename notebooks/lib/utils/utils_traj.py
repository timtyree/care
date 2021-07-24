# PlotMSD.py
# Tim Tyree
# 12.19.2020
from ..my_initialization import *
from .. import *
from .dist_func import *

def unwrap_for_each_jump(x_values,y_values,jump_index_array, width,height,
    jump_thresh=None, **kwargs):
    '''unwrap_for_each_jump iterates over the jumps indexed in jump_index_array
    and shifts all positions later in the trajectory such that the distance
    between consecutive points is minimized.
    Both boundaries may be crossed independently

    Example Usage:
    ux,yv = unwrap_for_each_jump(x_values,y_values,jump_index_array, width,height)
    '''
    yv = y_values.copy()
    xv = x_values.copy()
    for j in  sorted(jump_index_array):
        DX = xv[j]-xv[j+1]
        DY = yv[j]-yv[j+1]
        # BX = True
        # BY = True
        if np.abs(DY)>jump_thresh:
            #a jump happened over the y boundary
            if DY>0:
                #the jump happend from bottom to top
                yv[j+1:] = yv[j+1:]+height
            else:
                #the jump happened from top to bottom
                yv[j+1:] = yv[j+1:]-height

        if np.abs(DX)>jump_thresh:
            #a jump happened over the x boundary
            if DX>0:
                #the jump happend from left to right
                xv[j+1:] = xv[j+1:]+width
            else:
                #the jump happened from right to left
                xv[j+1:] = xv[j+1:]-width
    return xv,yv

def get_DT(df,t_col='t',pid_col='particle'):
    '''
    Example Usage:
    DT=get_DT(df) #ms
    '''
    DT=np.mean(df[df[pid_col]==0][t_col].diff().dropna().values)
    return DT

def unwrap_xy_values(x_values,y_values,width,height,jump_thresh=None,**kwargs):
    '''returns a path with periodic boundary conditions that is unwrapped and centered
    if jump_thresh=None, then jump_thresh=np.min(width,height)/2.
    TODO: add support for re-using jit compiled distance_L2_pbc'''
    jump_index_array, spd_lst = find_jumps(x_values,y_values,width,height,jump_thresh=jump_thresh,**kwargs)
    xv,yv = unwrap_for_each_jump(x_values,y_values,jump_index_array, width=width,height=height,jump_thresh=jump_thresh,**kwargs)
    return xv,yv

def unwrap_and_center_xy_values(x_values,y_values,width,height,jump_thresh=None,**kwargs):
    '''returns a path with periodic boundary conditions that is unwrapped and centered
    if jump_thresh=None, then jump_thresh=np.min(width,height)/2.
    TODO: add support for re-using jit compiled distance_L2_pbc'''
    xv,yv=unwrap_xy_values(x_values,y_values,width,height,jump_thresh=jump_thresh,**kwargs)
    #subtract off the initial position for plotting's sake
    xv -= xv[0]
    yv -= yv[0]
    return xv,yv

def unwrap_traj_and_center(d, width, height,jump_thresh=None,**kwargs):
    '''d is a dataframe of 1 trajectory with pbc listed in fields 'x' and 'y'.  edits d to have pbc-unwrapped x,y coords and returns d.
    jump_thresh is the minimum displacement to be considered a jump
    '''
    if d['x'].values.shape[0]==0:
        return None
    # DT = np.mean(d[t_col].diff().dropna().values) #ms per frame
    # DS = 5/200
    x_values = d.x.values.astype('float64')
    y_values = d.y.values.astype('float64')
    xv,yv = unwrap_and_center_xy_values(x_values,y_values,width,height,jump_thresh=jump_thresh,**kwargs)

    #store these values in the dataframe
    d = d.copy()
    #store these values in the dataframe
    d.loc[:,'x'] = xv
    d.loc[:,'y'] = yv
    return d




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


def get_all_longer_than(df,DT,T_min=1000,pid_col='particle'):
    '''df is a pandas.DataFrame instance with columns, x, y, t, frame, and some pid_col.
    returns a pandas.DataFrame instance with only the trajectories that last at least T_min, in same time units as DT.
    also removes all trajectories that do not move.'''
    durations=DT*df.groupby(pid_col).x.count()
    boo=T_min<durations
    #identify any particles that live less time than T_min
    pid_drop_lst=durations[~boo].index.values
    boo_drop=False
    for pid in pid_drop_lst:
        boo_drop|=df[pid_col]==pid
    #identify any particles that do not move
    stdx=df.groupby(pid_col).x.std()
    boo_drop|=stdx==0
    #drop those marked particles
    df.drop(index=boo_drop[boo_drop].index,inplace=True)
    return df

def find_jumps(x_values,y_values, width, height, jump_thresh=None, distance_L2_pbc=None, **kwargs):
    '''Example Usage:
    jump_index_array, spd_lst = find_jumps(x_values,y_values,width=200,height=200, DS,DT, jump_thresh=None)
    spd_lst is a list of speeds in units of DS/DT.
    x_values and y_values are numpy.array instances that describe a 2-dimensional trajectory that potentially exhibits jumps.
    jump_thresh is the minimum threshold displacement to be considered a jump in the same units as x and y.
    '''
    #compute the speed of this longest trajectory using pbc
    if jump_thresh is None:
        thresh = np.min((width,height))/2 #minimum threshold displacement to be considered a jump
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


def return_longest_n_and_truncate(input_file_name,n_tips, DS, DT, round_t_to_n_digits, width,height,jump_thresh=None,**kwargs):
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
        jump_index_array, spd_lst = find_jumps(x_values,y_values,width=width,height=height, jump_thresh=jump_thresh,**kwargs)#.25)
        if len(jump_index_array)>0:
            ji = jump_index_array[0]
            d.drop(index=index_values[ji:], inplace=True)
        df_lst.append(d)
    df_traj = pd.concat(df_lst)
    #round trajectory times to remove machine noise from floating point arithmatic
    df_traj['t'] = df_traj.t.round(round_t_to_n_digits)
    return df_traj

def comparable_velocities(d,d_alt,angle_threshold = np.pi/4, num_tail=5,num_head=5):
    '''TODO: compute velocities with pbc
    a comparable_velocity filter for whether d and d_alt have tails and heads have similar directions of mean velocity
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

def identify_birth_partner(df,cid,distance_L2_pbc,cid_others=None,verbose=False,pid_col='particle'):
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


def identify_death_partner(df,cid,distance_L2_pbc,cid_others=None,verbose=False,pid_col='particle'):
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
