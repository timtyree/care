#retreck_tips.py
from ..my_initialization import *
from ..utils import *

def decompose_trajectories(df, distance_L2_pbc,DS,DT,
    width,height,LT_thresh,tmin, jump_thresh=10., **kwargs):
    '''
    reads trajectories from the .csv file, input_file_name.
    remove any trajectories with no more than LT_thresh rows.
    remove any rows that occur earlier than tmin millaseconds.
    break the trajectories in input_file_name up into valid chunks indexed by the column `cid`.
    returns pandas.DataFrame
    '''
    # input_file_name = "/home/timothytyree/Documents/GitHub/care/notebooks/Data/initial-conditions-suite-2/ds_5_param_set_8_fastkernel_V_0.5_archive/trajectories/ic_200x200.001.22_traj_sr_400_mem_2.csv"
    # df = pd.read_csv(input_file_name)
    # try:
    #     df.drop(columns=['index'])
    #     print('index col dropped')
    # except:
    #     pass

    # df.reset_index(inplace=True)

    #list of length sorted trajectories
    s = df.groupby('particle').t.count()
    s = s.sort_values(ascending=False)
    #filter based on min trajectory length
    pid_black_lst = list(s[s.values<=LT_thresh].index.values)
    for pid in pid_black_lst:
        df = df.drop(labels=df[df.particle==pid].index)
    pid_longest_lst = list(s[s.values>LT_thresh].index.values)
    # pid_longest_lst = list(s.index.values)
    df = df[df.t>tmin].copy()
    # if distance_L2_pbc is None:
    #     distance_L2_pbc = get_distance_L2_pbc(width=width,height=height)
    df['cid'] = -9999
    df = chunk_traj(df,pid_lst=pid_longest_lst,width=width,height=height,jump_thresh=jump_thresh, LT_thresh=LT_thresh,distance_L2_pbc=distance_L2_pbc, DS=DS, DT=DT)
    assert(df.cid.max()>0)
    return df
    # d_lst = chunk_traj(df,pid_lst=pid_longest_lst,width=width,height=height,jump_thresh=10., LT_thresh=1,distance_L2_pbc=None):
    # return d_lst

def retrack_trajectories(df,distance_L2_pbc,LT_thresh,DS,width,height, jump_thresh=20., lifetime_thresh = 50,angle_threshold = np.pi/4, **kwargs):
    '''takes df returned by decompose_trajectories
    compute a new particle2 field for df, which is partitioned by the list d_lst,
    performing patches between two trajectories if the following conditions all hold:
    - the (nearest possible) deathmate of the given tip lives less than lifetime_thresh milliseconds, and
    - the (nearest possible) birthmate of that (nearest possible) deathmate begins with a mean velocity
     that is sufficiently consistent in direction to the final mean velocity
     of the given tip trajectory.
    '''
    #assert chunk id has been initialized
    assert(df.cid.max()>0)
    df['particle2'] = -9999
    pid2counter = 1
    cid_lst = sorted(set(df.cid.values))
    for n,cid in enumerate(cid_lst):
        d = df[(df.cid == cid)]#&(df.particle2<0)]
        #if there are not any particle2 assignments for this trajectory chunk
        if not (d.particle2>=0).any():
            #then initiate an attention head on this cid
            df,pid2counter = patch_traj_head(df,cid,pid2counter, distance_L2_pbc, lifetime_thresh, jump_thresh, angle_threshold, mode='backward')
        elif not (d.particle2<0).all():
            d = df[df.cid==cid]
            #I don't think this matters, perhaps.
            # print(f"Warning: some but not all rows where cid=={cid} already had a value for particle2!")
            # print(f"They were pid2 were in {{ {set(d.particle2)} }}")
            # print(f"Whilst pid2counter=={pid2counter}.")
            # print(f"\nd.head().particle2.values=={d.head().particle2.values}")
            # print(f"\nd.tail().particle2.values=={d.tail().particle2.values}")
    return df,pid2counter

def patch_traj_head(df,cid,pid2counter, distance_L2_pbc, lifetime_thresh, jump_thresh, angle_threshold, mode='backward'):
    '''a short lifetime deathmate is present a nontrivial patch might need to be made'''
    d_self = df[(df.cid == cid)]#&(df.particle2<0)]#df[df.cid == cid]
    if d_self.t.count()<1:
        #d_self is empty, return no change to df, effectively skipping this cid
        return df,pid2counter
    #identify the death partner
    nearest_cid, reaction_distance_death, t_of_death = identify_death_partner(df=df,cid=cid,distance_L2_pbc=distance_L2_pbc)
    #identify the birth partner of that death partner
    cid_alternative, reaction_distance_birth, t_of_life = identify_birth_partner(df=df,cid=nearest_cid,distance_L2_pbc=distance_L2_pbc)
    lifetime_of_killer = float(t_of_death-t_of_life)
    # if the killer is long lived,
    if lifetime_of_killer>lifetime_thresh:
        # copy particle into particle2 and end iteration for this attention head
        df.loc[df.cid==cid,'particle2'] = int(pid2counter)
        pid2counter += 1
        return df,pid2counter

    # and if the proposed alternative doesn't have unaccounted for rows to add to the end
    d_alt = df[(df.cid == cid_alternative)&(df.particle2<0)]
    frame_of_death = d_self.frame.max()
    num_rows_addable = d_alt.frame.max()-frame_of_death
    if num_rows_addable<1:
        # copy particle into particle2 and end iteration for this attention head
        df.loc[df.cid==cid,'particle2'] = int(pid2counter)
        pid2counter += 1
        return df,pid2counter

    # and if the proposed jump is too far
    xy_start = d_self[(d_self.frame==frame_of_death)][['x','y']].values.T[0]
    xy_end = d_alt[(d_alt.frame==frame_of_death+1)][['x','y']].values.T[0]
    range_of_jump = distance_L2_pbc(xy_end,xy_start)
    if range_of_jump>jump_thresh:
        # the tip must be valid without the patch, so copy particle to particle2 or
        df.loc[df.cid==cid,'particle2'] = int(pid2counter)
        pid2counter += 1
        return df,pid2counter

    # and if the velocities are not comparable
    boo = comparable_velocities(d_self,d_alt,angle_threshold=angle_threshold, num_tail=2,num_head=2)
    if not boo:
        # the tip must be valid without the patch, so copy particle to particle2 or
        df.loc[df.cid==cid,'particle2'] = int(pid2counter)
        pid2counter += 1
        return df,pid2counter

    # ONLY then, patch the two tip trajectory chunks, and recursively call on cid_alt
    # patch the two tips
    if mode=='backward':
        #backward consistent jump
        df.particle2 = d_self.particle*0+int(pid2counter)
        df.particle2 = d_alt.particle*0+int(pid2counter)
    elif mode=='forward':
        #forward consistent jump
        df.particle2 = d_alt.particle*0+int(pid2counter)
        df.particle2 = d_self.particle*0+int(pid2counter)
    else:
        raise("Error: mode must be `forward` or `backward`. Ignoring patch.")
        df.loc[df.cid==cid,'particle2'] = int(pid2counter)
        pid2counter += 1
        return df,pid2counter
        #TODO: recursive attention head for f.particle2==pid2counter
        # pid2counter += 1
    return patch_traj_head(df,cid,pid2counter, distance_L2_pbc, lifetime_thresh, jump_thresh, angle_threshold, mode=mode)
