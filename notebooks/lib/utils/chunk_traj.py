#compute_traj.py
from ..my_initialization import *
from . import *
def chunk_traj(df,pid_lst,width,height, DS, DT, jump_thresh=10., distance_L2_pbc=None, LT_thresh=1, **kwargs):
    # d_lst = []
    chunk_index=1
    if distance_L2_pbc is None:
        distance_L2_pbc = get_distance_L2_pbc(width=width,height=height)
    for pid in  pid_lst:
        d_raw = df[df.particle==pid].copy()
        #drop any rows before t=100ms
        #drop any rows that already have a value in particle2
        d_raw.reset_index(inplace=True)#,drop=True)
        x_values, y_values, c_values = d_raw[['x','y', 't']].values.T
        jump_index_array, spd_lst = find_jumps_non_pbc(x_values,y_values,distance_L2_pbc=distance_L2_pbc,width=width,height=height, DS=DS,DT=DT, jump_thresh=None)#.25)
        # jump_index_array_pbc, spd_lst = find_jumps(x_values,y_values,distance_L2_pbc=distance_L2_pbc,width=width,height=height, DS=DS,DT=DT, jump_thresh=None)#.25)
        # jump_index_array, spd_lst = find_jumps(x_values,y_values,distance_L2_pbc=distance_L2_pbc,width=width,height=height, DS=DS,DT=DT, jump_thresh=jump_thresh)#.25)
        # jump_index_array=sorted(set(jump_index_array).difference(set(jump_index_array_pbc)))
        jarry=np.hstack([jump_index_array,-9999])
        Nj = jarry.shape[0]
        for j,ji in enumerate(jarry):
            if ji==-9999:
                if len(jump_index_array)==0:
                    #no jumps exist
                    d = d_raw
                else:
                    #this is the final jump to the end
                    ji_prv=jarry[j-1]
                    d = d_raw.iloc[ji_prv:]#.copy()
            elif j==0:
                #this is the beginning up until the first jump
                d = d_raw.iloc[:ji]#.copy()
            else:#elif j<Nj:
                #this is an intermediate jump
                ji_prv=jarry[j-1]
                d = d_raw.iloc[ji_prv:ji]#.copy()
            # else:
            #     d = d_raw.iloc[ji:]#.copy()
            #         for ji in jump_index_array:
            #record datum only for long trajectory segments? yes.
            if d.t.count()>LT_thresh:
                #reset the index back to that of df
                # d = d.reindex(d['index'],copy=False).copy()
                df.loc[d['index'].values,'cid']=chunk_index
                chunk_index +=1
                # d_lst.append(d)
    return df


# def chunk_traj(df,pid_lst,width=200,height=200,jump_thresh=10., distance_L2_pbc=None, LT_thresh=1):
#     d_lst = []
#     if distance_L2_pbc is None:
#         distance_L2_pbc = get_distance_L2_pbc(width=200,height=200)
#     for pid in  pid_lst:
#         d_raw = df[df.particle==pid].copy()
#         #drop any rows before t=100ms
#         #drop any rows that already have a value in particle2
#         d_raw.reset_index(inplace=True)#,drop=True)
#         x_values ,y_values, c_values = d_raw[['x','y', 't']].values.T
#
#         jump_index_array, spd_lst = find_jumps(x_values,y_values,distance_L2_pbc,width=width,height=height, DS=DS,DT=DT, jump_thresh=jump_thresh)#.25)
#         jarry=np.hstack([0,jump_index_array])
#         Nj = jarry.shape[0]
#         for j,ji in enumerate(jarry):
#             if j<Nj-1:
#                 ji_next=jarry[j+1]
#                 d = d_raw.iloc[ji:ji_next].copy()
#             else:
#                 d = d_raw.iloc[ji:].copy()
#             #         for ji in jump_index_array:
#             #record datum only for long trajectory segments? yes.
#             if d.t.count()>LT_thresh:
#                 #reset the index back to that of df
#                 d = d.reindex(d['index'],copy=False).copy()
#                 d_lst.append(d)
#     return d_lst
