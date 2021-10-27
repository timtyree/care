from .. import *
import numpy as np, cudf, cupy as cp, itertools
def comp_intersecting_pairs_cu(df,pid_col='particle',trial_col='event_id_int',**kwargs):
    '''returns a cudf.DataFrame instance that identifies pairs of particles that coexist along with their start and end times.
    df is s cudf.DataFrame instance with columns tmin and tmax along with a unique index for each row and a particle id recorded in the column indicated by pid_col'''
    dfff=df
    #itertools appears to need to be run on cpu only this is the main time sink...
    trial_values=dfff[trial_col].drop_duplicates().values.get()
    index_pair_values_lst=[]
    #iterate over the trials
    for trial in trial_values:
        index_values=dfff[dfff[trial_col]==trial].index.values.get().copy()
        index_pair_values_lst.append(cp.array(list(itertools.combinations(index_values, 2))))
    index_pair_values=cp.concatenate(index_pair_values_lst,axis=0)
    del index_pair_values_lst
    #identify pairs that intersect with at least two time points
    tmin_self_values=dfff.loc[index_pair_values[:,0],'tmin'].values
    tmax_self_values=dfff.loc[index_pair_values[:,0],'tmax'].values
    tmin_other_values=dfff.loc[index_pair_values[:,1],'tmin'].values
    tmax_other_values=dfff.loc[index_pair_values[:,1],'tmax'].values
    boo_intersecting=(tmin_other_values<tmax_self_values) & (tmax_other_values>tmin_self_values)
    try:
        intersecting_index_pair_values=index_pair_values[boo_intersecting]
    except IndexError:
        intersecting_index_pair_values=index_pair_values[boo_intersecting.flatten()]
    #query and record results
    col_lst=[pid_col,'tmin','tmax']
    df_self=dfff.loc[intersecting_index_pair_values[:,0],col_lst]
    df_other=dfff.loc[intersecting_index_pair_values[:,1],col_lst]
    df_intersecting_pairs=cudf.DataFrame({
        'pid_self':df_self[pid_col].values,
        'pid_other':df_other[pid_col].values,
        'tmin_self':df_self['tmin'].values,
        'tmin_other':df_other['tmin'].values,
        'tmax_self':df_self['tmax'].values,
        'tmax_other':df_other['tmax'].values,
    })
    df_intersecting_pairs['tmin']=df_intersecting_pairs[["tmin_self", "tmin_other"]].max(axis=1)
    df_intersecting_pairs['tmax']=df_intersecting_pairs[["tmax_self", "tmax_other"]].min(axis=1)
    df_intersecting_pairs['duration']=df_intersecting_pairs['tmax']-df_intersecting_pairs['tmin']
    return df_intersecting_pairs


###################
# Deprecated :,(
###################
# def extract_xy_trajectory_pairs_cu(df,df_pairs,pid_col='particle',t_col='t', dropnull=True):
#     '''df_pairs is returned by comp_intersecting_pairs_cu
#     df contains only one trial (i.e. pid_col uniquely indexes all particles)
#     returns a cudf.DataFrame that has the xy positions from the particles indicated from pid_self to pid_other
#     Example Usage:
#     df_traj=extract_xy_trajectory_pairs_cu(df,df_pairs,pid_col=pid_col,t_col=t_col)
#     '''
#     # targ_index_col_lst=[trial_col,pid_col,t_col]
#     targ_index_col_lst=[pid_col,t_col]
#     dfmi=df.set_index(targ_index_col_lst)
#
#     #SUPPOSING THERE IS ONLY ONE EVENT_ID_INT PRESENT
#     # list(zip(targ_index_col_lst,targ_index_col_lst
#     # dmis=df_pairs.set_index([trial_col,pid_col])
#     # dmit=df.set_index([trial_col,pid_col])
#
#     df_pairs[pid_col]=df_pairs['pid_self']
#     dmis=df_pairs.set_index(pid_col)
#     dmit=df.set_index(pid_col)
#     df_traj=dmit.loc[dmis.index.values].copy().reset_index()
#     df_traj.rename(columns={pid_col:'pid_self'},inplace=True)
#
#     df_pairs[pid_col]=df_pairs['pid_other']
#     dmis=df_pairs.set_index(pid_col)
#     dmit=df.set_index(pid_col)
#     df_traj_other=dmit.loc[dmis.index.values].copy().reset_index()
#     df_traj_other.rename(columns={pid_col:'pid_other','x':'x_other','y':'y_other'},inplace=True)
#
#     df_traj['pid_other']=df_traj_other['pid_other']
#     df_traj['x_other']=df_traj_other['x_other']
#     df_traj['y_other']=df_traj_other['y_other']
#     if dropnull:
#         #remove any time points from self that are not specified in other
#         df_traj.dropna(inplace=True)
#     return df_traj
#
# def extract_trajectory_pairs_cu(df,df_pairs,pid_col='particle',t_col='t', dropnull=True):
#     '''df_pairs is returned by comp_intersecting_pairs_cu
#     df contains only one trial (i.e. pid_col uniquely indexes all particles)
#     returns a cudf.DataFrame that has the positions from the particles indicated from pid_self to pid_other
#     Example Usage:
#     df_traj=extract_xy_trajectory_pairs_cu(df,df_pairs,pid_col=pid_col,t_col=t_col)
#     '''
#     # targ_index_col_lst=[trial_col,pid_col,t_col]
#     # targ_index_col_lst=[pid_col,t_col]
#     # dfmi=df.set_index(targ_index_col_lst)
#
#     #SUPPOSING THERE IS ONLY ONE EVENT_ID_INT PRESENT
#     # list(zip(targ_index_col_lst,targ_index_col_lst
#     # dmis=df_pairs.set_index([trial_col,pid_col])
#     # dmit=df.set_index([trial_col,pid_col])
#
#     # df_pairs[pid_col]=df_pairs['pid_self']
#     # dmis=df_pairs.set_index(pid_col)
#     dmis=df_pairs.set_index('pid_self')
#     dmit=df.set_index(pid_col)
#     # pid_other_values=dmis['pid_other'].values
#     df_traj=dmit.loc[dmis.index.values].copy()
#     df_traj['pid_other']=dmis['pid_other']
#     # df_traj=df_traj.reset_index()
#     df_traj.reset_index(inplace=True)
#     df_traj.rename(columns={pid_col:'pid_self'},inplace=True)
#
#
#     # df_pairs[pid_col]=df_pairs['pid_other']
#     # dmis=df_pairs.set_index(pid_col)
#     dmis=df_pairs.set_index('pid_other')
#     dmit=df.set_index(pid_col)
#     df_traj_other=dmit.loc[dmis.index.values].copy().reset_index()
#     df_traj_other.rename(columns={pid_col:'pid'},inplace=True)
#     col_lst=list(df_traj_other.columns)
#     other_col_lst=[col+'_other' for col in col_lst]
#     df_traj_other.rename(columns=dict(zip(col_lst,other_col_lst)),inplace=True)
#     df_traj_other.rename(columns={t_col+'_other':t_col},inplace=True)
#
#     #index both df_traj instances by the same columns
#     df_traj=df_traj.set_index(['pid_other',t_col])
#     df_traj_other=df_traj_other.set_index(['pid_other',t_col])
#     for col in other_col_lst:
#         df_traj[col]=df_traj_other[col]
#     if dropnull:
#         #remove any time points from self that are not specified in other
#         df_traj.dropna(inplace=True)
#     return df_traj
