from .. import *
import numpy as np, cudf, cupy as cp, itertools
def comp_intersecting_pairs_cu(df,pid_col='particle',**kwargs):
    '''returns a cudf.DataFrame instance that identifies pairs of particles that coexist along with their start and end times.
    df is s cudf.DataFrame instance with columns tmin and tmax along with a unique index for each row and a particle id recorded in the column indicated by pid_col'''
    dfff=df
    index_values=dfff.index.values.get()
    index_pair_values=cp.array(list(itertools.combinations(index_values, 2)))
    tmin_self_values=dfff.loc[index_pair_values[:,0],'tmin'].values
    tmax_self_values=dfff.loc[index_pair_values[:,0],'tmax'].values
    tmin_other_values=dfff.loc[index_pair_values[:,1],'tmin'].values
    tmax_other_values=dfff.loc[index_pair_values[:,1],'tmax'].values

    boo_intersecting=(tmin_other_values<tmax_self_values) & (tmax_other_values>tmin_self_values)
    intersecting_index_pair_values=index_pair_values[boo_intersecting]
    intersecting_index_pair_values.shape

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
