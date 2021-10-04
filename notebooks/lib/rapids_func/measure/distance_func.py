# import cudf
import cupy as cp
def comp_xy_distance_L2_pbc_cu(d,width,height):
    '''comp_xy_distance_L2_pbc_cu is a pure cudf function that
    returns for the euclidean (L2) distance between
    point_1 and point_2, which are in a rectangular domain of shape (width,height)
    with periodic boundary conditions.
    returns d with many fields added. the distance is dist.
    supposes each row is uniquely indexed.

    Example Usage:
    df_traj=comp_xy_distance_L2_pbc_cu(df_traj,width,height)
    '''
    #compute the three options for the square distance with pbc
    d['sdx']=(d['x_self']-d['x_other'])**2
    d['sdxp']=(d['x_self']-d['x_other']+width)**2
    d['sdxm']=(d['x_self']-d['x_other']-width)**2
    d['sdy']=(d['y_self']-d['y_other'])**2
    d['sdyp']=(d['y_self']-d['y_other']+height)**2
    d['sdym']=(d['y_self']-d['y_other']-height)**2
    #choose the minimum of each class of option
    d['minsdx']=d[['sdx','sdxp','sdxm']].min(axis=1)
    d['minsdy']=d[['sdy','sdyp','sdym']].min(axis=1)
    d['dist']=cp.sqrt(d['minsdx'] + d['minsdy'])
    # d['dist']=(d['minsdx'] + d['minsdy'])**0.5
    d['R']=d['dist']
    return d
