import pandas as pd, numpy as np

def compute_angle_between_initial_velocities(d1,d2):
    '''computes angle between initial velocities near birth for one tip pair.
    Updates d1,d2 with fields.
    Example Usage:
    tbirth_values,angle_between_values=compute_angle_between_initial_velocities(d1,d2)
    '''
    d1[['dx','dy','dt']]=d1[['x','y','t']].diff().shift(-1).iloc[1:-1]
    d1['displacement']=np.sqrt(d1['dx']**2+d1['dy']**2)
    d1['dx_hat']=d1['dx']/d1['displacement']
    d1['dy_hat']=d1['dy']/d1['displacement']

    d2[['dx','dy','dt']]=d2[['x','y','t']].diff().shift(-1).iloc[1:-1]
    d2['displacement']=np.sqrt(d2['dx']**2+d2['dy']**2)
    d2['dx_hat']=d2['dx']/d2['displacement']
    d2['dy_hat']=d2['dy']/d2['displacement']

    cosine_series=d1['dx_hat']*d2['dx_hat']+d1['dy_hat']*d2['dy_hat']
    d1['angle_between']=np.arccos(cosine_series)   #radians
    angle_between_values=d1['angle_between'].values
    tbirth_values=d1['t'].values-d1['t'].values[0] #ms
    return tbirth_values,angle_between_values
