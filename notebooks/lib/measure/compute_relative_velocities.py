import pandas as pd, numpy as np
from ..utils.projection_func import get_subtract_pbc

def compute_DT(df,round_t_to_n_digits=3):
    '''DT is the time between two observations'''
    DT=np.around(df[(df.frame==1)].t.values[0]-df[(df.frame==0)].t.values[0],round_t_to_n_digits)
    return DT

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

def get_compute_angle_between_final_velocities(width,height):
    # compute_displacements_between=get_compute_displacements_between(width,height)
    subtract_pbc=get_subtract_pbc(width=width,height=height)
    def compute_angle_between_final_velocities(d1,d2):
        '''computes angle between final velocities near birth for one tip pair.
        Updates d1,d2 with fields.  aligns locations by index
        Example Usage:
        compute_angle_between_final_velocities=get_compute_angle_between_final_velocities(width,height)
        tdeath_values,angle_between_values=compute_angle_between_final_velocities(d1,d2)
        '''
        #compute displacement of d1 with pbc
        xy_values=np.array(list(zip(d1['x'],d1['y'])))
        dshifted=d1.shift(1).copy()
        # dshifted=d1.shift(-1).copy()
        xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
        dxy1_values=np.zeros_like(xy_values)+np.nan
        # compute displacement unit vector from tip 1 to tip 2
        xy_values=np.array(list(zip(d1['x'],d1['y'])))
        # dshifted=d1.shift(1).copy()
        dshifted=d1.shift(-1).copy()
        xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
        dxy1_values=np.zeros_like(xy_values)+np.nan
        #compute displacements between
        for j in range(dxy1_values.shape[0]):
            dxy1_values[j]=subtract_pbc(xy_next_values[j],xy_values[j])
        d1['dx']=dxy1_values[:,0]
        d1['dy']=dxy1_values[:,1]
        d1['dt']=d1['t'].diff().shift(-1).iloc[1:-1]
        # d1[['dx','dy','dt']]=d1[['x','y','t']].diff().shift(-1).iloc[1:-1]
        d1['displacement']=np.sqrt(d1['dx']**2+d1['dy']**2)
        d1['dx_hat']=d1['dx']/d1['displacement']
        d1['dy_hat']=d1['dy']/d1['displacement']

        #compute displacement of d2 with pbc
        xy_values=np.array(list(zip(d2['x'],d2['y'])))
        # dshifted=d2.shift(1).copy()
        dshifted=d2.shift(-1).copy()
        xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
        dxy2_values=np.zeros_like(xy_values)+np.nan
        # compute displacement unit vector from tip 1 to tip 2
        xy_values=np.array(list(zip(d2['x'],d2['y'])))
        dshifted=d2.shift(1).copy()
        # dshifted=d1.shift(-1).copy()
        xy_next_values=np.array(list(zip(dshifted['x'],dshifted['y'])))
        dxy2_values=np.zeros_like(xy_values)+np.nan
        #compute displacements between
        for j in range(dxy2_values.shape[0]):
            dxy2_values[j]=subtract_pbc(xy_next_values[j],xy_values[j])
        d2['dx']=dxy2_values[:,0]
        d2['dy']=dxy2_values[:,1]
        d2['dt']=d2['t'].diff().shift(-1).iloc[1:-1]
        d2[['dx','dy','dt']]=d2[['x','y','t']].diff().shift(-1).iloc[1:-1]
        d2['displacement']=np.sqrt(d2['dx']**2+d2['dy']**2)
        d2['dx_hat']=d2['dx']/d2['displacement']
        d2['dy_hat']=d2['dy']/d2['displacement']


        # compute dot product between tip 1 and tip 2
        cosine_series=d1['dx_hat']*d2['dx_hat']+d1['dy_hat']*d2['dy_hat']
        d1['angle_between']=np.arccos(cosine_series)   #radians

        angle_between_values=d1['angle_between'].values
        tdeath_values=d1['t'].values[-1]-d1['t'].values #ms
        # # limit the values of tdeath to d1 or d2 depending on who is shorter
        # tdeath2_values=d2['t'].values[-1]-d2['t'].values
        # t1_min=np.min(tdeath_values)
        # t2_min=np.min(tdeath_values)
        # t_min=np.min((t1_min,t2_min))
        # boo=tdeath_values>=t_min
        # tdeath_values=tdeath_values[boo]
        # angle_between_values=angle_between_values[boo]
        d1.dropna(inplace=True)
        return tdeath_values,angle_between_values
    return compute_angle_between_final_velocities
