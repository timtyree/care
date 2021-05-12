import pandas as pd, numpy as np

def compute_phase_angles_from_grad_voltage(d1,d2):
    #compute the displacement vector between these spiral tips
    d1['x2']=d2['x']
    d1['y2']=d2['y']
    d1['dx']=d1['x2']-d1['x']
    d1['dy']=d1['y2']-d1['y']
    d1['range']=np.sqrt(d1['dx']**2+d1['dy']**2)
    d1['dxhat']=d1['dx']/d1['range']
    d1['dyhat']=d1['dy']/d1['range']
    d2['dxhat']=d1['dxhat']
    d2['dyhat']=d1['dyhat']
    d1['grad_u_mag']=np.sqrt(d1['grad_ux']**2+d1['grad_uy']**2)
    d2['grad_u_mag']=np.sqrt(d2['grad_ux']**2+d2['grad_uy']**2)

#     #cosine method is equivalent up to sign
#     # - using cross product of dx,y_hat with +-ahat
#     #TODO: - using  dot  product of dx,y_hat with +,-grad_u
#     #compute the min/max absolute phase angles
#     cosine_values=(d1['dxhat']*d1['grad_ux']+d1['dyhat']*d1['grad_uy'])/d1['grad_u_mag']
#     d1['phi1']=np.arcsin(cosine_values)
#     cosine_values=(d2['dxhat']*d2['grad_ux']+d2['dyhat']*d2['grad_uy'])/d2['grad_u_mag']
#     d2['phi2']=np.arcsin(cosine_values)
#     boo=~d1.phi1.isnull()
#     t1_values=d1[boo]['t'].values
#     boo=~d2.phi2.isnull()
#     t2_values=d2[boo]['t'].values

    d1['a1x']=d1['grad_uy']/d1['grad_u_mag']
    d1['a1y']=-d1['grad_ux']/d1['grad_u_mag']
    sine_values_left=d1['a1x']*d1['dyhat']-d1['a1y']*d1['dxhat']
    sine_values_right=-d1['a1x']*d1['dyhat']+d1['a1y']*d1['dxhat']

    if np.arcsin(sine_values_left).values[-1]>0:
        # choose self-consistent convention that ends in giving phi a positive phase value
        d1['phi1']=np.arcsin(sine_values_left)
    else:
        d1['phi1']=np.arcsin(sine_values_right)

    d2['ax']=d2['grad_uy']/d2['grad_u_mag']
    d2['ay']=-d2['grad_ux']/d2['grad_u_mag']
    sine_values_left=d2['ax']*d2['dyhat']-d2['ay']*d2['dxhat']
    sine_values_right=-d2['ax']*d2['dyhat']+d2['ay']*d2['dxhat']

    if np.arcsin(sine_values_left).values[-1]<0:
        # choose self-consistent convention that ends in giving phi a negative phase value
        d2['phi2']=np.arcsin(sine_values_left)
    else:
        d2['phi2']=np.arcsin(sine_values_right)

    boo=~d1.phi1.isnull()
    t1_values=d1[boo]['t'].values
    x_values=t1_values[-1]-t1_values
    phi1_values=d1[boo]['phi1'].values

    boo=~d2.phi2.isnull()
    t2_values=d2[boo]['t'].values
    phi2_values=d2[boo]['phi2'].values

    phi_sum_values=phi1_values+phi2_values
    phi_diff_values=phi1_values-phi2_values
    t_to_death_values=x_values
    return t_to_death_values, phi1_values, phi2_values, phi_sum_values, phi_diff_values
