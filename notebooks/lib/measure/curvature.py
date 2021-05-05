import numpy as np

# TODO: unwrap array
def unwrap_contour(x_values,y_values,width,height):
    x_values = x_values.astype('float64')
    y_values = y_values.astype('float64')
    jump_index_array, spd_lst = find_jumps(x_values,y_values,width,height,DS=1.,DT=1.)#,**kwargs)
    # find_jumps(x_values,y_values,DS=DS,DT=DT)
    xv,yv = unwrap_for_each_jump(x_values,y_values,jump_index_array, width=width,height=height)

    #subtract off the initial position for plotting's sake
    xv -= xv[0]
    yv -= yv[0]
    return xv,yv



# TODO(for modeling the curvature dynamics...)
# TODO: get location values of an activation front identified via voltage>V_threshold values or dVcdt_avg>0
def compute_curvature(array):
    '''array is a numpy array of size N-by-2 that indicates
    a continuous curve embedded in the extended real plane.
    returns a dict of (signed?) curvature values (with other geometric values)
    Example Usage:
    dict_curvature=compute_curvature(array=a)
    sigma_unitless_values=np.linspace(0,1,curvature.shape[0])
    curvature_values=dict_curvature['curvature']
    plt.plot(sigma_unitless_values,curvature_values)
    plt.xlabel('position')
    plt.ylabel('curvature')
    plt.show()
    '''
    a=array
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1/ds_dt] * 2).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

    normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt

    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    t_component = np.array([d2s_dt2] * 2).transpose()
    n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()
    acceleration = t_component * tangent + n_component * normal
    dict_curvature={'normal':normal,
                   'curvature':curvature,
                    'n_component':n_component,
                    't_component':t_component,
                    'acceleration':acceleration,
                   }
    return dict_curvature
