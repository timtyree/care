import numpy as np
from ..utils.utils_traj import find_jumps, unwrap_for_each_jump
from scipy.interpolate import splprep, splev
from scipy import interpolate

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

def comp_curvature(xy_values,s=2):
    '''s is a smoothing parameter. s=0 forces agreement with xy_values, but does not produce smooth curvature
    xy_values is an Nx2 np.array discretizing a continuous curve
    Note that (i) we force interpolation by using s=0,
    (ii) the parameterization, u, is generated automatically.
    Example Usage:
    curvature_values=comp_curvature(xy_values)
    '''
    # x=xy_values[:,0]
    # y=xy_values[:,1]

    xp=xy_values[:,0]
    yp=xy_values[:,1]
    assert (xy_values.shape[1]==2)
    okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)
    x = np.r_[xp[okay], xp[-1], xp[0]]
    y = np.r_[yp[okay], yp[-1], yp[0]]
    # print(x.shape)#error has shape (430,)
    # print(y.shape)#error has shape (430,)
    # try:
    # xyavg=xy_values[:-1]/2+xy_values[1:]/2
    # tck, u = splprep(xy_values.T,s=s)#s=0)
    tck, u = splprep([x, y],s=s)#,per=1,nest=1000,task=0,k=3)#s=0)
    # except Exception as e:
    #     print("ERROR:")
    #     print(e)
    #     print(xy_values)
    #
    #
    #     # print(x.shape)
    #     # print(y.shape)
    # print(np.array(x))
    # print(np.array(y))
    new_points = splev(u, tck)
    dxds,dyds = splev(u, tck, der=1)
    dx2ds2,dy2ds2 = splev(u, tck, der=2)
    curvature_values = np.abs(dx2ds2 * dyds - dxds * dy2ds2) / (dxds * dxds + dyds * dyds)**1.5
    return curvature_values

def comp_interpolated_points(xy_values,s=2):
    '''s is a smoothing parameter. s=0 forces agreement with xy_values, but does not produce smooth curvature
    xy_values is an Nx2 np.array discretizing a continuous curve
    Note that (i) we force interpolation by using s=0,
    (ii) the parameterization, u, is generated automatically.
    Example Usage:
    new_points=comp_interpolated_points(xy_values)
    '''
    xp=xy_values[:,0]
    yp=xy_values[:,1]
    assert (xy_values.shape[1]==2)
    okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)
    x = np.r_[xp[okay], xp[-1], xp[0]]
    y = np.r_[yp[okay], yp[-1], yp[0]]
    tck, u = splprep([x, y],s=s)#,per=1)#s=0)
    new_points = splev(u, tck)
    return new_points
