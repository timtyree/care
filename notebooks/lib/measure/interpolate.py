# Module for interpolating numpy arrays of shape (width,height) to 
# subpixel values with periodic boundary conditions
#Timothy Tyree
#1.13.2021
import numpy as np
from numba import njit

#for measurement
@njit
def interpolate_img(x_values,y_values,width,height,img):
    '''Measure color of txt at zip(x_values,y_values).  This was used to measure EP states.
    Example Usage:
    v_lst, f_lst, s_lst = interpolate_states(x_values,y_values,width,height,txt)
    '''
    states_bilinear_lst = []
    for x,y in zip(x_values,y_values):
        states_bilinear = bilinear_interpolate_channel(x,y,width,height,img)
        states_bilinear_lst.append(states_bilinear)
    return states_bilinear_lst

@njit
def ___pbc_1d(x_in,width):
    if np.greater_equal(x_in , width):
        return int(x_in-width)
    elif np.less(x_in, 0):
        return int(x_in+width)
    return int(x_in)

@njit
def bilinear_interpolate_channel(x,y,width,height,img):
    r0 = np.floor(y)
    r1 = np.ceil(y)
    frac_r = y-r0
    if r0==r1:
        r1+=1
    c0 = np.floor(x)
    c1 = np.ceil(x)
    frac_c = x-c0
    if c0==c1:
        c1+=1
    c0 = ___pbc_1d(c0,width)
    c1 = ___pbc_1d(c1,width)
    r0 = ___pbc_1d(r0,height)
    r1 = ___pbc_1d(r1,height)
    a11 = img[r0,c0]
    a21 = img[r1,c0]-a11
    a12 = img[r0,c1]-a11
    a22 = img[r1,c1]-a11-a12-a21
    states_bilinear = a11 + a21*frac_c + a12*frac_r + a22*frac_c*frac_r
    return states_bilinear