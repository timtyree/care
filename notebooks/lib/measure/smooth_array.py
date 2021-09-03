import scipy,numpy as np,pandas as pd

#TODO: move to lib.measure.smooth_array.py
def spline_ysmooth(x,y,der=0,s=0):
    tck  = scipy.interpolate.splrep(x, y, s=s)
    ynew = scipy.interpolate.splev(x, tck, der=der)
    return ynew

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#computation of level set curves set to powerlaw fits from the full models
def smooth_contour_xy(contour_values,navg=20,xcol=0,ycol=1):
    '''returns contour smoothed by a simple moving average with a moving window length of navg nodes/values.
    uses <function lib_care.measure.compute_forces_at_annihilation.moving_average(x, w)>, which should be moved to lib_care.measure.moving_average.py
    Example Usage:
    x_smoothed,y_smoothed=smooth_contour_xy(contour_values,w=20)
    '''
    x=contour_values[:,xcol]
    y=contour_values[:,ycol]
    x_smoothed=moving_average(x,w=navg)
    y_smoothed=moving_average(y,w=navg)
    return x_smoothed,y_smoothed

def smooth_contour(contour_values,navg=20):
    '''returns contour smoothed by a simple moving average with a moving window length of navg nodes/values.
    uses <function lib_care.measure.compute_forces_at_annihilation.moving_average(x, w)>, which should be moved to lib_care.measure.moving_average.py
    Example Usage:
    smoothed_contour_value_lst=smooth_contour(contour_values,navg=20)
    '''
    smoothed_contour_value_lst=[]
    for col in range(contour_values.shape[1]):
        x=contour_values[:,col]
        x_smoothed=moving_average(x,w=navg)
        smoothed_contour_value_lst.append(x_smoothed)
    return smoothed_contour_value_lst
