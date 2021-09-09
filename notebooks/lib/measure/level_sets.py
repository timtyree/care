import numpy as np, pandas as pd
from sklearn import linear_model
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator

# try:
from .smooth_array import *
from .interpolate import interpolate_txt_to_contour
from .intersection import intersection
from ._find_contours import find_contours
from ..routines.interp_texture_from_scatter_data import interp_txt_from_scatter
# except ModuleNotFoundError or ImportError as e:
#     try:
#         from ...lib_care.measure._find_contours import find_contours
#     except ImportError as e:
#         pass

#TODO: These were copied to lib.measure... and lib_care.measure and care/lib.measure
def comp_largest_level_set(array,level,txt):
    #extract the contour in pixel coordinates
    contours=find_contours(array=array, level=level, fully_connected='low', positive_orientation='low', mode='hard_boundary')#, *, mask=None)
    num_contours=len(contours)
    assert(num_contours>0)
    minsize=0
    for contour_any in contours:
        if contour_any.shape[0]>minsize:
            minsize=contour_any.shape[0]
            contour=contour_any
    # plt.plot(contour[:,1],contour[:,0])

    #map pixel coordinates to input coordinates using bilinear interpolation
    #doubling kwargs width and height effectively nullifies the periodic boundary conditions used in interpolate_txt_to_contour by default
    width=2*array.shape[0]
    height=2*array.shape[1]
    contour_values=np.array(interpolate_txt_to_contour(contour,width,height,txt=txt))[1:]#[1:] fixes bug from periodic boundary conditions
    return contour_values


def comp_longest_level_set_and_smooth(X,y,level,navg=20):
    '''returns the navg-moving average xy coordinates of the level-set
    curve where y(X)=level that has the most segments.
    - X is an N by 2 numpy array
    - y is an N by 1 numpy array
    Example Usage:
    output_col='m'; model_name='lr_pbc'
    level=wjr[model_name][output_col]
    y=df.loc[query,output_col].values
    smoothed_contour_values=comp_longest_level_set(X,y,level,navg=20)
    '''


    x1_values,x2_values,y_values=interp_txt_from_scatter(X,y)
    txt=np.stack((x1_values,x2_values)).T

    #TODO(later, add support for measuring M, Delta_M along the m-level set): concat txt from all output arrays of interest at once
    #- compute the y_values for each target output variable
    # - stack y_value_lst at the end as a txt
    #HINT: modify to use only first 2 axis XI_lst=comp_meshgrid_from_X(X) where X is N by D, for D channels you want the value for along the contour
    # ###### MemoryError: Unable to allocate 6.94 EiB for an array with shape (1000, 1000, 1000, 1000, 1000, 1000) and data type float64
    # #interpolate any desired fields to txt
    # field_txt_col_lst=[x1_col,x2_col,'m','Delta_m','M','Delta_M']
    # X=df.loc[query,field_txt_col_lst].values
    # XI_lst=comp_meshgrid_from_X(X)
    # txt=np.stack(XI_lst).T

    #interpolate the smoothed contour values
    array=y_values
    contour_values=comp_largest_level_set(array,level,txt)
    x_smoothed,y_smoothed=smooth_contour_xy(contour_values,navg=navg)
    smoothed_contour_values=np.stack((x_smoothed,y_smoothed)).T
    #re: ###### MemoryError: Unable to allocate 6.94 EiB
    # smoothed_contour_value_lst=smooth_contour(contour_values,navg=20)
    return smoothed_contour_values


def compute_intersections(contour_A_values,contour_B_values):
    '''
    Example Usage:
    x1star_values, x2star_values=compute_intersections(contour_A_values,contour_B_values)
    '''
    contour_m_values=contour_A_values
    contour_M_values=contour_B_values
    x1=contour_m_values[:,0]
    y1=contour_m_values[:,1]
    x2=contour_M_values[:,0]
    y2=contour_M_values[:,1]
    x1star_values, x2star_values=intersection(x1, y1, x2, y2)
    return x1star_values, x2star_values

def compute_self_consistent_astar_rstar(contour_m_values,contour_M_values):
    '''
    Example Usage:
    rstar,astar=compute_self_consistent_astar_rstar(contour_m_values,contour_M_values)
    '''
    #compute the intersection point as rstar,astar
    x1star_values, x2star_values=compute_intersections(contour_m_values,contour_M_values)
    assert (x1star_values.shape[0]==1)
    assert (x2star_values.shape[0]==1)
    rstar=x1star_values[0]
    astar=x2star_values[0]
    return rstar,astar
    print(f"the one intersection found is centered at rstar={rstar:.5f} cm and astar={astar:.5f} cm^2/s.")
