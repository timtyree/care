import numpy as np
from .dist_func import get_distance_L2_pbc
from numba import njit

@njit
def comp_arclen(segment_array):
    Nseg=segment_array.shape[0]
    arclen=0.
    for i in range(Nseg-1):
        arclen+=distance_L2_pbc(segment_array[i],segment_array[i+1])
    return arclen #arclength in pixels

@njit
def comp_perimeter(contour):
    Nseg=contour.shape[0]
    arclen=0.
    for i in range(-1,Nseg-1):
        arclen+=distance_L2_pbc(contour[i],contour[i+1])
    return arclen #arclength in pixels
