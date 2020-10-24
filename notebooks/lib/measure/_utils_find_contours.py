import numpy as np

#boolean functions for continuity.  helps with periodic functions
def _get_contiguous_mask(contour, threshold=2):
    '''returns boolean array based on proximity according to Euclidean distance.
    an entry is False if the contour jumps accross a boundary.'''
    return np.linalg.norm(np.diff(contour, axis=0), axis=-1)<threshold

def get_contiguous_mask(contour, threshold=2):
    return _get_contiguous_mask(contour, threshold=threshold)

def _is_contour_contiguous(contour, threshold=2):
    '''returns True is the contour makes no jumps accross the boundary.'''
    mask = _get_contiguous_mask(contour, threshold=threshold)
    return mask.all()
def _are_ends_contiguous(contour):
    '''returns whether the first and final contour points are contiguous.'''
    return (contour[0] == contour[-1]).all()

def _split_contour_into_contiguous(contour, threshold=2):
    '''returns a list of contiguous contours. 
    threshold = the max number of pixels two points may be separated by to be considered contiguous.
    split the contour into a contour_lst of contiguous contours.
    the last contour may have length 1 or 0.  simply ignore those later.'''
    mask = _get_contiguous_mask(contour, threshold=threshold)
    indices = np.nonzero(~mask)[0]+1
    contour_lst = np.split(contour,indices)
    return contour_lst

def split_contour_into_contiguous(contour, threshold=2):
    return _split_contour_into_contiguous(contour, threshold=threshold)