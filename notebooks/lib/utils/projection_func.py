import numpy as np
from numba import njit
# 2D vector subtraction with periodic boundary conditions enforced explicitly
# Programmer: Tim Tyree
# Date: 4.29.2021

@njit
def __diff_pbc_1d(diff,width):
    dmin=np.abs(diff)
    dtmp=diff+width
    if np.abs(dtmp)<dmin:
        return dtmp
    else:
        dtmp=diff-width
        if np.abs(dtmp)<dmin:
            return dtmp
        else:
            return diff

def get_subtract_pbc(width=200.,height=200.):
    '''subtract to get to local coords'''
    @njit
    def subtract_pbc(point1,point2):
        '''tries all pbc combinations and returns the vector difference corresponding to the smallest magnitude.'''
        diff=point1-point2
        xdiff=__diff_pbc_1d(diff[0],width)
        ydiff=__diff_pbc_1d(diff[1],height)
        return np.array((xdiff,ydiff))
    return subtract_pbc


def get_project_point_2D(width=200.,height=200.):
    '''returns project_point_2D
    Obeys periodic boundary conditions on a width-x-height square domain.

    returns the fraction of the length of segment where the projection of point onto segment lies.
    returns 0. when point=segment[0].
    returns a value in the interval [0,1) if this point lies approximately within this segment
    Example Usage:
    project_point_2D=get_project_point_2D(width=200,height=200)
    frac=project_point_2D(point,segment)
    '''
    subtract_pbc=get_subtract_pbc(width=width,height=height)
    @njit
    def project_point_2D(point,segment):
        '''Obeys periodic boundary conditions on a width-x-height square domain.

        returns the fraction of the length of segment where the projection of point onto segment lies.
        returns 0. when point=segment[0].
        returns a value in the interval [0,1) if this point lies approximately within this segment
        Example Usage:
        project_point_2D=get_project_point_2D(width=200,height=200)
        frac=project_point_2D(point,segment)
        '''
        w=subtract_pbc(point,segment[0])
        u=subtract_pbc(segment[1],segment[0])
        c=np.dot(w,u)
        l=np.sum(u**2)
        frac=c/l
        return frac
    return project_point_2D
