#!/bin/env python3
import numpy as np
from numba import jit, njit
"""
Give, two x,y curves this gives intersection points,
forked on May 24.2020
forked from: https://github.com/sukhbinder/intersection.git
Based on: http://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections
"""
##TODO: make sure that all contours are properly considered
# def get_tips(contours_raw,contours_inc):
#     x1, y1 = (contours_raw[0][:,0], contours_raw[0][:,1])
#     x2, y2 = (contours_inc[0][:,0], contours_inc[0][:,1])
#     return intersection(x1,y1,x2,y2)
# @jit
def _rect_inter_inner(x1, x2):
    # assert(type(x1)==np.ndarray)
    # assert(type(x2)==np.ndarray)
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.asarray((x1[:-1], x1[1:])).T#np.c_[x1[:-1], x1[1:]]#
    X2 = np.asarray((x2[:-1], x2[1:])).T#np.concatenate((x2[:-1], x2[1:]),axis=-1)
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T#np.asarray(np.min(list(X1)))
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4
# @njit
def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj

# @njit
def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.

usage:
x,y=intersection(x1,y1,x2,y2)

    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)

    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # a piece of a prolate cycloid, and am going to find
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2 = phi
    y2 = np.sin(phi)+2
    x, y = intersection(x1, y1, x2, y2)
    plt.plot(x1, y1, c='r')
    plt.plot(x2, y2, c='g')
    plt.plot(x, y, '*k')
    plt.show()
