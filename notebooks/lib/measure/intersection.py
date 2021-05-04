#!/bin/env python3
import numpy as np
from numba import njit, prange
from numba.typed import List
"""
Give, two x,y curves this gives intersection points,
forked on May 24.2020
forked from: https://github.com/sukhbinder/intersection.git
Based on: http://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections
njistu by Tim Tyree. just in time compilation via numba.njit --> first run with new data takes ~5 seconds.  later runs are very fast (<50ms).
"""

@njit#(fastmath=True, parallel=True)
def _min_within(x,n):
    '''compare every two entries and return the min of the two.
    n = int(x.shape[0] - 1)'''
    out = np.zeros(n, dtype=np.float64)
    for j in prange(n):
        out[j] = min((x[j], x[j+1]))
    return out
@njit#(fastmath=True, parallel=True)
def _max_within(x,n):
    '''compare every two entries and return the max of the two.
    n = int(x.shape[0] - 1)'''
    out = np.zeros(n, dtype=np.float64)
    for j in prange(n):
        out[j] = max((x[j], x[j+1]))
    return out

@njit#(fastmath=True, parallel=True)
def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1

    minx1 = _min_within(x1,n1)
    maxx1 = _max_within(x1,n1)
    minx2 = _min_within(x2,n2)
    maxx2 = _max_within(x2,n2)

    S1 = np.zeros((n2,n1))
    S3 = S1.copy()
    for j in prange(n2):
        S1[j] = minx1
        S3[j] = maxx1
    S2 = np.zeros((n1,n2))
    S4 = S2.copy()
    for j in prange(n1):
        S2[j] = maxx2
        S4[j] = minx2
    return S1.T, S2, S3.T, S4

@njit#(fastmath=True, parallel=True)
def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj

@njit#(fastmath=True, parallel=True)
def _intersection_after_r_i_(x1, y1, x2, y2,ii,jj):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   cannot be broken with NaNs but should be able to have vertical segments.

Fails when lines intersect when parallel to machine precision.

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
    # x1 = np.asarray(x1)
    # x2 = np.asarray(x2)
    # y1 = np.asarray(y1)
    # y2 = np.asarray(y2)

    n = len(ii)  #number of contours ** 2, return 'no contours' if n==0
    # if n==0:
    #     return np.array([]),np.array([])

    dxy1 = np.column_stack((x1, y1))
    dxy1 = dxy1[1:] - dxy1[:-1]
    dxy2 = np.column_stack((x2, y2))
    dxy2 = dxy2[1:] - dxy2[:-1]

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

    for i in prange(n):
        T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]

@njit()#fastmath=True) #fastmath might lower precision...
def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   cannot be broken with NaNs but should be able to have vertical segments.

Fails when lines intersect when parallel to machine precision.

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
    # x1 = np.asarray(x1)
    # x2 = np.asarray(x2)
    # y1 = np.asarray(y1)
    # y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)  #number of contours ** 2, return 'no contours' if n==0
    # if n==0:
    #     return np.array([]),np.array([])

    dxy1 = np.column_stack((x1, y1))
    dxy1 = dxy1[1:] - dxy1[:-1]
    dxy2 = np.column_stack((x2, y2))
    dxy2 = dxy2[1:] - dxy2[:-1]

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

    for i in prange(n):
        T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])

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
