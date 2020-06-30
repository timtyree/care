# Module of functions for computing the distance between points.  Periodi
# Tim Tyree
# 6.29.2020

import numpy as np
from numba import njit, jit, float32

#############################################################
# Example Usage: L2 Distance using Periodic Boundary Conditions
# distance_L2_pbc = get_distance_L2_pbc(width=200,height=200)
# distance_L2_pbc(np.array([1.,1.]),np.array([199.,199.]))
#############################################################

@njit
def pbc(x,y, width, height):
	'''
	(x, y) coordinates that go from 0 to width or height, respectively.
	tight boundary rounding is in use.'''
	if ( x < 0  ):				# // Left P.B.C.
		x = width - 1
	elif ( x > (width - 1) ):	# // Right P.B.C.
		x = 0
	if( y < 0 ):				# //  Bottom P.B.C.
		y = height - 1
	elif ( y > (height - 1)):	# // Top P.B.C.
		y = 0
	return x,y

def get_distance_L2_pbc(width=200,height=200):
    '''returns a function for the euclidean (L2) distance measure for a 2D rectangle with periodic boundary conditions.
    width, height are the shape of that 2D rectangle.'''
    @jit('f8(f8[:],f8[:])', nopython=True)
    def distance_L2_pbc(point_1, point_2):
        '''assumes getting shortest distance between two points with periodic boundary conditions in 2D.'''
        mesh_shape=np.array((width,height))
        dq2 = 0.
        #     for q1, q2, width in zip(point_1[:2], point_2[:2], mesh_shape):
        for q1, q2, wid in zip(point_1, point_2, mesh_shape):
            dq2 = dq2 + min(((q2 - q1)**2, (q2 + wid - q1 )**2, (q2 - wid - q1 )**2))
        return np.sqrt(dq2)
    return distance_L2_pbc

def test_get_distance_L2_pbc():
    import trackpy, pandas as pd

    #testing the pbc distance function
    distance_L2_pbc = get_distance_L2_pbc(width=200,height=200)
    assert(np.isclose(distance_L2_pbc(np.array([1.,1.]),np.array([1.,1.])),0.))
    assert(distance_L2_pbc(np.array([1.,1.]),np.array([199.,199.]))<3.)
    assert(distance_L2_pbc(np.array([1.,1.]),np.array([199.,199.]))<3.)
    assert(distance_L2_pbc(np.array([1.,1.]),np.array([1.,199.]))<3.)
    assert(distance_L2_pbc(np.array([1.,1.]),np.array([199.,1.]))<3.)

    #testing distance_L2_pbc in trackpy.link_df
    df_test = pd.DataFrame({'frame':[1,2,2],'x':[1,100,199],'y':[1,100,199]})

    # test that the default distance function maps 1,1 to 100,100 instead of 199,199
    traj_test = trackpy.link_df(df_test, search_range=210)
    assert((traj_test['particle'].values == np.array([0, 0, 1])).all())

    # test that the pbc distance function maps 1,1 to 199,199 instead of 100,100
    traj_test = trackpy.link_df(df_test, search_range=210, dist_func = distance_L2_pbc)
    assert((traj_test['particle'].values == np.array([0, 1, 0])).all())

    # test that the pbc distance function can see accross boundarys when search_range is small
    traj_test = trackpy.link_df(df_test, search_range=110, dist_func = distance_L2_pbc)
    assert((traj_test['particle'].values == np.array([0, 1, 0])).all())
    return True

# #@njit
# def _get_distance_ND_cartesian_pbc(point_1, point_2, mesh_shape):
# 	'''assumes getting shortest distance between two points with periodic boundary conditions '''
# 	dq2 = 0.
# 	for q1, q2, width in zip(point_1, point_2, mesh_shape):
# 		dq2 += np.min(((q2 - q1)**2, (q2 + width - q1 )**2))
# 	return np.sqrt(dq2)

# #@njit
# def _get_distance_ND_cartesian(point_1, point_2, mesh_shape):
# 	'''assumes getting shortest distance between two points with periodic boundary conditions '''
# 	dq2 = 0.
# 	for q1, q2, width in zip(point_1, point_2, mesh_shape):
# 		dq2 += (q2 - q1)**2
# 	return np.sqrt(dq2)

# #@njit
# def get_distance(point_1, point_2):
# 	'''return the 2D distance between two points using periodic boundary conditions'''
# 	return _get_distance_2D_pbc(point1, point_2)
