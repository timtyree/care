import numpy as np
from numba import njit
from ..utils.projection_func import *
# Tests for 2D vector subtraction with periodic boundary conditions enforced explicitly
# Programmer: Tim Tyree
# Date: 4.29.2021

if __name__=='__main__':
    #test cases for subtract_pbc
    subtract_pbc=get_subtract_pbc(width=200.,height=200.)
    assert( (np.array([0.,0.])==subtract_pbc(np.array([0.,2.]),np.array([0.,2.]))).all())
    assert( (np.array([0.,0.])==subtract_pbc(np.array([0.,2.]),np.array([0.,202.]))).all())
    assert( (np.array([0.,0.])==subtract_pbc(np.array([0.,2.]),np.array([0.,2-200.]))).all())
    assert( (np.array([0.,0.])==subtract_pbc(np.array([200.,2.]),np.array([0.,2-200.]))).all())
    assert (np.isclose((subtract_pbc(np.array([200.234,2.]),np.array([0.,2-200.]))-np.array([0.234, 0.   ])),0.).all())

    #test cases for project_point_2D
    project_point_2D=get_project_point_2D(width=200.,height=200.)
    point=np.array(    (129,180.))
    segment=np.array([[129-200, 180.        ],
                       [129, 381.        ]])
    frac=project_point_2D(point,segment)
    assert(np.isclose(frac,0))
    point=np.array(    (129,180.7))
    frac=project_point_2D(point,segment)
    assert(np.isclose(frac,0.7))
    point=np.array(    (129,181.))
    frac=project_point_2D(point,segment)
    assert(np.isclose(frac,1.0))
    point=np.array(    (129,381.))
    frac=project_point_2D(point,segment)
    assert(np.isclose(frac,1.0))
    point=np.array(    (129-200,381.))
    frac=project_point_2D(point,segment)
    assert(np.isclose(frac,1.0))
    print(f"all test cases passed for lib.utils.projection_func.py!")
