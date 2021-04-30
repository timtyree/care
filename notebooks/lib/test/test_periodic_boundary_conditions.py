#!/bin/bash/env python3
import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from .. import *
from ..model.minimal_model import pbc

def testme():
    pass

# test cases of periodic boundary conditions on a random matrix
test = np.random.rand(111,111,3)

# trivial tests, do nothing/ slots agree
(pbc(test,1,2)==test[1,2]).all()
assert(not (pbc(test,2,1)==test[1,2]).all())

#test each pbc boundary
assert((pbc(test,-1,2)==test[110,2]).all()) # test left
assert((pbc(test,111,2)==test[0,2]).all() ) # test right
assert((pbc(test,11,112)==test[11,0]).all() ) # test top
assert((pbc(test,12,-1)==test[12,110]).all() ) # test bottom
assert((pbc(test,-1,-1)==test[110,110]).all() ) #test bottom left corner




#padded spiral tips are produced with at pixel percision of about 13 digits.
# note that this is not the same as accuracy, which will depend on sigma, threshold, and V_threshold

# test functions for unpad
# assert(0==unpad(X=20, pad=20, width=500, rejection_distance=10))
# assert(unpad(X=19, pad=20, width=500, rejection_distance=10)==499)
# assert(280==unpad(X=300, pad=20, width=500, rejection_distance=10))
# assert(499==unpad(X=519, pad=20, width=500, rejection_distance=10))
# assert(10==unpad(X=530, pad=20, width=500, rejection_distance=10))
# assert(-9999==unpad(X=531, pad=20, width=500, rejection_distance=10))
# assert(490==unpad(X=10, pad=20, width=500, rejection_distance=10))
# assert(-9999==unpad(X=9, pad=20, width=500, rejection_distance=10))
