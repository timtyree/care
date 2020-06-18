#!/bin/bash/env python3
from lib import *

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