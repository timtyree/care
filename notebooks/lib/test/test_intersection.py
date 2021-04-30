#!/bin/env python3
import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import numpy as np
from ..measure.intersection import intersection

def testme():
    pass

#test cases
test = True
a, b = 1, 2
phi = np.linspace(3, 10, 100)
x1 = a*phi - b*np.sin(phi)
y1 = a - b*np.cos(phi)

x2=phi
y2=np.sin(phi)+2
x,y=intersection(x1,y1,x2,y2)
assert(np.array(x).shape[0]>0)
assert(np.array(y).shape[0]>0)

#visualize the test
# plt.plot(x1,y1,c='r')
# plt.plot(x2,y2,c='g')
# plt.plot(x,y,'*k')
# plt.show()
