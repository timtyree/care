import numpy as np
from .. import *
from ..measure._utils_find_contours import _split_contour_into_contiguous, _is_contour_contiguous
from ..measure import find_contours as fc


def testme():
    pass

#test that split_contour_into_contiguous returns only contours that are contiguous
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
contours = fc(r, level=.5, mode='pbc')

for contour in contours:
    contour_lst = _split_contour_into_contiguous(contour)
    for ct in contour_lst:
        assert(_is_contour_contiguous(ct))

# #informal print tests for the above boolean functions
# print('# discontinuities, is contiguous?, are ends equal?')
# for n, contour in enumerate(contours):
#     print(f"for contour #{n}:")
#     mask = ~_get_contiguous_mask(contour, threshold=2)
#     print('\t',sum(mask), _is_contour_contiguous(contour, threshold=2),_are_ends_contiguous(contour))
