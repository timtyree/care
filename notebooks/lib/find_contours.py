#!/bin/bash/env python3
#The following modifiable code was originally returned by the following three pythonic commands
# *flexing* Tim Tyree 6.8.2020
# import inspect, skimage.measure as measure
# ss = inspect.getsource(measure.find_contours)
# [print(l) for l in ss.splitlines()]

def find_contours(array, level,
                  fully_connected='low', positive_orientation='low'):
    """Find iso-valued contours in a 2D array for a given level value.

    Uses the "marching squares" method to compute a the iso-valued contours of
    the input 2D array for a particular level value. Array values are linearly
    interpolated to provide better precision for the output contours.

    Parameters
    ----------
    array : 2D ndarray of double
        Input data in which to find contours.
    level : float
        Value along which to find contours in the array.
    fully_connected : str, {'low', 'high'}
         Indicates whether array elements below the given level value are to be
         considered fully-connected (and hence elements above the value will
         only be face connected), or vice-versa. (See notes below for details.)
    positive_orientation : either 'low' or 'high'
         Indicates whether the output contours will produce positively-oriented
         polygons around islands of low- or high-valued elements. If 'low' then
         contours will wind counter- clockwise around elements below the
         iso-value. Alternately, this means that low-valued elements are always
         on the left of the contour. (See below for details.)

    Returns
    -------
    contours : list of (n,2)-ndarrays
        Each contour is an ndarray of shape ``(n, 2)``,
        consisting of n ``(row, column)`` coordinates along the contour.

    Notes
    -----
    The marching squares algorithm is a special case of the marching cubes
    algorithm [1]_.  A simple explanation is available here::

      http://www.essi.fr/~lingrand/MarchingCubes/algo.html

    There is a single ambiguous case in the marching squares algorithm: when
    a given ``2 x 2``-element square has two high-valued and two low-valued
    elements, each pair diagonally adjacent. (Where high- and low-valued is
    with respect to the contour value sought.) In this case, either the
    high-valued elements can be 'connected together' via a thin isthmus that
    separates the low-valued elements, or vice-versa. When elements are
    connected together across a diagonal, they are considered 'fully
    connected' (also known as 'face+vertex-connected' or '8-connected'). Only
    high-valued or low-valued elements can be fully-connected, the other set
    will be considered as 'face-connected' or '4-connected'. By default,
    low-valued elements are considered fully-connected; this can be altered
    with the 'fully_connected' parameter.

    Output contours are not guaranteed to be closed: contours which intersect
    the array edge will be left open. All other contours will be closed. (The
    closed-ness of a contours can be tested by checking whether the beginning
    point is the same as the end point.)

    Contours are oriented. By default, array values lower than the contour
    value are to the left of the contour and values greater than the contour
    value are to the right. This means that contours will wind
    counter-clockwise (i.e. in 'positive orientation') around islands of
    low-valued pixels. This behavior can be altered with the
    'positive_orientation' parameter.

    The order of the contours in the output list is determined by the position
    of the smallest ``x,y`` (in lexicographical order) coordinate in the
    contour.  This is a side-effect of how the input array is traversed, but
    can be relied upon.

    .. warning::

       Array coordinates/values are assumed to refer to the *center* of the
       array element. Take a simple example input: ``[0, 1]``. The interpolated
       position of 0.5 in this array is midway between the 0-element (at
       ``x=0``) and the 1-element (at ``x=1``), and thus would fall at
       ``x=0.5``.

    This means that to find reasonable contours, it is best to find contours
    midway between the expected "light" and "dark" values. In particular,
    given a binarized array, *do not* choose to find contours at the low or
    high value of the array. This will often yield degenerate contours,
    especially around structures that are a single array element wide. Instead
    choose a middle value, as above.

    References
    ----------
    .. [1] Lorensen, William and Harvey E. Cline. Marching Cubes: A High
           Resolution 3D Surface Construction Algorithm. Computer Graphics
           (SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).

    Examples
    --------
    >>> a = np.zeros((3, 3))
    >>> a[0, 0] = 1
    >>> a
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> find_contours(a, 0.5)
    [array([[ 0. ,  0.5],
           [ 0.5,  0. ]])]
    """
    array = np.asarray(array, dtype=np.double)
    if array.ndim != 2:
        raise ValueError('Only 2D arrays are supported.')
    level = float(level)
    if (fully_connected not in _param_options or
       positive_orientation not in _param_options):
        raise ValueError('Parameters "fully_connected" and'
        ' "positive_orientation" must be either "high" or "low".')
    point_list = _find_contours_cy.iterate_and_store(array, level,
                                                     fully_connected == 'high')
    contours = _assemble_contours(_take_2(point_list))
    if positive_orientation == 'high':
        contours = [c[::-1] for c in contours]
    return contours