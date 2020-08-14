from trackpy.linking import Linker
import pandas as pd, numpy as np
from trackpy.linking.linking import guess_pos_columns, pandas_sort, coords_from_df

#trackpy.link_iter with a verbose option added to shut the darn thing up.  It's logger.info calls dump the log and cause GIL interference between my dask.bag workers! :( Boo!
def link_iter(coords_iter, search_range, verbose=True, **kwargs):
    """
    link_iter(coords_iter, search_range, memory=0, predictor=None,
        adaptive_stop=None, adaptive_step=0.95, neighbor_strategy=None,
        link_strategy=None, dist_func=None, to_eucl=None)

    Link an iterable of per-frame coordinates into trajectories.

    Parameters
    ----------
    coords_iter : iterable
        the iterable produces 2d numpy arrays of coordinates (shape: N, ndim).
        to tell link_iter what frame number each array is, the iterable may
        be enumerated so that it produces (number, 2d array) tuples
    search_range : float or tuple
        the maximum distance features can move between frames,
        optionally per dimension
    memory : integer, optional
        the maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. Default: 0
    predictor : function, optional
        Improve performance by guessing where a particle will be in
        the next frame.
        For examples of how this works, see the "predict" module.
    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range until the subnet is solvable. If search_range
        becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
    adaptive_step : float, optional
        Reduce search_range by multiplying it by this factor.
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features. Default 'KDTree'.
    link_strategy : {'recursive', 'nonrecursive', 'hybrid', 'numba', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses hybrid (numba+recursive) if available
        'drop' causes particles in subnetworks to go unlinked
    dist_func : function, optional
        a custom distance function that takes two 1D arrays of coordinates and
        returns a float. Must be used with the 'BTree' neighbor_strategy.
    to_eucl : function, optional
        function that transforms a N x ndim array of positions into coordinates
        in Euclidean space. Useful for instance to link by Euclidean distance
        starting from radial coordinates. If search_range is anisotropic, this
        parameter cannot be used.

    Yields
    ------
    tuples (t, list of particle ids)

    See also
    --------
    link

    Notes
    -----
    This is an implementation of the Crocker-Grier linking algorithm.
    [1]_

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217
    """
    # ensure that coords_iter is iterable
    coords_iter = iter(coords_iter)

    # interpret the first element of the iterable
    val = next(coords_iter)
    if isinstance(val, np.ndarray):
        # the iterable was not enumerated, so enumerate the remainder
        coords_iter = enumerate(coords_iter, start=1)
        t, coords = 0, val
    else:
        t, coords = val

    # initialize the linker and yield the particle ids of the first frame
    linker = Linker(search_range, **kwargs)
    linker.init_level(coords, t)
    yield t, linker.particle_ids

    for t, coords in coords_iter:
        linker.next_level(coords, t)
        if verbose:
            logger.info("Frame {0}: {1} trajectories present.".format(t, len(linker.particle_ids)))
        yield t, linker.particle_ids

def link(f, search_range, pos_columns=None, t_column='frame', verbose=True, **kwargs):
    """
    link(f, search_range, pos_columns=None, t_column='frame', memory=0,
        predictor=None, adaptive_stop=None, adaptive_step=0.95,
        neighbor_strategy=None, link_strategy=None, dist_func=None,
        to_eucl=None)

    Link a DataFrame of coordinates into trajectories.

    Parameters
    ----------
    f : DataFrame
        The DataFrame must include any number of column(s) for position and a
        column of frame numbers. By default, 'x' and 'y' are expected for
        position, and 'frame' is expected for frame number. See below for
        options to use custom column names.
    search_range : float or tuple
        the maximum distance features can move between frames,
        optionally per dimension
    pos_columns : list of str, optional
        Default is ['y', 'x'], or ['z', 'y', 'x'] when 'z' is present in f
    t_column : str, optional
        Default is 'frame'
    memory : integer, optional
        the maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. 0 by default.
    predictor : function, optional
        Improve performance by guessing where a particle will be in
        the next frame.
        For examples of how this works, see the "predict" module.
    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range until the subnet is solvable. If search_range
        becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
    adaptive_step : float, optional
        Reduce search_range by multiplying it by this factor.
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features. Default 'KDTree'.
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'hybrid', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses hybrid (numba+recursive) if available
        'drop' causes particles in subnetworks to go unlinked
    dist_func : function, optional
        a custom distance function that takes two 1D arrays of coordinates and
        returns a float. Must be used with the 'BTree' neighbor_strategy.
    to_eucl : function, optional
        function that transforms a N x ndim array of positions into coordinates
        in Euclidean space. Useful for instance to link by Euclidean distance
        starting from radial coordinates. If search_range is anisotropic, this
        parameter cannot be used.

    Returns
    -------
    DataFrame with added column 'particle' containing trajectory labels.
    The t_column (by default: 'frame') will be coerced to integer.

    See also
    --------
    link_iter

    Notes
    -----
    This is an implementation of the Crocker-Grier linking algorithm.
    [1]_

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217

    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)

    # copy the dataframe
    f = f.copy()
    # coerce t_column to integer type
    if not np.issubdtype(f[t_column].dtype, np.integer):
        f[t_column] = f[t_column].astype(np.integer)
    # sort on the t_column
    pandas_sort(f, t_column, inplace=True)

    coords_iter = coords_from_df(f, pos_columns, t_column)
    ids = []
    for i, _ids in link_iter(coords_iter, search_range, verbose=verbose, **kwargs):
        ids.extend(_ids)

    f['particle'] = ids
    return f