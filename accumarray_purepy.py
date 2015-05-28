def accum_py(idx, vals, func=np.sum, sz=None, fillvalue=0, order='F'):
    """ Accumulation function similar to Matlab's `accumarray` function.
    
        See readme file at https://github.com/ml31415/accumarray for 
        full description.

        This implementation is from the scipy cookbook:
            http://www.scipy.org/Cookbook/AccumarrayLike
    """
    raise NotImplemented("Need to finish refactoring, and provide pure-python implementations for functions specified by string")
    
    if mode == 'downscaled':
        _, idx = np.unique(idx, return_inverse=True)
    _check_idx(idx, vals)
    _check_mode(mode)

    dtype = dtype or _dtype_by_func(func, vals)
    if idx.shape == vals.shape:
        idx = np.expand_dims(idx, -1)

    adims = tuple(xrange(vals.ndim))
    if sz is None:
        sz = 1 + np.squeeze(np.apply_over_axes(np.max, idx, axes=adims))
    sz = np.atleast_1d(sz)

    # Create an array of python lists of values.
    groups = np.empty(sz, dtype='O')
    for s in product(*[xrange(k) for k in sz]):
        # All fields in groups
        groups[s] = []

    for s in product(*[xrange(k) for k in vals.shape]):
        # All fields in vals
        indx = tuple(idx[s])
        val = vals[s]
        groups[indx].append(val)

    # Create the output array.
    ret = np.zeros(sz, dtype=dtype)
    for s in product(*[xrange(k) for k in sz]):
        # All fields in groups
        if groups[s] == []:
            ret[s] = fillvalue
        else:
            ret[s] = func(groups[s])

    return ret

