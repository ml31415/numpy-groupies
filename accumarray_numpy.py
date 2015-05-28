import numpy as np

import accumarray_utils as utils

_func_alias, no_separate_nan_version = utils.get_alias_info(with_numpy=True)

def _fill_untouched(idx, ret, fillvalue):
    """any elements of ret not indexed by idx are set to fillvalue."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fillvalue

def _check_boolean(x, name='fillvalue'):
    if not (isinstance(x, bool) or x in (0,1)):
        raise Exception(name + " must be boolean or 0, 1")

_next_dtype = dict(uint8=np.dtype(np.int16), 
                   int8=np.dtype(np.int16),
                   uint16=np.dtype(np.int32),
                   int16=np.dtype(np.int32),
                   uint32=np.dtype(np.int64),
                   int32=np.dtype(np.int64),
                   uint64=np.dtype(np.float32),
                   int64=np.dtype(np.float32),
                   float16=np.dtype(np.float32),
                   float32=np.dtype(np.float64),
                   float64=np.dtype(np.complex64),
                   complex64=np.dtype(np.complex128))

def _get_minimum_dtype(x, dtype=bool):
    """returns the "most basic" dtype which represents `x` properly, which is
    at least as "complicated" as the specified dtype."""
    dtype = np.dtype(dtype)
    if ~np.isfinite(x):
        def test_foo(v, d):
            try:
                v = np.array(v, dtype=d)
                return True
            except (ValueError, OverflowError):
                return False
    else:
        def test_foo(v, d):
            return np.array(v, dtype=d) == v
        
    if test_foo(x, dtype):
        return dtype
    elif dtype.name not in _next_dtype:
        dtype = float # simplest thing to do for now
        
    while True:
        try:
            dtype = _next_dtype[dtype.name]
            if test_foo(x, dtype):
                return dtype
        except KeyError:
            return np.array(x).dtype # let numpy guess it for us
    

def _sort(idx, vals, n, fillvalue, dtype=None, reversed_=False):
    if isinstance(vals.dtype, np.complex):
        raise NotImplementedError("vals must be real, could use np.lexsort or sort with recarray for complex.")
    if not (np.isscalar(fillvalue) or len(fillvalue) == 0):
        raise Exception("fillvalue must be scalar or an empty sequence")
    if reversed_:
        order_idx = np.argsort(idx + -1j*vals)
    else:
        order_idx = np.argsort(idx + 1j*vals)
    counts = np.bincount(idx, minlength=n)
    if np.ndim(vals) == 0:
        vals = np.full(n, vals)
    ret = np.split(vals[order_idx], np.cumsum(counts)[:-1])
    ret = np.asarray(ret, dtype=object)
    if np.isscalar(fillvalue):
        _fill_untouched(idx, ret, fillvalue)
    return ret
    
def _rsort(idx, vals, n, fillvalue, dtype=None):
    return _sort(idx, vals, n, fillvalue, dtype=None, reversed_=True)
    
def _array(idx, vals, n, fillvalue, dtype=None):
    """groups vals into separate arrays, keeping the order intact."""
    if not (np.isscalar(fillvalue) or len(fillvalue) == 0):
        raise Exception("fillvalue must be scalar or an empty sequence")
    order_idx = np.argsort(idx, kind='mergesort')
    counts = np.bincount(idx, minlength=n)   
    ret = np.split(vals[order_idx], np.cumsum(counts)[:-1])
    ret = np.asarray(ret, dtype=object)
    if np.isscalar(fillvalue):
        _fill_untouched(idx, ret, fillvalue)
    return ret

def _sum(idx, vals, n, fillvalue, dtype=None):
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    if np.ndim(vals) == 0:
        ret = np.bincount(idx, minlength=n).astype(dtype)
        if vals != 1:
            ret *= vals
    else:            
        ret = np.bincount(idx, weights=vals, minlength=n).astype(dtype)
    if fillvalue != 0:
        _fill_untouched(idx, ret, fillvalue)
    return ret

def _last(idx, vals, n, fillvalue, dtype=None):
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    if fillvalue == 0:
        ret = np.zeros(n, dtype=dtype)
    else:
        ret = np.full(n, fillvalue, dtype=dtype)
    # repeated indexing gives last value, see:
    # the phrase "leaving behind the last value"  on this page:
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial
    ret[idx] = vals
    return ret

def _first(idx, vals, n, fillvalue, dtype=None):
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    if fillvalue == 0:
        ret = np.zeros(n, dtype=dtype)
    else:
        ret = np.full(n, fillvalue, dtype=dtype)
    ret[idx[::-1]] = vals[::-1]  # same trick as _last, but in reverse
    return ret


def _prod(idx, vals, n, fillvalue, dtype=None):
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    ret = np.full(n, 1, dtype=dtype)
    np.multiply.at(ret, idx, vals)
    if fillvalue != 1:
        _fill_untouched(idx, ret, fillvalue)
    return ret


def _all(idx, vals, n, fillvalue, dtype=bool):
    _check_boolean(fillvalue, name="fillvalue")
    ret = np.full(n, fillvalue, dtype=bool)
    if fillvalue:
        pass # already initialised to True
    else:
        ret[idx] = True
    idx = idx[~vals.astype(bool)]
    ret[idx] = False
    return ret
    
def _any(idx, vals, n, fillvalue, dtype=bool):
    _check_boolean(fillvalue, name="fillvalue")
    ret = np.full(n, fillvalue, dtype=bool)
    if fillvalue:
        ret[idx] = False
    else:
        pass # already initialsied to False
    idx = idx[vals.astype(bool)]
    ret[idx] = True
    return ret

def _min(idx, vals, n, fillvalue, dtype=None):
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    minfill = np.iinfo(vals.dtype).max if issubclass(vals.dtype.type, np.integer) else np.finfo(vals.dtype).max
    ret = np.full(n, minfill, dtype=dtype)
    np.minimum.at(ret, idx, vals)
    if fillvalue != minfill:
        _fill_untouched(idx, ret, fillvalue)
    return ret

def _max(idx, vals, n, fillvalue, dtype=None):
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    maxfill = np.iinfo(vals.dtype).min if issubclass(vals.dtype.type, np.integer) else np.finfo(vals.dtype).min
    ret = np.full(n, maxfill, dtype=dtype)
    np.maximum.at(ret, idx, vals)
    if fillvalue != maxfill:
        _fill_untouched(idx, ret, fillvalue)
    return ret

def _mean(idx, vals, n, fillvalue, dtype=None):
    if np.ndim(vals) == 0:
        raise Exception("cannot take mean with scalar vals")
    dtype = float if dtype is None else dtype
    counts = np.bincount(idx, minlength=n)
    sums = np.bincount(idx, weights=vals, minlength=n)
    with np.errstate(divide='ignore'):
        ret = sums.astype(dtype) / counts
    if not np.isnan(fillvalue):
        ret[counts == 0] = fillvalue
    return ret
       
def _var(idx, vals, n, fillvalue, dtype=None, sqrt=False):
    if np.ndim(vals) == 0:
        raise Exception("cannot take variance with scalar vals")
    dtype = float if dtype is None else dtype
    counts = np.bincount(idx, minlength=n)
    sums = np.bincount(idx, weights=vals, minlength=n)
    with np.errstate(divide='ignore'):
        means = sums.astype(dtype)/counts
        ret = np.bincount(idx, (vals - means[idx])**2, minlength=n) / counts
    if sqrt:
        ret = np.sqrt(ret) # this is now std not var
    if not np.isnan(fillvalue):
        ret[counts==0] = fillvalue
    return ret
        
def _std(idx, vals, n, fillvalue, dtype=None):
    return _var(idx, vals, n, fillvalue, dtype=dtype, sqrt=True)
   
def _allnan(idx, vals, n, fillvalue, dtype=bool):
    return _all(idx, np.isnan(vals), n, fillvalue=fillvalue, dtype=dtype)

def _anynan(idx, vals, n, fillvalue, dtype=bool):
    return _any(idx, np.isnan(vals), n, fillvalue=fillvalue, dtype=dtype)
    
def _generic_callable(idx, vals, n, fillvalue, dtype=None, foo=lambda g: g):
    """groups vals by inds, and then applies foo to each group in turn, placing
    the results in an array."""
    groups = _array(idx, vals, n, (), dtype=dtype)
    ret = np.full(n, fillvalue, dtype=object)
    for ii, g in enumerate(groups):
        if np.ndim(g) == 1 and len(g) > 0:
            ret[ii] = foo(g)
    return ret

_func_dict = dict(min=_min, max=_max, sum=_sum, prod=_prod, last=_last, first=_first,
                    all=_all, any=_any, mean=_mean, std=_std, var=_var,
                    anynan=_anynan, allnan=_allnan, sort=_sort, rsort=_rsort, 
                    array=_array)


def accumarray(idx, vals, sz=None, func='sum', fillvalue=0, order='F'):
    '''
    Accumulation function similar to Matlab's `accumarray` function.
    
    See readme file at https://github.com/ml31415/accumarray for 
    full description.

    This implementation is by DM, May 2015.

    Parameters
    ----------
    idx : 1D or ndarray or sequence of 1D ndarrays
        The length of the 1d array(s) should be the same shape as `vals`.
        This gives the "bin" (aka "group" or "index") in which to put the 
        given values, before* evaluating the aggregation function. 
        [*actually it's not really done in a separate step beforehand 
        in most cases, but you can think of it like that.]
    vals : 1D ndarray or scalar
        The data to be accumulated.  Note that the matlab version of this
        function accepts ndimensional inputs, but this does not.  Instead
        you must use `inds.ravel(), vals.ravel()`. (Note that if your arrays 
        are `order='F'` you can use this as a kwarg to `ravel` to prevent
        excess work being done, although the two arrays must match).
    sz : scalar or 1D sequence or None
        The desired shape of the output.  Note that no check is performed
        to ensure that indices of `idx` are within the specified size.
        If `idx` is a sequence of 1D arrays `sz` must be a 1d sequence or None
        rather than a scalar.
    func : string or callable (i.e. function)
        The primary list is: `"sum", "min", "max", "mean", "std", "var", "prod",
        "all", "any", "first", "last", "sort", "rsort", "array", "allnan", "anynan"`.  
        All, but the last five, are also available in a nan form as: 
        `"nansum", "nanmin"...etc.`  Note that any groups with only nans will
        be considered empty and assigned `fillvalue`, rather than being assinged
        `nan`. (If you want such groups to have the value `nan` you can use
        `"allnan"` to check which groups are all nan, and then set them to 
        `nan` in your output data.)
        
        For convenience a few aliases are defined (for both the nan and basic 
        versions):
         * `"min"`: `"amin"` and `"minimum"`       
         * `"max"`: `"amin"` and `"minimum"`       
         * `"prod"`: `"product"` and `"times"` and `"multiply"`    
         * `"sum"`: `"plus"` and `"add"`    
         * `"any"`: `"or"`     
         * `"all"`: `"and"`   
         * `"array"`: `"split"` and `"splice"`    
         * `"sort"`: `"sorted"` and `"asort"` and `"asorted"`     
         * `"rsort"`: `"rsorted"` and `"dsort"` and `"dsorted"`    
        
        The string matching is case-insensitive.
        
        By providing one of the recognised string inputs you will get an optimised
        function (although, as of numpy 1.9, `"min"`, `"max"` and `"prod"
        are actually not as fast as they should be, by a factor of 10x or more.)
        
        If instead of providing a string you provide a numpy function, e.g.
        `np.sum`, in most cases, this will be aliased to one of the above strings.
        If no alias is recognised, it will be treated as a generic callable function.
        
        For the case of generic callable functions, the data will be split into 
        actual groups and fed into the callable, one at a time.
        This is true even for `np.ufunc`s, which could potentially use their
        `.at` methods.  However using `.at` requires some understanding of what 
        the function is diong, e.g. logical_or should be initialised with 0s,
        but logical_and should be initialised with 1s.
        
    fillvalue: scalar
        specifies the value to put in output where there was no input data,
        default is `0`, but you might want `np.nan` or some other specific
        value of your choosing.
    
    Returns
    -------
    out : ndarray
        The accumulated results.  The dtype of the result will be float in cases
        where division occurs as part of the accumulation, otherwise the minimal
        dtype will be chosen to match `vals` and the `fillvalue`.
    
    Examples
    --------
    >>> from numpy import array
    >>> vals = array([12.0, 3.2, -15, 88, 12.9])
    >>> idx = array([1,    0,    1,  4,   1  ])
    >>> accumarray(idx, vals) # group vals by idx and sum
    array([3.2, 9.9, 0, 88.])
    >>> accumarray(idx, vals, sz=8, func='min', fillval=np.nan)
    array([3.2, -15., nan, 88., nan, nan, nan, nan])
    >>> accumarray(test_idx, test_vals, sz=5, func=lambda x: ' + '.join(str(xx) for xx in x),fillvalue='')
    ['3.2' '12.0 + -15.0 + 12.9' '' '' '88.0']
    '''

        
    vals = np.asanyarray(vals)
    idx = np.asanyarray(idx)
    
    # do some basic checking on vals
    if not issubclass(idx.dtype.type, np.integer):
        raise TypeError("idx must be of integer type")
    if np.ndim(vals) > 1:
        raise Exception("vals must be scalar or 1 dimensional, use .ravel to flatten.")

    # Do some fairly extensive checking of idx and vals, trying to give the user
    # as much help as possible with what is wrong.
    # Also, convert ndindexing to 1d indexing
    ndim_idx = np.ndim(idx)
    if ndim_idx not in (1,2):
        raise Exception("Expected indices to have either 1 or 2 dimension.")
    elif ndim_idx == 1:        
        if not (np.ndim(vals)==0 or len(vals) == idx.shape[0]):
            raise Exception("idx and vals must be of the same length, or vals can be scalar")
        if np.any(idx<0):
            raise Exception("Negative indices not supported.")
        if sz is not None:
            if not np.isscalar(sz):
                raise Exception("Output size must be scalar or None")
            if np.any(idx>sz-1):
                raise Exception("One or more indices are too large for size %d." % sz)
        else:
            sz = np.max(idx) + 1
        sz_n = sz
    else: #ndim_idx == 2
        if  not (np.ndim(vals)==0 or len(vals) == idx.shape[1]):
            raise Exception("vals has length %d, but idx has length %d." % (len(vals), idx.shape[1]))
        if sz is None:
            sz = np.max(idx, axis=1) + 1
        else:
            if np.isscalar(sz):
                raise Exception("Output size must be None or 1d sequence of length %d" % idx.shape[0])
            if len(sz) != idx.shape[1]:
                raise Exception("%d sizes given, but %d output dimensions specified in index" % (len(sz), idx.shape[0]))
            
        idx = np.ravel_multi_index(tuple(idx), sz, order=order, mode='raise')
        sz_n = np.prod(sz)
    
    if not isinstance(func, basestring):
        if func in _func_alias:
            func = _func_alias[func]
        elif not callable(func):
            raise Exception("func is neither a string nor a callable object.")

    if not isinstance(func, basestring):            
        # do simple grouping and execute function in loop
        ret = _generic_callable(idx, vals, sz_n, fillvalue, foo=func)
    else:
        # deal with nans and find the function
        original_func = func
        func = func.lower()
        if func.startswith('nan'):
            func = func[3:]
            func = _func_alias.get(func, func)
            if func in no_separate_nan_version:
                raise Exception(original_func[3:] + " does not have a nan- version.")
            if np.ndim(vals) == 0:
                raise Exception("nan- version not supported for scalar input.")
            good = ~np.isnan(vals)
            vals = vals[good]
            idx = idx[good]
        else:
            func = _func_alias.get(func, func)
        if func not in _func_dict:
            raise Exception(original_func + " not found in list of available functions.")
        func = _func_dict[func]
    
        # run the function
        ret = func(idx, vals, sz_n, fillvalue=fillvalue)
        
    # deal with ndimensional indexing
    if ndim_idx == 2:
        ret = ret.reshape(sz, order=order)
        
    return ret
    
