import accumarray_utils as utils
import math # needed for nan

_func_alias, no_separate_nan_version = utils.get_alias_info(with_numpy=False)


# Implementation of main functions....

# min - builtin
# max - builtin
# sum - builtin
# all - builtin
# any - builtin
_last = lambda x: x[-1]
_first = lambda x: x[0]
_mean = lambda x: sum(x)/len(x)
_array = lambda x: x
_sort = lambda x: sorted(x)
_rsort = lambda x: sorted(x, reverse=True)
def _anynan(x):
    return any(math.isnan(xx) for xx in x)
def _allnan(x):
    return all(math.isnan(xx) for xx in x)
def _var(x):
    mean = _mean(x)
    return sum((xx-mean)**2 for xx in x)/len(x)
def _std(x):
    return math.sqrt(_var(x))
def _prod(x):
    r = x[0]
    for xx in x[1:]:
        r *= xx
    return r
_func_dict = dict(min=min, max=max, sum=sum, prod=_prod, last=_last, first=_first,
                all=all, any=any, mean=_mean, std=_std, var=_var,
                anynan=_anynan, allnan=_allnan, sort=_sort, rsort=_rsort, 
                array=_array)


def accum_py(idx, vals, func='sum', sz=None, fillvalue=0, order='F'):
    """ Accumulation function similar to Matlab's `accumarray` function.
    
        See readme file at https://github.com/ml31415/accumarray for 
        full description.

        This implementation is from the scipy cookbook:
            http://www.scipy.org/Cookbook/AccumarrayLike
    """
    original_func = func
    func = _func_alias.get(func, func)
    if func.startswith('nan') and func in no_separate_nan_version:
        raise Exception(original_func[3:] + " does not have a nan- version.")
                
    if func.startswith('nan'):
        raise NotImplemented("nan versions of functions not implemented.")
        
    if isinstance(func, basestring):
        if func not in _func_dict:
            raise Exception(func + " not found in list of functions.")
        func = _func_dict[func]
    elif callable(func):
        pass # we can use it as is
    else:
        raise Exception("func should be a callable function or recognised function name")
        
        
    raise NotImplemented("Need to provide pure-python implementation.")
    
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

