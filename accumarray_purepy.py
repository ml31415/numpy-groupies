import math # needed for nan 
import itertools # needed for groupby

import accumarray_utils as utils

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


def accumarray(idx, vals, func='sum', sz=None, fillvalue=0, order=None):
    """ Accumulation function similar to Matlab's `accumarray` function.
    
        See readme file at https://github.com/ml31415/accumarray for 
        full description.

        This implementation is by DM, May 2015.
    """
    original_func = func
    func = _func_alias.get(func, func)
    if isinstance(func, basestring) and func.startswith('nan') and func in no_separate_nan_version:
        raise Exception(original_func[3:] + " does not have a nan- version.")
                
    if isinstance(func, basestring) and func.startswith('nan'):
        raise NotImplementedError("nan versions of functions not implemented.")
        
    # find the function
    if isinstance(func, basestring):
        if func not in _func_dict:
            raise Exception(func + " not found in list of functions.")
        func = _func_dict[func]
    elif callable(func):
        pass # we can use it as is
    else:
        raise Exception("func should be a callable function or recognised function name")

    # Check for 2d idx        
    for x in idx:
        try:
            x[0]
            raise NotImplementedError("pure python implementation doesn't accept 2d idx input.")
        except IndexError:
            continue # getting an error is good, it means this is scalar
            
    if sz is None:
        sz = 1 + max(idx)        

    # sort data and evaluate function on groups
    data = sorted(zip(idx, vals), key=lambda tp: tp[0])
    ret = [fillvalue]*sz
    for ix, group in itertools.groupby(data, key=lambda tp: tp[0]):
        ret[ix] = func(tuple(val for _,val in group))

    return ret

