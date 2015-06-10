import logging
import numpy as np
from numba import jit, double

logging.captureWarnings(True)

from .utils import check_group_idx, _doc_str
from .aggregate_numpy import aggregate as aggregate_np


@jit
def _iter_sum(wi, val, res, counter, tmp):
    res[wi] += val

@jit
def _iter_prod(wi, val, res, counter, tmp):
    res[wi] *= val

@jit
def _iter_min(wi, val, res, counter, tmp):
    if val < res[wi]:
        res[wi] = val

@jit
def _iter_max(wi, val, res, counter, tmp):
    if val > res[wi]:
        res[wi] = val

@jit
def _iter_all(wi, val, res, counter, tmp):
    if counter[wi] == 0:
        res[wi] = 1
    counter[wi] = 1
    if not val:
        res[wi] = 0

@jit
def _iter_any(wi, val, res, counter, tmp):
    if counter[wi] == 0:
        res[wi] = 0
    counter[wi] = 1
    if val:
        res[wi] = 1

@jit
def _iter_allnan(wi, val, res, counter, tmp):
    if counter[wi] == 0:
        res[wi] = 1
    counter[wi] = 1
    if val == val:
        res[wi] = 0

@jit
def _iter_anynan(wi, val, res, counter, tmp):
    if counter[wi] == 0:
        res[wi] = 0
    counter[wi] = 1
    if val != val:
        res[wi] = 1

@jit
def _iter_mean(wi, val, res, counter, tmp):
    counter[wi] += 1
    res[wi] += val

@jit
def _iter_std(wi, val, res, counter, tmp):
    counter[wi] += 1
    tmp[wi] += val
    res[wi] += val * val

@jit
def _finish_mean(res, counter, tmp, fillvalue):
    for i in range(len(res)):
        if counter[i]:
            res[i] /= counter[i]
        else:
            res[i] = fillvalue

@jit(locals=dict(mean=double))
def _finish_std(res, counter, tmp, fillvalue):
    for i in range(len(res)):
        if counter[i]:
            mean = tmp[i] / counter[i]
            res[i] = np.sqrt(res[i] / counter[i] - mean * mean)
        else:
            res[i] = fillvalue

@jit
def _count_steps(group_idx):
    cmp_pos = 0
    res_len = 0
    for i in range(len(group_idx)):
        if group_idx[i] != group_idx[cmp_pos]:
            cmp_pos = i
            res_len += 1
    return res_len

@jit
def _maxval(group_idx):
    m = group_idx[0]
    for i in group_idx:
        if i > m:
            m = i
    return m


iter_funcs = {'sum': _iter_sum, 'prod': _iter_prod,
              'min': _iter_min, 'max': _iter_max,
              'amin': _iter_min, 'amax': _iter_max,
              'mean': _iter_mean, 'std':_iter_std,
              'nansum': _iter_sum, 'nanprod': _iter_prod,
              'nanmin': _iter_min, 'nanmax': _iter_max,
              'nanmean': _iter_mean, 'nanstd': _iter_std,
              'all': _iter_all, 'any': _iter_any,
              'allnan': _iter_allnan, 'anynan': _iter_anynan}

finish_funcs = {_iter_std: _finish_std, _iter_mean: _finish_mean}

dtype_by_func = {list: 'object',
                 tuple: 'object',
                 sorted: 'object',
                 np.array: 'object',
                 np.sort: 'object',
                 np.mean: 'float',
                 np.std: 'float',
                 np.all: 'bool',
                 np.any: 'bool',
                 all: 'bool',
                 any: 'bool',
                 'mean': 'float',
                 'std': 'float',
                 'nanmean': 'float',
                 'nanstd': 'float',
                 'all': 'bool',
                 'any': 'bool',
                 'allnan': 'bool',
                 'anynan': 'bool',
                 }


@jit
def _loop_contiguous(iter_func, group_idx, a, res, counter, tmp, nanskip, fillvalue):
    wi = 0
    cmp_pos = 0
    for i in range(len(group_idx)):
        if group_idx[i] != group_idx[cmp_pos]:
            cmp_pos = i
            wi += 1
        if nanskip and a[i] != a[i]:
            continue
        iter_func(wi, a[i], res, counter, tmp)


@jit
def _loop_incontiguous(iter_func, group_idx, a, res, counter, tmp, nanskip, fillvalue):
    wi = 0
    for i in range(len(group_idx)):
        wi = group_idx[i]
        if nanskip and a[i] != a[i]:
            continue
        iter_func(wi, a[i], res, counter, tmp)


def aggregate(group_idx, a, func='sum', dtype=None, fillvalue=0):
    """ For most common cases, operates like usual matlab aggregatearray
        http://www.mathworks.com/help/matlab/ref/aggregatearray.html
    
        group_idx and a are generally treated as flattened arrays.
        
        Contiguous:
        Same values within group_idx can be expected to be grouped
        or be treated as new values starting a new group, in 
        case they should appear another time
        E.g. group_idx = [1 1 2 2 2 1 1 3 3] with contiguous set will 
        be treated the same way as [0 0 1 1 1 2 2 3 3]
        That way, feeding data through np.unique, maintaining order
        etc. can be omitted. It also gives a nice speed boost, as
        np.argsort of group_idx can also be omitted.
    """
    if not isinstance(func, basestring):
        if getattr(func, '__name__', None) in iter_funcs:
            func = func.__name__
        else:
            # Fall back to acuum_np if no optimized version is available
            return aggregate_np(group_idx, a, func=func, dtype=dtype,
                            fillvalue=fillvalue)
    if func not in iter_funcs:
        raise ValueError("No optimized function %s available" % func)

    check_group_idx(group_idx, a, check_min=False)

    iter_func = iter_funcs[func]
    nanskip = isinstance(a.dtype, np.float) and func.startswith('nan')
    dtype = dtype or dtype_by_func.get(func, a.dtype)
    res = np.zeros(_count_steps(group_idx), dtype=dtype)
    if fillvalue != 0 and iter_func not in {_iter_mean, _iter_std}:
        res.fill(fillvalue)

    if iter_func in {_iter_min, _iter_max, _iter_sum, _iter_prod}:
        counter = 0
        tmp = 0
    elif iter_func == _iter_std:
        counter = np.zeros_like(res, dtype=int)
        tmp = np.zeros_like(res)
    else:
        counter = np.zeros_like(res, dtype=int)
        tmp = 0

    _loop_incontiguous(iter_func, group_idx, a, res, counter, tmp, nanskip, fillvalue)

    try:
        finish_func = finish_funcs[iter_func]
    except KeyError:
        pass
    else:
        finish_func(res, counter, tmp, fillvalue)

    return res
    
aggregate.__doc__ = _doc_str

@jit
def unpack(group_idx, res):
    """ Take an aggregate packed array and uncompress it to the size of group_idx. 
        This is equivalent to res[group_idx], but gives a more than 
        3-fold speedup.
    """
    check_group_idx(group_idx)
    unpacked = np.zeros_like(group_idx, dtype=res.dtype)

    res_len = len(res)
    for i in range(len(group_idx)):
        if group_idx[i] >= 0 and group_idx[i] < res_len:
            unpacked[i] = res[group_idx[i]]
    return unpacked

@jit
def step_indices(group_idx):
    """ Get the edges of areas within group_idx, which are filled 
        with the same value
    """
    ilen = _count_steps(group_idx) + 1
    indices = np.empty(ilen, int)
    indices[0] = 0
    indices[-1] = group_idx.size

    cmp_pos = 0
    wi = 1
    for i in range(1, len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            indices[wi] = i
            wi += 1

    return indices


# if __name__ == '__main__':
#    group_idx = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
#    a = np.arange(group_idx.size, dtype=float)
#    mode = 'contiguous'
#    for fn in (np.mean, np.std, 'allnan', 'anynan'):
#        res = aggregate(group_idx, a, mode=mode, func=fn)
#        print res
