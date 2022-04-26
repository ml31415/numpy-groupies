import numpy as np

from .utils import aggregate_common_doc, check_boolean, funcs_no_separate_nan, get_func, isstr
from .utils_numpy import (aliasing, check_dtype, check_fill_value, input_validation, iscomplexobj,
                          minimum_dtype, minimum_dtype_scalar)


def _sum(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype_scalar(fill_value, dtype, a)

    if np.ndim(a) == 0:
        ret = np.bincount(group_idx, minlength=size).astype(dtype)
        if a != 1:
            ret *= a
    else:
        if iscomplexobj(a):
            ret = np.empty(size, dtype=dtype)
            ret.real = np.bincount(group_idx, weights=a.real,
                                   minlength=size)
            ret.imag = np.bincount(group_idx, weights=a.imag,
                                   minlength=size)
        else:
            ret = np.bincount(group_idx, weights=a,
                              minlength=size).astype(dtype)

    if fill_value != 0:
        _fill_untouched(group_idx, ret, fill_value)
    return ret


def _prod(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype_scalar(fill_value, dtype, a)
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != 1:
        ret[group_idx] = 1  # product starts from 1
    np.multiply.at(ret, group_idx, a)
    return ret


def _len(group_idx, a, size, fill_value, dtype=None):
    return _sum(group_idx, 1, size, fill_value, dtype=int)


def _last(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    # repeated indexing gives last value, see:
    # the phrase "leaving behind the last value"  on this page:
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial
    ret[group_idx] = a
    return ret


def _first(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    ret[group_idx[::-1]] = a[::-1]  # same trick as _last, but in reverse
    return ret


def _all(group_idx, a, size, fill_value, dtype=None):
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if not fill_value:
        ret[group_idx] = True
    ret[group_idx.compress(np.logical_not(a))] = False
    return ret


def _any(group_idx, a, size, fill_value, dtype=None):
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if fill_value:
        ret[group_idx] = False
    ret[group_idx.compress(a)] = True
    return ret


def _min(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmax = np.iinfo(a.dtype).max if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).max
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmax:
        ret[group_idx] = dmax  # min starts from maximum
    np.minimum.at(ret, group_idx, a)
    return ret


def _max(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmin = np.iinfo(a.dtype).min if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).min
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmin:
        ret[group_idx] = dmin  # max starts from minimum
    np.maximum.at(ret, group_idx, a)
    return ret


def _argmax(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or int)
    dmin = np.iinfo(a.dtype).min if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).min
    group_max = _max(group_idx, a, size, dmin)
    is_max = a == group_max[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_max = group_idx[is_max]
    argmax, = is_max.nonzero()
    ret[group_idx_max[::-1]] = argmax[::-1]  # reverse to ensure first value for each group wins
    return ret


def _argmin(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or int)
    dmax = np.iinfo(a.dtype).max if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).max
    group_min = _min(group_idx, a, size, dmax)
    is_min = a == group_min[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_min = group_idx[is_min]
    argmin, = is_min.nonzero()
    ret[group_idx_min[::-1]] = argmin[::-1]  # reverse to ensure first value for each group wins
    return ret


def _mean(group_idx, a, size, fill_value, dtype=np.dtype(np.float64)):
    if np.ndim(a) == 0:
        raise ValueError("cannot take mean with scalar a")
    counts = np.bincount(group_idx, minlength=size)
    if iscomplexobj(a):
        dtype = a.dtype  # TODO: this is a bit clumsy
        sums = np.empty(size, dtype=dtype)
        sums.real = np.bincount(group_idx, weights=a.real,
                                minlength=size)
        sums.imag = np.bincount(group_idx, weights=a.imag,
                                minlength=size)
    else:
        sums = np.bincount(group_idx, weights=a,
                           minlength=size).astype(dtype)

    with np.errstate(divide='ignore', invalid='ignore'):
        ret = sums.astype(dtype) / counts
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    return ret


def _var(group_idx, a, size, fill_value, dtype=np.dtype(np.float64),
         sqrt=False, ddof=0):
    if np.ndim(a) == 0:
        raise ValueError("cannot take variance with scalar a")
    counts = np.bincount(group_idx, minlength=size)
    sums = np.bincount(group_idx, weights=a, minlength=size)
    with np.errstate(divide='ignore', invalid='ignore'):
        means = sums.astype(dtype) / counts
        counts = np.where(counts > ddof, counts - ddof, 0)
        ret = np.bincount(group_idx, (a - means[group_idx]) ** 2,
                          minlength=size) / counts
    if sqrt:
        ret = np.sqrt(ret)  # this is now std not var
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    return ret


def _std(group_idx, a, size, fill_value, dtype=np.dtype(np.float64), ddof=0):
    return _var(group_idx, a, size, fill_value, dtype=dtype, sqrt=True,
                ddof=ddof)


def _allnan(group_idx, a, size, fill_value, dtype=bool):
    return _all(group_idx, np.isnan(a), size, fill_value=fill_value,
                dtype=dtype)


def _anynan(group_idx, a, size, fill_value, dtype=bool):
    return _any(group_idx, np.isnan(a), size, fill_value=fill_value,
                dtype=dtype)


def _sort(group_idx, a, size=None, fill_value=None, dtype=None, reverse=False):
    sortidx = np.lexsort((-a if reverse else a, group_idx))
    # Reverse sorting back to into grouped order, but preserving groupwise sorting
    revidx = np.argsort(np.argsort(group_idx, kind='mergesort'), kind='mergesort')
    return a[sortidx][revidx]


def _array(group_idx, a, size, fill_value, dtype=None):
    """groups a into separate arrays, keeping the order intact."""
    if fill_value is not None and not (np.isscalar(fill_value) or
                                       len(fill_value) == 0):
        raise ValueError("fill_value must be None, a scalar or an empty "
                         "sequence")
    order_group_idx = np.argsort(group_idx, kind='mergesort')
    counts = np.bincount(group_idx, minlength=size)
    ret = np.split(a[order_group_idx], np.cumsum(counts)[:-1])
    ret = np.asanyarray(ret, dtype="object")
    if fill_value is None or np.isscalar(fill_value):
        _fill_untouched(group_idx, ret, fill_value)
    return ret


def _generic_callable(group_idx, a, size, fill_value, dtype=None,
                      func=lambda g: g, **kwargs):
    """groups a by inds, and then applies foo to each group in turn, placing
    the results in an array."""
    groups = _array(group_idx, a, size, ())
    ret = np.full(size, fill_value, dtype=dtype or np.float64)

    for i, grp in enumerate(groups):
        if np.ndim(grp) == 1 and len(grp) > 0:
            ret[i] = func(grp)
    return ret


def _cumsum(group_idx, a, size, fill_value=None, dtype=None):
    """
    N to N aggregate operation of cumsum. Perform cumulative sum for each group.

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _cumsum(group_idx, a, np.max(group_idx) + 1)
    >>> array([ 3,  4,  5,  6, 15,  9, 15, 22,  7,  0, 15, 17,  6, 14, 31, 39])
    """
    sortidx = np.argsort(group_idx, kind='mergesort')
    invsortidx = np.argsort(sortidx, kind='mergesort')
    group_idx_srt = group_idx[sortidx]

    a_srt = a[sortidx]
    a_srt_cumsum = np.cumsum(a_srt, dtype=dtype)

    increasing = np.arange(len(a), dtype=int)
    group_starts = _min(group_idx_srt, increasing, size, fill_value=0)[group_idx_srt]
    a_srt_cumsum += -a_srt_cumsum[group_starts] + a_srt[group_starts]
    return a_srt_cumsum[invsortidx]


def _nancumsum(group_idx, a, size, fill_value=None, dtype=None):
    a_nonans = np.where(np.isnan(a), 0, a)
    group_idx_nonans = np.where(np.isnan(group_idx), np.nanmax(group_idx) + 1, group_idx)
    return _cumsum(group_idx_nonans, a_nonans, size, fill_value=fill_value, dtype=dtype)


_impl_dict = dict(min=_min, max=_max, sum=_sum, prod=_prod, last=_last,
                  first=_first, all=_all, any=_any, mean=_mean, std=_std,
                  var=_var, anynan=_anynan, allnan=_allnan, sort=_sort,
                  array=_array, argmax=_argmax, argmin=_argmin, len=_len,
                  cumsum=_cumsum, generic=_generic_callable)
_impl_dict.update(('nan' + k, v) for k, v in list(_impl_dict.items())
                  if k not in funcs_no_separate_nan)


def _aggregate_base(group_idx, a, func='sum', size=None, fill_value=0,
                    order='C', dtype=None, axis=None, _impl_dict=_impl_dict,
                    _nansqueeze=False, cache=None, **kwargs):
    group_idx, a, flat_size, ndim_idx, size, unravel_shape = input_validation(group_idx, a,
                                                                              size=size, order=order, axis=axis, func=func)

    if group_idx.dtype == np.dtype("uint64"):
        # Force conversion to signed int, to avoid issues with bincount etc later
        group_idx = group_idx.astype(int)

    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        # do simple grouping and execute function in loop
        ret = _impl_dict.get('generic', _generic_callable)(group_idx, a, flat_size, fill_value, func=func,
                                                           dtype=dtype, **kwargs)
    else:
        # deal with nans and find the function
        if func.startswith('nan'):
            if np.ndim(a) == 0:
                raise ValueError("nan-version not supported for scalar input.")
            if _nansqueeze:
                good = ~np.isnan(a)
                a = a[good]
                group_idx = group_idx[good]

        dtype = check_dtype(dtype, func, a, flat_size)
        check_fill_value(fill_value, dtype, func=func)
        func = _impl_dict[func]
        ret = func(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype,
                   **kwargs)

    # deal with ndimensional indexing
    if ndim_idx > 1:
        if unravel_shape is not None:
            # A negative fill_value cannot, and should not, be unraveled.
            mask = ret == fill_value
            ret[mask] = 0
            ret = np.unravel_index(ret, unravel_shape)[axis]
            ret[mask] = fill_value
        ret = ret.reshape(size, order=order)
    return ret


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    return _aggregate_base(group_idx, a, size=size, fill_value=fill_value,
                           order=order, dtype=dtype, func=func, axis=axis,
                           _impl_dict=_impl_dict, _nansqueeze=True, **kwargs)


aggregate.__doc__ = """
    This is the pure numpy implementation of aggregate.
    """ + aggregate_common_doc


def _fill_untouched(idx, ret, fill_value):
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value
