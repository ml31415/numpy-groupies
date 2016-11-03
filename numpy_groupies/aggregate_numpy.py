import numpy as np

from .utils import (check_boolean, _no_separate_nan_version, get_func,
                    aliasing, fill_untouched, minimum_dtype, input_validation,
                    check_dtype, minimum_dtype_scalar, _doc_str, isstr)


def _sort(group_idx, a, size, fill_value, dtype=None, reversed_=False):
    if np.iscomplexobj(a):
        raise NotImplementedError("a must be real, could use np.lexsort or "
                                  "sort with recarray for complex.")
    if not (np.isscalar(fill_value) or len(fill_value) == 0):
        raise ValueError("fill_value must be scalar or an empty sequence")
    if reversed_:
        order_group_idx = np.argsort(group_idx + -1j * a, kind='mergesort')
    else:
        order_group_idx = np.argsort(group_idx + 1j * a, kind='mergesort')
    counts = np.bincount(group_idx, minlength=size)
    if np.ndim(a) == 0:
        a = np.full(size, a, dtype=type(a))
    ret = np.split(a[order_group_idx], np.cumsum(counts)[:-1])
    ret = np.asarray(ret, dtype=object)
    if np.isscalar(fill_value):
        fill_untouched(group_idx, ret, fill_value)
    return ret


def _rsort(group_idx, a, size, fill_value, dtype=None):
    return _sort(group_idx, a, size, fill_value, dtype=None, reversed_=True)


def _array(group_idx, a, size, fill_value, dtype=None):
    """groups a into separate arrays, keeping the order intact."""
    if fill_value is not None and not (np.isscalar(fill_value) or
                                       len(fill_value) == 0):
        raise ValueError("fill_value must be None, a scalar or an empty "
                         "sequence")
    order_group_idx = np.argsort(group_idx, kind='mergesort')
    counts = np.bincount(group_idx, minlength=size)
    ret = np.split(a[order_group_idx], np.cumsum(counts)[:-1])
    ret = np.asanyarray(ret)
    if fill_value is None or np.isscalar(fill_value):
        fill_untouched(group_idx, ret, fill_value)
    return ret


def _sum(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype_scalar(fill_value, dtype, a)

    if np.ndim(a) == 0:
        ret = np.bincount(group_idx, minlength=size).astype(dtype)
        if a != 1:
            ret *= a
    else:
        if np.iscomplexobj(a):
            ret = np.empty(size, dtype=dtype)
            ret.real = np.bincount(group_idx, weights=a.real,
                              minlength=size)
            ret.imag = np.bincount(group_idx, weights=a.imag,
                              minlength=size)
        else:
            ret = np.bincount(group_idx, weights=a,
                              minlength=size).astype(dtype)

    if fill_value != 0:
        fill_untouched(group_idx, ret, fill_value)
    return ret


def _len(group_idx, a, size, fill_value, dtype=None):
    return _sum(group_idx, 1, size, fill_value, dtype=int)


def _last(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    if fill_value == 0:
        ret = np.zeros(size, dtype=dtype)
    else:
        ret = np.full(size, fill_value, dtype=dtype)
    # repeated indexing gives last value, see:
    # the phrase "leaving behind the last value"  on this page:
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial
    ret[group_idx] = a
    return ret


def _first(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    if fill_value == 0:
        ret = np.zeros(size, dtype=dtype)
    else:
        ret = np.full(size, fill_value, dtype=dtype)
    ret[group_idx[::-1]] = a[::-1]  # same trick as _last, but in reverse
    return ret


def _prod(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype_scalar(fill_value, dtype, a)
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != 1:
        ret[group_idx] = 1  # product starts from 1
    np.multiply.at(ret, group_idx, a)
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
    print dtype
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
    if np.iscomplexobj(a):
        dtype = a.dtype  # TODO: this is a bit clumsy
        sums = np.empty(size, dtype=dtype)
        sums.real = np.bincount(group_idx, weights=a.real,
                                minlength=size)
        sums.imag = np.bincount(group_idx, weights=a.imag,
                                minlength=size)
    else:
        sums = np.bincount(group_idx, weights=a,
                           minlength=size).astype(dtype)

    with np.errstate(divide='ignore'):
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
    with np.errstate(divide='ignore'):
        means = sums.astype(dtype) / counts
        ret = np.bincount(group_idx, (a - means[group_idx]) ** 2,
                          minlength=size) / (counts - ddof)
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


def _generic_callable(group_idx, a, size, fill_value, dtype=None,
                      func=lambda g: g):
    """groups a by inds, and then applies foo to each group in turn, placing
    the results in an array."""
    groups = _array(group_idx, a, size, (), dtype=dtype)
    ret = np.full(size, fill_value, dtype=object)

    for i, grp in enumerate(groups):
        if np.ndim(grp) == 1 and len(grp) > 0:
            ret[i] = func(grp)
    return ret

_impl_dict = dict(min=_min, max=_max, sum=_sum, prod=_prod, last=_last,
                  first=_first, all=_all, any=_any, mean=_mean, std=_std,
                  var=_var, anynan=_anynan, allnan=_allnan, sort=_sort,
                  rsort=_rsort, array=_array, argmax=_argmax, argmin=_argmin,
                  len=_len)
_impl_dict.update(('nan' + k, v) for k, v in list(_impl_dict.items())
                  if k not in _no_separate_nan_version)


def _aggregate_base(group_idx, a, func='sum', size=None, fill_value=0,
                    order='C', dtype=None, axis=None, _impl_dict=_impl_dict,
                    _nansqueeze=False, **kwargs):
    group_idx, a, flat_size, ndim_idx, size = input_validation(group_idx, a,
                                             size=size, order=order, axis=axis)
    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        # do simple grouping and execute function in loop
        ret = _generic_callable(group_idx, a, flat_size, fill_value, func=func,
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
        func = _impl_dict[func]
        ret = func(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype,
                   **kwargs)

    # deal with ndimensional indexing
    if ndim_idx > 1:
        ret = ret.reshape(size, order=order)
    return ret


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    return _aggregate_base(group_idx, a, size=size, fill_value=fill_value,
                           order=order, dtype=dtype, func=func, axis=axis,
                           _impl_dict=_impl_dict, _nansqueeze=True, **kwargs)

aggregate.__doc__ = """
    This is the pure numpy implementation of aggregate.
    """ + _doc_str
