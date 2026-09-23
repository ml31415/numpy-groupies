import numpy as np

from .utils import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    aliasing,
    check_boolean,
    check_dtype,
    check_fill_value,
    check_nton_shape,
    funcs_no_separate_nan,
    get_func,
    input_validation,
    iscomplexobj,
    maxval,
    minimum_dtype,
    minimum_dtype_scalar,
    minval,
    resolve_fill_value,
)


def _sum(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype_scalar(fill_value, dtype, a)

    if np.ndim(a) == 0:
        ret = np.bincount(group_idx, minlength=size).astype(dtype, copy=False)
        if a != 1:
            ret *= a
    else:
        if iscomplexobj(a):
            ret = np.empty(size, dtype=dtype)
            ret.real = np.bincount(group_idx, weights=a.real, minlength=size)
            ret.imag = np.bincount(group_idx, weights=a.imag, minlength=size)
        else:
            ret = np.bincount(group_idx, weights=a, minlength=size).astype(dtype, copy=False)

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
    # convert to bool explicitly - ndarray.compress on a float mask is slow
    mask = np.asarray(a, dtype=bool)
    # numpy quirk: compress wins for sparse masks, fancy indexing for dense
    # ones (up to ~3x either way) - so pick based on mask density
    if np.count_nonzero(mask) < mask.size / 2:
        ret[group_idx.compress(mask)] = True
    else:
        ret[group_idx[mask]] = True
    return ret


def _min(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmax = maxval(fill_value, dtype)
    with np.errstate(invalid="ignore"):
        ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmax:
        ret[group_idx] = dmax  # min starts from maximum
    with np.errstate(invalid="ignore"):
        np.minimum.at(ret, group_idx, a)
    return ret


def _max(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmin = minval(fill_value, dtype)
    with np.errstate(invalid="ignore"):
        ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmin:
        ret[group_idx] = dmin  # max starts from minimum
    with np.errstate(invalid="ignore"):
        np.maximum.at(ret, group_idx, a)
    return ret


def _argmax(group_idx, a, size, fill_value, dtype=int, _nansqueeze=False):
    a_ = np.where(np.isnan(a), -np.inf, a) if _nansqueeze else a
    group_max = _max(group_idx, a_, size, np.nan)
    # nan should never be maximum, so use a and not a_
    is_max = a == group_max[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_max = group_idx[is_max]
    (argmax,) = is_max.nonzero()
    ret[group_idx_max[::-1]] = argmax[::-1]  # reverse to ensure first value for each group wins
    return ret


def _argmin(group_idx, a, size, fill_value, dtype=int, _nansqueeze=False):
    a_ = np.where(np.isnan(a), np.inf, a) if _nansqueeze else a
    group_min = _min(group_idx, a_, size, np.nan)
    # nan should never be minimum, so use a and not a_
    is_min = a == group_min[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_min = group_idx[is_min]
    (argmin,) = is_min.nonzero()
    ret[group_idx_min[::-1]] = argmin[::-1]  # reverse to ensure first value for each group wins
    return ret


def _mean(group_idx, a, size, fill_value, dtype=np.dtype(np.float64)):
    if np.ndim(a) == 0:
        raise ValueError("cannot take mean with scalar a")
    counts = np.bincount(group_idx, minlength=size)
    if iscomplexobj(a):
        dtype = a.dtype  # TODO: this is a bit clumsy
        sums = np.empty(size, dtype=dtype)
        sums.real = np.bincount(group_idx, weights=a.real, minlength=size)
        sums.imag = np.bincount(group_idx, weights=a.imag, minlength=size)
    else:
        sums = np.bincount(group_idx, weights=a, minlength=size)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = sums / counts
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    if iscomplexobj(a):
        return ret
    else:
        return ret.astype(dtype, copy=False)


def _median(group_idx, a, size, fill_value, dtype=None):
    """
    Aggregate operation of the median within each group.

    The median is an order statistic, so unlike the streaming reductions it
    works on group-ordered data: the values are gathered group by group and
    the middle of each group is then *selected* via partition (O(n) on
    average) instead of sorting the values within the groups.

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _median(group_idx, a, np.max(group_idx) + 1)
    >>> array([0. , 8. , 4.5, 3. , 0. , 0. , 0. , 7.5, 0. ])
    """
    if group_idx.size == 0:
        return np.full(size, fill_value, dtype=dtype or np.float64)
    # any argsort kind works - the median is insensitive to the order of
    # equal group labels
    sortidx = np.argsort(group_idx, kind="stable")
    group_idx_srt = group_idx[sortidx]
    a_srt = a[sortidx]

    new_group = np.empty(group_idx_srt.size, dtype=bool)
    new_group[0] = True
    np.not_equal(group_idx_srt[1:], group_idx_srt[:-1], out=new_group[1:])
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.append(starts, group_idx_srt.size))
    mids = counts // 2
    odd = counts % 2 == 1

    # partition places the kth element(s) at their sorted position while
    # leaving the rest unsorted - the two middles of even-sized groups are
    # obtained with a single call using both kths
    vals = np.empty(starts.size, dtype=np.float64)
    for grp in range(starts.size):
        start, mid = starts[grp], mids[grp]
        part = np.partition(a_srt[start : start + counts[grp]], (mid - 1, mid))
        if odd[grp]:
            vals[grp] = part[mid]
        else:
            vals[grp] = (part[mid - 1] + part[mid]) / 2
    if np.issubdtype(a_srt.dtype, np.floating):
        # like np.median, any nan poisons its whole group - partition does
        # not order nans, so they are detected with a per-group reduction
        nan_groups = np.logical_or.reduceat(np.isnan(a_srt), starts)
        vals[nan_groups] = np.nan

    ret = np.full(size, fill_value, dtype=dtype or vals.dtype)
    ret[group_idx_srt[starts]] = vals
    return ret


def _trapezoid(group_idx, a, size, fill_value, dtype=None, dx=1.0):
    """
    Trapezoidal integration of each group, keeping the order of the input.

    With the constant sample spacing dx, the integral of a group of n
    samples collapses to its sum minus the half weighted endpoints:

        sum_{k=1..n-1} dx / 2 * (y[k-1] + y[k])
        = dx * (sum(y) - (y[0] + y[n-1]) / 2)

    which spares both the grouping sort and the pairwise products - the
    order sensitive endpoints are exactly what the order sensitive "first"
    and "last" reductions deliver.

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _trapezoid(group_idx, a, np.max(group_idx) + 1)
    >>> array([ 0. , 30.5,  0. ,  8. , 14.5,  0. ,  0. ,  7.5,  0. ])
    """
    dtype = dtype or np.float64
    total = _sum(group_idx, a, size, 0, dtype=dtype)
    first = _first(group_idx, a, size, 0, dtype=dtype)
    last = _last(group_idx, a, size, 0, dtype=dtype)
    ret = (total - 0.5 * (first + last)) * dx

    counts = np.bincount(group_idx, minlength=size)
    if np.any(counts < 2):
        # without a whole pair there is no extent to integrate over, so the
        # group integrates to zero rather than to whatever sum - y - y is
        ret[counts < 2] = 0.0
    if np.any(counts == 0):
        ret[counts == 0] = fill_value
    return ret


def _sum_of_squres(group_idx, a, size, fill_value, dtype=np.dtype(np.float64)):
    ret = np.bincount(group_idx, weights=a * a, minlength=size)
    if fill_value != 0:
        counts = np.bincount(group_idx, minlength=size)
        ret[counts == 0] = fill_value
    if iscomplexobj(a):
        return ret
    else:
        return ret.astype(dtype, copy=False)


def _var(group_idx, a, size, fill_value, dtype=np.dtype(np.float64), sqrt=False, ddof=0):
    if np.ndim(a) == 0:
        raise ValueError("cannot take variance with scalar a")
    counts = np.bincount(group_idx, minlength=size)
    sums = np.bincount(group_idx, weights=a, minlength=size)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
        counts = np.where(counts > ddof, counts - ddof, 0)
        ret = np.bincount(group_idx, (a - means[group_idx]) ** 2, minlength=size) / counts
    if sqrt:
        ret = np.sqrt(ret)  # this is now std not var
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    if iscomplexobj(a):
        return ret
    else:
        return ret.astype(dtype, copy=False)


def _std(group_idx, a, size, fill_value, dtype=np.dtype(np.float64), ddof=0):
    return _var(group_idx, a, size, fill_value, dtype=dtype, sqrt=True, ddof=ddof)


def _allnan(group_idx, a, size, fill_value, dtype=bool):
    return _all(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)


def _anynan(group_idx, a, size, fill_value, dtype=bool):
    return _any(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)


def _sort(group_idx, a, size=None, fill_value=None, dtype=None, reverse=False):
    sortidx = np.lexsort((-a if reverse else a, group_idx))
    # Unsort back into original order, but preserving the groupwise value
    # sorting: scattering through the group-stable argsort is exactly the
    # inverse permutation, avoiding the classic argsort-of-argsort.
    gsort = np.argsort(group_idx, kind="stable")
    ret = np.empty_like(a)
    ret[gsort] = a[sortidx]
    return ret


def _array(group_idx, a, size, fill_value, dtype=None):
    """groups a into separate arrays, keeping the order intact."""
    if fill_value is not None and not (np.isscalar(fill_value) or len(fill_value) == 0):
        raise ValueError("fill_value must be None, a scalar or an empty sequence")
    order_group_idx = np.argsort(group_idx, kind="stable")
    counts = np.bincount(group_idx, minlength=size)
    ret = np.split(a[order_group_idx], np.cumsum(counts)[:-1])
    ret = np.asanyarray(ret, dtype="object")
    if fill_value is None or np.isscalar(fill_value):
        _fill_untouched(group_idx, ret, fill_value)
    return ret


def _generic_callable(group_idx, a, size, fill_value, dtype=None, func=lambda g: g, **kwargs):
    """groups a by inds, and then applies foo to each group in turn, placing
    the results in an array."""
    groups = _array(group_idx, a, size, ())
    ret = np.full(size, fill_value, dtype=dtype or np.float64)

    for i, grp in enumerate(groups):
        if np.ndim(grp) == 1 and len(grp) > 0:
            ret[i] = func(grp)
    return ret


def _group_cumsum_sorted(group_idx_srt, a_srt, dtype=None):
    """
    Cumsum within each group of a group-sorted nan-free array.

    group_idx_srt is sorted, so each group occupies one contiguous block and
    the cumulative sum of every group simply needs offsetting by the value
    of its first element (minus the preceding groups' sums, which the
    global cumsum already includes).
    """
    a_srt_cumsum = np.cumsum(a_srt, dtype=dtype)

    new_group = np.empty(group_idx_srt.size, dtype=bool)
    new_group[0] = True
    np.not_equal(group_idx_srt[1:], group_idx_srt[:-1], out=new_group[1:])
    start_positions = np.flatnonzero(new_group)
    group_starts = start_positions[np.cumsum(new_group) - 1]
    # First subtract large numbers
    a_srt_cumsum -= a_srt_cumsum[group_starts]
    # Then add potentially small numbers
    a_srt_cumsum += a_srt[group_starts]
    return a_srt_cumsum


def _cumsum(group_idx, a, size, fill_value=None, dtype=None):
    """
    N to N aggregate operation of cumsum. Perform cumulative sum for each group.

    NaNs propagate within their own group only - entries of other groups as
    well as entries preceding the first NaN of the same group are unaffected
    (issue #91).

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _cumsum(group_idx, a, np.max(group_idx) + 1)
    >>> array([ 3,  4,  5,  6, 15,  9, 15, 22,  7,  0, 15, 17,  6, 14, 31, 39])
    """
    sortidx = np.argsort(group_idx, kind="stable")
    group_idx_srt = group_idx[sortidx]
    a_srt = a[sortidx]

    nans_from = None
    if np.issubdtype(a_srt.dtype, np.floating):
        nans = np.isnan(a_srt)
        if nans.any():
            # nan-free running sums cannot be poisoned across group
            # boundaries; the per-group prefix-count of nans (itself a
            # nan-free cumsum) flags every position from the first nan of
            # its group onwards.
            a_srt = np.where(nans, 0, a_srt)
            nan_prefix = _group_cumsum_sorted(group_idx_srt, nans.astype(np.int64))
            nans_from = nan_prefix > 0

    a_srt_cumsum = _group_cumsum_sorted(group_idx_srt, a_srt, dtype=dtype)
    if nans_from is not None:
        a_srt_cumsum[nans_from] = np.nan

    ret = np.empty_like(a_srt_cumsum)
    ret[sortidx] = a_srt_cumsum
    return ret


def _nancumsum(group_idx, a, size, fill_value=None, dtype=None):
    a_nonans = np.where(np.isnan(a), 0, a)
    group_idx_nonans = np.where(np.isnan(group_idx), np.nanmax(group_idx) + 1, group_idx)
    return _cumsum(group_idx_nonans, a_nonans, size, fill_value=fill_value, dtype=dtype)


_impl_dict = {
    "min": _min,
    "max": _max,
    "sum": _sum,
    "prod": _prod,
    "last": _last,
    "first": _first,
    "all": _all,
    "any": _any,
    "mean": _mean,
    "median": _median,
    "std": _std,
    "var": _var,
    "anynan": _anynan,
    "allnan": _allnan,
    "sort": _sort,
    "array": _array,
    "argmax": _argmax,
    "argmin": _argmin,
    "len": _len,
    "cumsum": _cumsum,
    "sumofsquares": _sum_of_squres,
    "trapezoid": _trapezoid,
    "generic": _generic_callable,
}
_impl_dict.update(("nan" + k, v) for k, v in list(_impl_dict.items()) if k not in funcs_no_separate_nan)
_impl_dict["nancumsum"] = _nancumsum


def _aggregate_base(
    group_idx,
    a,
    func="sum",
    size=None,
    fill_value=DEFAULT_FILL_VALUE,
    order="C",
    dtype=None,
    axis=None,
    _impl_dict=_impl_dict,
    is_pandas=False,
    **kwargs,
):
    iv = input_validation(group_idx, a, size=size, order=order, axis=axis, func=func)
    group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

    if group_idx.dtype == np.dtype("uint64"):
        # Force conversion to signed int, to avoid issues with bincount etc later
        group_idx = group_idx.astype(int)

    func = get_func(func, aliasing, _impl_dict)
    funcname = func
    if not isinstance(func, str):
        fill_value = resolve_fill_value(func, fill_value, dtype if dtype is not None else np.asarray(a).dtype)
        # do simple grouping and execute function in loop
        ret = _impl_dict.get("generic", _generic_callable)(
            group_idx, a, flat_size, fill_value, func=func, dtype=dtype, **kwargs
        )
    else:
        # deal with nans and find the function
        if func.startswith("nan"):
            if np.ndim(a) == 0:
                raise ValueError("nan-version not supported for scalar input.")
            if "nan" in func:
                if "arg" in func:
                    kwargs["_nansqueeze"] = True
                elif "cum" in func:
                    pass
                else:
                    good = ~np.isnan(a)
                    if "len" not in func or is_pandas:
                        # a is not needed for len, nanlen!
                        a = a[good]
                    group_idx = group_idx[good]

        dtype = check_dtype(dtype, func, a, flat_size)
        fill_value = resolve_fill_value(func, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=func)
        funcname, func = func, _impl_dict[func]
        ret = func(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype, **kwargs)

    # deal with ndimensional indexing
    if ndim_idx > 1:
        if unravel_shape is not None:
            # A negative fill_value cannot, and should not, be unraveled.
            mask = ret == fill_value
            ret[mask] = 0
            ret = np.unravel_index(ret, unravel_shape)[axis]
            ret[mask] = fill_value
        check_nton_shape(ret, size, funcname)
        ret = ret.reshape(size, order=order)
    return ret


def aggregate(
    group_idx,
    a,
    func="sum",
    size=None,
    fill_value=DEFAULT_FILL_VALUE,
    order="C",
    dtype=None,
    axis=None,
    **kwargs,
):
    return _aggregate_base(
        group_idx,
        a,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        func=func,
        axis=axis,
        _impl_dict=_impl_dict,
        **kwargs,
    )


aggregate.__doc__ = (
    """
    This is the pure numpy implementation of aggregate.
    """
    + aggregate_common_doc
)


def _fill_untouched(idx, ret, fill_value):
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value
