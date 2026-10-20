from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_common import (
    DEFAULT_FILL_VALUE,
    EXACT_FLOAT64_INTEGER_LIMIT,
    INT64_DEVIATION_LIMIT,
    aggregate_common_doc,
    build_dispatch,
    custom_callable,
    funcs_no_separate_nan,
    missing_impl_error,
    nan_of,
    raise_integer_mean_overflow,
    raise_integer_variance_span_overflow,
    resolve_fill_value,
)
from numpy_groupies.utils import (
    AValue,
    RavelOrder,
    aliasing,
    check_boolean,
    check_dtype,
    check_fill_value,
    check_nton_shape,
    input_validation,
    iscomplexobj,
    maxval,
    minimum_dtype,
    minimum_dtype_scalar,
    minval,
)


def step_count(group_idx: npt.NDArray[np.integer]) -> int:
    """Return the number of contiguous runs in ``group_idx``.

    A strictly increasing array of length ``n`` has ``n`` runs, an empty
    array has 0, and an array of length 1 has 1. The return value matches
    ``aggregate_numba.step_count`` exactly, so the two backends can be used
    interchangeably.
    """
    group_idx = np.asarray(group_idx)
    if group_idx.size < 1:
        return 0
    return 1 + int(np.count_nonzero(np.diff(group_idx) != 0))


def step_indices(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """Return the run-edge indices in ``group_idx``.

    The first element is always 0, the last is always ``group_idx.size``,
    and the interior elements are the start indices of each new run. An
    empty input returns ``[0]`` to match the numba implementation.
    """
    group_idx = np.asarray(group_idx)
    n = group_idx.size
    if n < 1:
        return np.array([0], dtype=np.int64)
    starts = np.flatnonzero(np.diff(group_idx) != 0) + 1
    return np.concatenate(([0], starts, [n])).astype(np.int64, copy=False)


def _sum(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype_scalar(fill_value, dtype, a)

    if np.ndim(a) == 0:
        ret = np.bincount(group_idx, minlength=size).astype(dtype, copy=False)
        if a != 1:
            ret *= a
    else:
        exact_integer = False
        if a.dtype.kind in "iu" and np.dtype(dtype).kind in "iu":
            # the kind test above established an integer dtype
            limits = np.iinfo(cast("np.dtype[np.integer]", a.dtype))
            bound = max(abs(int(limits.min)), int(limits.max))
            if a.size * bound > 2**53:
                bound = max(abs(int(a.min(initial=0))), abs(int(a.max(initial=0))))
                exact_integer = a.size * bound > 2**53
        if iscomplexobj(a):
            ret = np.empty(size, dtype=dtype)
            ret.real = np.bincount(group_idx, weights=a.real, minlength=size)
            ret.imag = np.bincount(group_idx, weights=a.imag, minlength=size)
        elif exact_integer:
            # bincount converts weights to float64. Keep integer accumulation
            # when its partial sums are not guaranteed exactly representable.
            ret = np.zeros(size, dtype=dtype)
            np.add.at(ret, group_idx, a.astype(dtype, copy=False))
        else:
            # integer input, the kind test above guarantees it, the stubs want
            # the union of all bincount-legal dtypes
            ret = np.bincount(group_idx, weights=cast("npt.NDArray[np.integer]", a), minlength=size).astype(
                dtype, copy=False
            )

    if fill_value != 0:
        _fill_untouched(group_idx, ret, fill_value)
    return ret


def _prod(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype_scalar(fill_value, dtype, a)
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != 1:
        ret[group_idx] = 1  # product starts from 1
    np.multiply.at(ret, group_idx, a)
    return ret


def _len(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.int64]:
    # _sum folds a scalar column of ones into its ndim-0 branch
    return cast(
        "npt.NDArray[np.int64]",
        _sum(group_idx, cast("npt.NDArray[np.number]", 1), size, fill_value, dtype=int),
    )


def _last(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    # repeated indexing gives last value, see:
    # the phrase "leaving behind the last value"  on this page:
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial
    ret[group_idx] = a
    return ret


def _first(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    ret[group_idx[::-1]] = a[::-1]  # same trick as _last, but in reverse
    return ret


def _all(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.bool_],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.bool_]:
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if not fill_value:
        ret[group_idx] = True
    ret[group_idx.compress(np.logical_not(a))] = False
    return ret


def _any(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.bool_],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.bool_]:
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


def _min(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmax = maxval(fill_value, dtype)
    with np.errstate(invalid="ignore"):
        ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmax:
        ret[group_idx] = dmax  # min starts from maximum
    with np.errstate(invalid="ignore"):
        np.minimum.at(ret, group_idx, a)
    return ret


def _max(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmin = minval(fill_value, dtype)
    with np.errstate(invalid="ignore"):
        ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmin:
        ret[group_idx] = dmin  # max starts from minimum
    with np.errstate(invalid="ignore"):
        np.maximum.at(ret, group_idx, a)
    return ret


def _argmax(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = int,
    _nansqueeze: bool = False,
) -> npt.NDArray[np.int64]:
    # the mask value has to be the smallest value of numpy's order, which for a
    # complex dtype is -inf-infj and not -inf+0j (see minval and maxval)
    # Integer inputs have no NaNs; replacing them can promote uint64 to float64.
    a_ = np.where(np.isnan(a), minval(fill_value, a.dtype), a) if _nansqueeze and a.dtype.kind in "fc" else a
    group_max = _max(group_idx, a_, size, np.nan)
    # nan should never be maximum, so use a and not a_
    is_max = a == group_max[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_max = group_idx[is_max]
    (argmax,) = is_max.nonzero()
    ret[group_idx_max[::-1]] = argmax[::-1]  # reverse to ensure first value for each group wins
    return ret


def _argmin(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = int,
    _nansqueeze: bool = False,
) -> npt.NDArray[np.int64]:
    # the mask value has to be the largest value of numpy's order, which for a
    # complex dtype is inf+infj and not inf+0j (see minval and maxval)
    # Integer inputs have no NaNs; replacing them can promote uint64 to float64.
    a_ = np.where(np.isnan(a), maxval(fill_value, a.dtype), a) if _nansqueeze and a.dtype.kind in "fc" else a
    group_min = _min(group_idx, a_, size, np.nan)
    # nan should never be minimum, so use a and not a_
    is_min = a == group_min[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_min = group_idx[is_min]
    (argmin,) = is_min.nonzero()
    ret[group_idx_min[::-1]] = argmin[::-1]  # reverse to ensure first value for each group wins
    return ret


def _mean(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
) -> npt.NDArray[np.inexact]:
    if np.ndim(a) == 0:
        raise ValueError("cannot take mean with scalar a")
    if a.dtype.kind in "iu" and a.size:
        # the sums are accumulated in float64; once the exact integer sums can
        # leave the +/-2**53 window, the mean would silently round - refuse
        # instead (every backend raises for the same inputs)
        bound = max(int(a.max()), -int(a.min()))
        if a.size * bound > EXACT_FLOAT64_INTEGER_LIMIT:
            raise_integer_mean_overflow(a.size, bound)
    counts = np.bincount(group_idx, minlength=size)
    if iscomplexobj(a):
        # np.bincount does not take complex weights - accumulate the real and
        # imaginary parts separately
        sums = np.empty(size, dtype=dtype)
        # the parts of a complex column are floating views, which is what
        # bincount takes as weights - the signature only says generic
        sums.real = np.bincount(group_idx, weights=cast("npt.NDArray[np.floating]", a.real), minlength=size)
        sums.imag = np.bincount(group_idx, weights=cast("npt.NDArray[np.floating]", a.imag), minlength=size)
    else:
        sums = np.bincount(
            group_idx,
            weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating] | npt.NDArray[np.bool_]", a),
            minlength=size,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = sums / counts
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    return ret.astype(dtype, copy=False)


def _median(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.inexact]:
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
        return np.full(size, fill_value, dtype=dtype or np.result_type(a, np.float64))
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
    # obtained with a single call using both kths; np.partition orders
    # complex values lexicographically, so the dtype is kept
    vals = np.empty(starts.size, dtype=np.result_type(a_srt, np.float64))
    # the mean of the two middles is taken in float64 for integral input -
    # adding them in their own dtype overflows (int8 100 + 100) and bool
    # addition is a logical or (True + True), while np.median promotes first
    promote = np.issubdtype(a_srt.dtype, np.integer) or np.issubdtype(a_srt.dtype, np.bool_)
    for grp in range(starts.size):
        start, mid = starts[grp], mids[grp]
        part = np.partition(a_srt[start : start + counts[grp]], (mid - 1, mid))
        if odd[grp]:
            vals[grp] = part[mid]
        elif promote:
            vals[grp] = (part[mid - 1].astype(np.float64) + part[mid]) / 2
        else:
            vals[grp] = (part[mid - 1] + part[mid]) / 2
    if np.issubdtype(a_srt.dtype, np.inexact):
        # like np.median, any nan poisons its whole group - partition does
        # not order nans, so they are detected with a per-group reduction, and
        # the poison is shaped like the result rather than written in as a plain
        # nan, which for a complex group would leave its other part at a zero
        nan_groups = np.logical_or.reduceat(np.isnan(a_srt), starts)
        vals[nan_groups] = nan_of(vals.dtype)

    ret = np.full(size, fill_value, dtype=dtype or vals.dtype)
    ret[group_idx_srt[starts]] = vals
    return ret


def _trapezoid(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
    dx: float = 1.0,
) -> npt.NDArray[np.number]:
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
    # complex samples integrate to complex values, real ones stay float64
    dtype = dtype or (a.dtype if iscomplexobj(a) else np.float64)
    # the helpers answer with the numeric dtype of their own signature, which
    # is all the arithmetic here needs - the policy above decides between the
    # complex dtype of the samples and float64, this function promises number
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


def _sum_of_squres(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
) -> npt.NDArray[np.floating]:
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(dtype, copy=False)
    if iscomplexobj(a):
        # the squared magnitude |a|^2 - the power of complex values, giving
        # a real result like np.var, instead of the complex a*a
        sq = a.real**2
        sq += a.imag**2
    else:
        # the square of the values, in the dtype of the input itself
        sq = a * a
    # the square of a real or complex value is real, bincount takes it
    ret = np.bincount(group_idx, weights=sq, minlength=size)
    if fill_value != 0:
        counts = np.bincount(group_idx, minlength=size)
        ret[counts == 0] = fill_value
    return ret.astype(dtype, copy=False)


def _var(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
    sqrt: bool = False,
    ddof: int = 0,
) -> npt.NDArray[np.floating]:
    if np.ndim(a) == 0:
        raise ValueError("cannot take variance with scalar a")
    if a.dtype.kind in "iu" and a.size:
        # center each group on its first value: the deviations stay exact in
        # int64 arithmetic, so a large common offset can no longer erase small
        # deviations after the conversion to float64
        low = int(a.min())
        high = int(a.max())
        if a.size * max(high, -low) > EXACT_FLOAT64_INTEGER_LIMIT:
            if high - low > INT64_DEVIATION_LIMIT:
                raise_integer_variance_span_overflow(low, high)
            origins = _first(group_idx, a, size, 0, dtype=a.dtype)
            if a.dtype == np.dtype(np.uint64):
                # uint64 subtraction wraps modulo 2**64, which is the exact
                # two's-complement deviation while |deviation| stays below
                # 2**63 (guaranteed by the span check above) - reinterpret the
                # bit pattern instead of letting a cast round the values
                a = (a - origins[group_idx]).astype(np.int64)
            else:
                a = a.astype(np.int64) - origins.astype(np.int64)[group_idx]
    counts = np.bincount(group_idx, minlength=size)
    if iscomplexobj(a):
        # np.bincount does not take complex weights - accumulate the real and
        # imaginary parts separately
        # the parts of a complex column are floating views, what bincount takes
        sums = np.bincount(group_idx, weights=cast("npt.NDArray[np.floating]", a.real), minlength=size)
        sums = sums + 1j * np.bincount(group_idx, weights=cast("npt.NDArray[np.floating]", a.imag), minlength=size)
    else:
        sums = np.bincount(
            group_idx,
            weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating] | npt.NDArray[np.bool_]", a),
            minlength=size,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
        counts = np.where(counts > ddof, counts - ddof, 0)
        devs = a - means[group_idx]
        if iscomplexobj(a):
            # the magnitude squared of the complex deviation is real, so
            # like np.var the result is real
            devs_sq = devs.real**2
            devs_sq += devs.imag**2
        else:
            devs_sq = devs
            devs_sq *= devs
        ret = np.bincount(group_idx, devs_sq, minlength=size) / counts
    if sqrt:
        ret = np.sqrt(ret)  # this is now std not var
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    return ret.astype(dtype, copy=False)


def _std(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
    ddof: int = 0,
) -> npt.NDArray[np.floating]:
    return _var(group_idx, a, size, fill_value, dtype=dtype, sqrt=True, ddof=ddof)


def _allnan(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = bool,
) -> npt.NDArray[np.bool_]:
    return _all(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)


def _anynan(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = bool,
) -> npt.NDArray[np.bool_]:
    return _any(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)


def _sort(
    group_idx: npt.NDArray[np.integer],
    a: Any,
    size: int | tuple[int, ...] | None = None,
    fill_value: Any = None,
    dtype: npt.DTypeLike | None = None,
    reverse: bool = False,
) -> npt.NDArray[np.number]:
    if reverse:
        # the descending key.  Bitwise inversion reverses the order of the
        # integral and boolean dtypes exactly, where negation does not: the
        # smallest value of a signed dtype survives negation as itself, and a
        # zero of an unsigned one stays the smallest key of all.  The inexact
        # dtypes have no inversion and negate cleanly - a nan stays last
        # either way, and -0.0 compares equal to 0.0.
        key = cast("npt.NDArray[np.integer]", ~a) if a.dtype.kind in "iub" else cast("npt.NDArray[np.floating]", -a)
    else:
        key = a
    sortidx = np.lexsort((key, group_idx))
    # Unsort back into original order, but preserving the groupwise value
    # sorting: scattering through the group-stable argsort is exactly the
    # inverse permutation, avoiding the classic argsort-of-argsort.
    gsort = np.argsort(group_idx, kind="stable")
    ret = cast("npt.NDArray[np.number]", np.empty_like(a))
    ret[gsort] = a[sortidx]
    return ret


def blocks_to_object_array(blocks: Iterable[npt.NDArray[np.number]]) -> npt.NDArray[np.object_]:
    """The blocks of the groups as an object array, one sequence per group.

    The array is built element by element: ``np.asanyarray`` of a list of
    blocks collapses it into a regular array whenever the groups happen to be
    of the same length - and the result would lose the sequences it is meant
    to hold (and the size the caller asked for).
    """
    blocks = list(blocks)
    ret = np.empty(len(blocks), dtype=object)
    for g, block in enumerate(blocks):
        ret[g] = block
    return ret


def _array(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.object_]:
    """groups a into separate arrays, keeping the order intact."""
    if fill_value is not None and not (np.isscalar(fill_value) or len(fill_value) == 0):
        raise ValueError("fill_value must be None, a scalar or an empty sequence")
    order_group_idx = np.argsort(group_idx, kind="stable")
    counts = np.bincount(group_idx, minlength=size)
    blocks = np.split(a[order_group_idx], np.cumsum(counts)[:-1])
    ret = blocks_to_object_array(blocks)
    if fill_value is None or np.isscalar(fill_value):
        _fill_untouched(group_idx, ret, fill_value)
    return ret


def _generic_callable(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any,
    dtype: npt.DTypeLike | None = None,
    func: Callable[..., Any] = lambda g: g,
    **kwargs: Any,
) -> npt.NDArray[np.number]:
    """groups a by inds, and then applies foo to each group in turn, placing
    the results in an array."""
    groups = _array(group_idx, a, size, ())
    ret = np.full(size, fill_value, dtype=dtype or np.float64)

    for i, grp in enumerate(groups):
        if np.ndim(grp) == 1 and len(grp) > 0:
            ret[i] = func(grp)
    return ret


def _group_cumsum_sorted(
    group_idx_srt: npt.NDArray[np.integer], a_srt: npt.NDArray[np.number], dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.number]:
    """
    Cumsum within each contiguous group, with independent accumulators.

    Subtracting offsets from a global cumsum loses small increments after
    large groups, and nonfinite values contaminate subsequent groups.
    """
    new_group = np.empty(group_idx_srt.size, dtype=bool)
    new_group[0] = True
    np.not_equal(group_idx_srt[1:], group_idx_srt[:-1], out=new_group[1:])
    starts = np.flatnonzero(new_group)
    ends = np.append(starts[1:], group_idx_srt.size)
    # Match cumsum's default integer promotion, including for singleton groups.
    result_dtype = np.cumsum(a_srt[:0], dtype=dtype).dtype
    a_srt_cumsum = a_srt.astype(result_dtype, copy=True)
    counts = ends - starts
    # Batch equal-length groups into rows, keeping every accumulator separate
    # without a Python call for each short group. Singleton groups are done.
    for count in np.unique(counts[counts > 1]):
        selected = starts[counts == count]
        if selected.size == starts.size:
            np.cumsum(a_srt.reshape(-1, count), axis=1, dtype=dtype, out=a_srt_cumsum.reshape(-1, count))
        elif selected.size == 1:
            start = selected[0]
            np.cumsum(a_srt[start : start + count], dtype=dtype, out=a_srt_cumsum[start : start + count])
        else:
            positions = selected[:, None] + np.arange(count)
            a_srt_cumsum[positions] = np.cumsum(a_srt[positions], axis=1, dtype=dtype)
    return a_srt_cumsum


def _cumsum(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
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

    a_srt_cumsum = _group_cumsum_sorted(group_idx_srt, a_srt, dtype=dtype)

    ret = np.empty_like(a_srt_cumsum)
    ret[sortidx] = a_srt_cumsum
    return ret


def _nancumsum(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Any = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    a_nonans = np.where(np.isnan(a), 0, a)
    group_idx_nonans = np.where(np.isnan(group_idx), np.nanmax(group_idx) + 1, group_idx)
    return _cumsum(group_idx_nonans, a_nonans, size, fill_value=fill_value, dtype=dtype)


_impl_dict: dict[str, Callable[..., Any]] = {
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

_dispatch = build_dispatch(_impl_dict, aliasing)


def _aggregate_base(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: str | Callable[..., Any] | None = "sum",
    size: int | Sequence[int] | npt.NDArray[np.integer] | None = None,
    fill_value: Any = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    _impl_dict: dict[str, Callable[..., Any]] = _impl_dict,
    _dispatch: Mapping[Any, tuple[str, Callable[..., Any] | None]] = _dispatch,
    is_pandas: bool = False,
    **kwargs: Any,
) -> npt.NDArray[np.number]:
    iv = input_validation(group_idx, a, size=size, order=order, axis=axis, func=func)
    group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

    if group_idx.dtype == np.dtype("uint64"):
        # Force conversion to signed int, to avoid issues with bincount etc later
        group_idx = group_idx.astype(int)

    try:
        funcname, func = _dispatch[func]
    except (KeyError, TypeError):
        # no key of the table, either because no function goes by that name or
        # because func could not be one of its keys (a dict asks its keys for a
        # hash, which an object defining __eq__ without __hash__ does not have):
        # both leave a callable of the caller's own to run, and anything else
        # says that it named nothing
        func = custom_callable(func)
        funcname = cast("Any", func)
        fill_value = resolve_fill_value(func, fill_value, dtype if dtype is not None else np.asarray(a).dtype)
        # do simple grouping and execute function in loop
        ret = _impl_dict.get("generic", _generic_callable)(
            group_idx, a, flat_size, fill_value, func=func, dtype=dtype, **kwargs
        )
        if ndim_idx > 1:
            check_nton_shape(ret, size, funcname)
            ret = ret.reshape(size, order=order)
        return cast("npt.NDArray[np.number]", ret)
    else:
        if func is None:
            # a function that is known but not implemented here
            raise missing_impl_error(funcname)
        # deal with nans and find the function
        if funcname.startswith("nan"):
            if np.ndim(a) == 0:
                raise ValueError("nan-version not supported for scalar input.")
            if "nan" in funcname:
                if "arg" in funcname:
                    kwargs["_nansqueeze"] = True
                elif "cum" in funcname:
                    pass
                else:
                    good = ~np.isnan(a)
                    if "len" not in funcname or is_pandas:
                        # a is not needed for len, nanlen!
                        # a scalar a raised a branch above, only columns remain
                        a = cast("npt.NDArray[np.number]", a)[good]
                    group_idx = group_idx[good]

        # the sum overflow guess needs the maximum number of values that can
        # be summed into a single group, i.e. the (nan-filtered) input length
        # - flat_size counts output groups and can be arbitrarily smaller
        dtype = check_dtype(dtype, funcname, a, len(group_idx))
        fill_value = resolve_fill_value(funcname, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=funcname)
        ret = func(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype, **kwargs)

    # deal with ndimensional indexing
    if ndim_idx > 1:
        if unravel_shape is not None:
            # A negative fill_value cannot, and should not, be unraveled.
            mask = ret == fill_value
            ret[mask] = 0
            ret = cast(
                "npt.NDArray[np.number]",
                np.unravel_index(ret, unravel_shape)[cast("int", axis)],
            )
            ret[mask] = fill_value
        check_nton_shape(ret, size, funcname)
        ret = ret.reshape(size, order=order)
    return cast("npt.NDArray[np.number]", ret)


def aggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: str | Callable[..., Any] | None = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: Any = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.number]:
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


def _fill_untouched(idx: npt.NDArray[np.integer], ret: npt.NDArray[np.generic], fill_value: Any) -> None:
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value
