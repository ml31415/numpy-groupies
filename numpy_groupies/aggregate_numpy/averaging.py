"""The averaging operations of the numpy implementation.

The mean, the variance, the standard deviation and the trapezoidal
integration - the operations that combine the values of a group into a single
number of their own.
"""

from typing import cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.reductions import _first, _last, _sum
from numpy_groupies.common import (
    EXACT_FLOAT64_INTEGER_LIMIT,
    check_integer_mean_overflow,
    check_integer_variance_span_overflow,
    is_scalar_input,
)
from numpy_groupies.utils_np import AScalar, iscomplexobj


def _mean(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
) -> npt.NDArray[np.inexact]:
    if is_scalar_input(a):
        raise ValueError("cannot take mean with scalar a")
    if a.dtype.kind in "iu" and a.size:
        # the sums are accumulated in float64; once the exact integer sums can
        # leave the +/-2**53 window, the mean would silently round - refuse
        # instead (every backend raises for the same inputs)
        bound = max(int(a.max()), -int(a.min()))
        check_integer_mean_overflow(a.size, bound)
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


def _trapezoid(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
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
    fill_value: AScalar,
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


def _central_moments(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """The counts and the sums of the squared, cubed and fourth powers of the
    deviations of the values of every group from their mean.

    The mean comes first and the deviations from it second, as in the
    variance: the moments of the raw values would cancel each other out.
    """
    counts = np.bincount(group_idx, minlength=size)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = (
            np.bincount(
                group_idx,
                weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating] | npt.NDArray[np.bool_]", a),
                minlength=size,
            )
            / counts
        )
    devs = a - means[group_idx]
    # bincount takes the weights as floats - the signature says generic
    s2 = np.bincount(
        group_idx, weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating]", devs * devs), minlength=size
    )
    s3 = np.bincount(
        group_idx,
        weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating]", devs * devs * devs),
        minlength=size,
    )
    s4 = np.bincount(
        group_idx,
        weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating]", devs * devs * devs * devs),
        minlength=size,
    )
    return counts, s2, s3, s4


def _skew(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
) -> npt.NDArray[np.floating]:
    """The skewness of the values of every group.

    The skewness is the third central moment over the second one to the
    three-halves power, corrected for the size of the group the way pandas
    corrects it: a group of fewer than three values has no skewness, and a
    group whose values are all the same has none either - its spread is zero,
    and the answer is zero as well.
    """
    if iscomplexobj(a):
        # the skewness of a complex value is not defined - the moments of the
        # deviations would be complex, and their ratio meaningless
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    if a.dtype.kind in "iu" and a.size:
        # the cubes of the deviations leave the window float64 represents
        # exactly - refuse instead of answering an inexact number
        bound = max(int(a.max()), -int(a.min()))
        check_integer_mean_overflow(a.size, bound)
    counts, s2, s3, _ = _central_moments(group_idx, a, size)
    present = counts > 0
    enough = present & (counts >= 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = counts.astype(np.float64)
        values = n * np.sqrt(n - 1) / (n - 2) * s3 / s2**1.5
    # a group without spread has no skewness to speak of - pandas answers
    # zero for it, and so do we
    values = np.where(enough & (s2 == 0), 0.0, values)
    values = np.where(enough, values, np.nan)
    ret = np.full(size, fill_value, dtype=np.float64)
    ret[present] = values[present]
    return ret.astype(dtype, copy=False)


def _kurt(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
) -> npt.NDArray[np.floating]:
    """The excess kurtosis of the values of every group.

    The kurtosis is the fourth central moment over the square of the second
    one, minus three - the excess over the three of a normal distribution -
    corrected for the size of the group the way pandas corrects it: a group
    of fewer than four values has no kurtosis, and a group whose values are
    all the same has none either.
    """
    if iscomplexobj(a):
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    if a.dtype.kind in "iu" and a.size:
        bound = max(int(a.max()), -int(a.min()))
        check_integer_mean_overflow(a.size, bound)
    counts, s2, _, s4 = _central_moments(group_idx, a, size)
    present = counts > 0
    enough = present & (counts >= 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = counts.astype(np.float64)
        values = n * (n + 1) * (n - 1) * s4 / ((n - 2) * (n - 3) * s2**2) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    values = np.where(enough & (s2 == 0), 0.0, values)
    values = np.where(enough, values, np.nan)
    ret = np.full(size, fill_value, dtype=np.float64)
    ret[present] = values[present]
    return ret.astype(dtype, copy=False)


def _var(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
    sqrt: bool = False,
    ddof: int = 0,
) -> npt.NDArray[np.floating]:
    if is_scalar_input(a):
        raise ValueError("cannot take variance with scalar a")
    if a.dtype.kind in "iu" and a.size:
        # center each group on its first value: the deviations stay exact in
        # int64 arithmetic, so a large common offset can no longer erase small
        # deviations after the conversion to float64
        low = int(a.min())
        high = int(a.max())
        if a.size * max(high, -low) > EXACT_FLOAT64_INTEGER_LIMIT:
            check_integer_variance_span_overflow(low, high)
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
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
    ddof: int = 0,
) -> npt.NDArray[np.floating]:
    return _var(group_idx, a, size, fill_value, dtype=dtype, sqrt=True, ddof=ddof)
