"""The averaging operations of the numpy implementation: the mean, the variance,
the standard deviation and the trapezoidal integration."""

from typing import cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.dtypes import iscomplexobj
from numpy_groupies.groups import exact_integer_group_mean
from numpy_groupies.npg_numpy.reductions import _first, _last, _sum
from numpy_groupies.typing import AScalar
from numpy_groupies.validation import (
    EXACT_FLOAT64_INTEGER_LIMIT,
    MOMENTS_NOT_FOR_COMPLEX,
    check_integer_mean_overflow,
    check_integer_variance_span_overflow,
    is_scalar_input,
)


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
        # the sums go through float64, which rounds once the integer sums
        # leave the +/-2**53 window - the limb sums answer the mean exactly
        # instead, in the dtype the caller asks for
        bound = max(int(a.max()), -int(a.min()))
        if a.size * bound > EXACT_FLOAT64_INTEGER_LIMIT:
            return cast(
                "npt.NDArray[np.inexact]",
                exact_integer_group_mean(
                    group_idx,
                    cast("npt.NDArray[np.integer]", a),
                    size,
                    fill_value,
                    np.dtype(dtype),
                ),
            )
    counts = np.bincount(group_idx, minlength=size)
    if iscomplexobj(a):
        # np.bincount takes no complex weights - the floating views of the real
        # and imaginary parts are accumulated separately instead
        sums = np.empty(size, dtype=dtype)
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

    With the constant sample spacing dx, the integral of a group collapses to
    dx * (sum(y) - (y[0] + y[n-1]) / 2), which spares the grouping sort and
    the pairwise products - the endpoints come from "first" and "last".

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _trapezoid(group_idx, a, np.max(group_idx) + 1)
    >>> array([ 0. , 30.5,  0. ,  8. , 14.5,  0. ,  0. ,  7.5,  0. ])
    """
    # complex samples integrate to complex values, real ones stay float64
    dtype = dtype or (a.dtype if iscomplexobj(a) else np.float64)
    # the helpers answer in the dtype of their own signature - the policy
    # above decides between the complex dtype of the samples and float64
    total = _sum(group_idx, a, size, 0, dtype=dtype)
    first = _first(group_idx, a, size, 0, dtype=dtype)
    last = _last(group_idx, a, size, 0, dtype=dtype)
    ret = (total - 0.5 * (first + last)) * dx

    counts = np.bincount(group_idx, minlength=size)
    if np.any(counts < 2):
        # without a whole pair there is no extent, the group integrates to zero
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
        # the squared magnitude |a|^2 - a real result like np.var, not the
        # complex a*a; a square is real, so bincount takes it as weight
        sq = a.real**2
        sq += a.imag**2
        ret = np.bincount(group_idx, weights=sq, minlength=size)
    elif np.dtype(dtype).kind in "iu":
        # the squares accumulate in the integral dtype the resolution chose -
        # the float64 weights of bincount would round and wrap on the way back
        ret = np.zeros(size, dtype=np.dtype(dtype))
        np.add.at(ret, group_idx, a.astype(np.dtype(dtype)) ** 2)
    else:
        # the square of the values, in the dtype of the input itself
        sq = a * a
        ret = np.bincount(group_idx, weights=cast("npt.NDArray[np.floating]", sq), minlength=size)
    if fill_value != 0:
        counts = np.bincount(group_idx, minlength=size)
        ret[counts == 0] = fill_value
    return ret.astype(dtype, copy=False)


def _central_moments(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """The counts and the sums of the squared, cubed and fourth powers of the
    deviations of every group from their mean, centered on the group's first
    value and re-centered on the mean - the moments of the raw values would
    cancel each other out.
    """
    counts = np.bincount(group_idx, minlength=size)
    origins = _first(group_idx, a, size, 0, dtype=a.dtype)
    # narrow-integer deviations wrap in their dtype (0 against an origin of
    # 3 is a 253 there) - the arithmetic runs in float64, where no value of
    # the input can lose its sign
    devs = a.astype(np.float64, copy=False) - origins[group_idx].astype(np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = (
            np.bincount(
                group_idx,
                weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating] | npt.NDArray[np.bool_]", devs),
                minlength=size,
            )
            / counts
        )
    devs = devs - means[group_idx]
    # bincount takes floats as weights, the signature only says generic -
    # numpy 2.4 types the result int, 2.5 float; floats either way.
    s2 = cast(
        "npt.NDArray[np.floating]",
        np.bincount(
            group_idx, weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating]", devs * devs), minlength=size
        ),
    )
    s3 = cast(
        "npt.NDArray[np.floating]",
        np.bincount(
            group_idx,
            weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating]", devs * devs * devs),
            minlength=size,
        ),
    )
    s4 = cast(
        "npt.NDArray[np.floating]",
        np.bincount(
            group_idx,
            weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating]", devs * devs * devs * devs),
            minlength=size,
        ),
    )
    return counts, s2, s3, s4


def _skew(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = np.dtype(np.float64),
) -> npt.NDArray[np.floating]:
    """The skewness of the values of every group, pandas-corrected for size."""
    if iscomplexobj(a):
        # the moments of the complex deviations are complex, their ratio
        # meaningless
        raise TypeError(MOMENTS_NOT_FOR_COMPLEX)
    if a.dtype.kind in "iu" and a.size:
        # the cubes leave the exact float64 window - refuse, not round
        bound = max(int(a.max()), -int(a.min()))
        check_integer_mean_overflow(a.size, bound)
    counts, s2, s3, _ = _central_moments(group_idx, a, size)
    present = counts > 0
    enough = present & (counts >= 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = counts.astype(np.float64)
        values = n * np.sqrt(n - 1) / (n - 2) * s3 / s2**1.5
    # a group without spread answers zero, pandas-corrected
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
    """The excess kurtosis of the values of every group, pandas-corrected."""
    if iscomplexobj(a):
        raise TypeError(MOMENTS_NOT_FOR_COMPLEX)
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
        # centered on the first value, the deviations stay exact in int64 -
        # a large common offset cannot erase small ones after the float64 cast
        low = int(a.min())
        high = int(a.max())
        if a.size * max(high, -low) > EXACT_FLOAT64_INTEGER_LIMIT:
            check_integer_variance_span_overflow(low, high)
            origins = _first(group_idx, a, size, 0, dtype=a.dtype)
            if a.dtype == np.dtype(np.uint64):
                # uint64 subtraction wraps modulo 2**64, the exact two's
                # complement deviation while |deviation| < 2**63 (the span
                # check above) - reinterpret the bits instead of a rounding cast
                a = (a - origins[group_idx]).astype(np.int64)
            else:
                a = a.astype(np.int64) - origins.astype(np.int64)[group_idx]
    counts = np.bincount(group_idx, minlength=size)
    if iscomplexobj(a):
        # np.bincount takes no complex weights - the floating views of the
        # real and imaginary parts are accumulated separately
        sums = np.bincount(group_idx, weights=cast("npt.NDArray[np.floating]", a.real), minlength=size)
        sums = sums + 1j * np.bincount(group_idx, weights=cast("npt.NDArray[np.floating]", a.imag), minlength=size)
    else:
        sums = np.bincount(
            group_idx,
            weights=cast("npt.NDArray[np.integer] | npt.NDArray[np.floating] | npt.NDArray[np.bool_]", a),
            minlength=size,
        )
    # no degrees of freedom - the group takes the fill value, like a
    # nan-poisoned one
    answerable = (counts > 0) & (counts > ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
        counts = np.where(answerable, counts - ddof, 0)
        devs = a - means[group_idx]
        if iscomplexobj(a):
            # |dev|^2 is real, so the result is real like np.var
            devs_sq = devs.real**2
            devs_sq += devs.imag**2
        else:
            devs_sq = devs
            devs_sq *= devs
        ret = np.bincount(group_idx, devs_sq, minlength=size) / counts
    if sqrt:
        ret = np.sqrt(ret)  # this is now std not var
    # no degrees of freedom: the division leaves an inf or nan behind, and
    # the fill is set after the square root, which would turn a negative one
    # into a nan
    ret[~answerable] = fill_value
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
