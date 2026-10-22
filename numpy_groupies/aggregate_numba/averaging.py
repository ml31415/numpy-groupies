"""Two-pass averaging: mean, std and var, including the integer exactness policy.

Integer input is centered on the first value of each group and the global
min/max is tracked in the same accumulation pass; the exactness gates run
between the passes and raise ``OverflowError`` for the same inputs in every
backend (see ``common`` for the policy constants).
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.compose import build_kernel, rule_float, rule_real_float
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.validation import (
    EXACT_FLOAT64_INTEGER_LIMIT,
    INT64_DEVIATION_LIMIT,
    INT64_MIN,
    mean_overflow_message,
    variance_span_overflow_message,
)


# the building blocks and overload shims of this file are jitted: numba
# compiles them from the argument types and ignores these annotations,
# the type checker needs them.  One exception the checker does not see:
# numba compares the *signature* of an ``@overload`` function with that of the
# implementation it returns, annotations included, so the shim, its overload
# and every one of its alternative implementations must be spelled alike -
# a lone ``int`` where the overload says ``float`` is refused as
# "Typing and implementation arguments differ in argument names" and the
# integer path of var/std/mean then fails to compile
def _track_extrema(val: float, low: float, high: float) -> None:
    """Fold the global min/max tracking for the integer limit checks into the
    accumulation pass - replaced per dtype by the overload below, inexact
    input never tracks anything."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_track_extrema)
def _track_extrema_overload(val: float, low: float, high: float) -> Callable[..., None]:
    if isinstance(val, nb.types.Integer):

        def _impl(val: float, low: float, high: float) -> None:
            low[0] = min(low[0], val)
            high[0] = max(high[0], val)

        return _impl

    def _impl(val: float, low: float, high: float) -> None:  # type: ignore[no-redef]
        pass

    return _impl


@njit_stable
def _mean_overflow_message(count: int, bound: int) -> str:
    """The numba twin of ``check_integer_mean_overflow`` - numba cannot call
    the python helper, but it can compile the message builder of
    ``validation`` (a concatenation of strings and numbers), so the text
    exists once."""
    return mean_overflow_message(count, bound)


@njit_stable
def _span_overflow_message(low: int, high: int) -> str:
    """The numba twin of ``check_integer_variance_span_overflow``."""
    # numba's str() negates int64-min internally, wraps, and then fails to
    # size the string buffer - spell that one value out
    low_text = "-9223372036854775808" if low == INT64_MIN else str(low)
    return variance_span_overflow_message(low_text, high)


def _mean_limit_check(low: float, high: float, count: int) -> None:
    """Refuse integer means float64 cannot compute exactly - replaced per
    dtype by the overload below, a no-op for inexact input."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_mean_limit_check)
def _mean_limit_check_overload(low: float, high: float, count: int) -> Callable[..., None]:
    if not isinstance(low, nb.types.Integer):

        def _impl(low: float, high: float, count: int) -> None:
            pass

        return _impl

    if not low.signed:

        def _impl(low: float, high: float, count: int) -> None:
            # uint64 values are non-negative, so the largest magnitude is the
            # maximum itself; compare by division, count * high would wrap
            if count > 0 and high > EXACT_FLOAT64_INTEGER_LIMIT // count:
                raise OverflowError(_mean_overflow_message(count, high))

        return _impl

    def _impl(low: float, high: float, count: int) -> None:  # type: ignore[no-redef]  # the overload hands numba one of the two same-named alternatives
        # -low wraps at int64-min and count * bound wraps for large values -
        # treat the former as the out-of-range case it is and compare the
        # latter by division
        if count > 0:
            if low == INT64_MIN:
                # |int64-min| is not an int64 itself - the uint64 constant
                # keeps the reported magnitude from wrapping
                raise OverflowError(_mean_overflow_message(count, np.uint64(2**63)))
            if max(high, -low) > EXACT_FLOAT64_INTEGER_LIMIT // count:
                raise OverflowError(_mean_overflow_message(count, max(high, -low)))

    return _impl


def _span_limit_check(low: float, high: float, count: int) -> None:
    """Refuse integer variances whose exact centering would overflow int64 -
    replaced per dtype by the overload below, a no-op for inexact input."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_span_limit_check)
def _span_limit_check_overload(low: float, high: float, count: int) -> Callable[..., None]:
    if not isinstance(low, nb.types.Integer):

        def _impl(low: float, high: float, count: int) -> None:
            pass

        return _impl

    if not low.signed:

        def _impl(low: float, high: float, count: int) -> None:
            # uint64 values are non-negative, so high - low cannot wrap
            if count > 0 and high - low > INT64_DEVIATION_LIMIT:
                raise OverflowError(_span_overflow_message(low, high))

        return _impl

    def _impl(low: float, high: float, count: int) -> None:  # type: ignore[no-redef]  # the overload hands numba one of the two same-named alternatives
        # signed: only the mixed-sign case can exceed the deviation limit,
        # and high - low itself would wrap there - compare without it
        if count > 0 and low < 0 and high >= 0 and high > INT64_DEVIATION_LIMIT + low:
            raise OverflowError(_span_overflow_message(low, high))

    return _impl


class MeanState(NamedTuple):
    """``out`` doubles as the running sum until the finishing pass divides it
    by ``count``; ``low``/``high`` are the single-element global extrema
    scratch that feed the integer exactness gate."""

    out: npt.NDArray[np.inexact]
    count: npt.NDArray[np.int64]  # type: ignore[assignment]  # the field name shadows tuple.count, which numba does not see
    low: npt.NDArray[np.number]
    high: npt.NDArray[np.number]


def _mean_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> MeanState:
    return MeanState(
        np.zeros(size, out_dtype),
        np.zeros(size, np.int64),
        np.empty(1, in_dtype),
        np.empty(1, in_dtype),
    )


def _mean_seed(st: MeanState, a: npt.NDArray[np.number]) -> None:
    # seed the global extrema scratch with the first value
    if len(a) > 0:
        st.low[0] = a[0]
        st.high[0] = a[0]


def _mean_update(st: MeanState, ri: int, i: int, val: float, fill_value: float, ddof: int) -> None:
    st.count[ri] += 1
    st.out[ri] += val
    _track_extrema(val, st.low, st.high)  # type: ignore[arg-type]  # low/high are 1-element arrays; body uses low[0]


def _mean_gate(st: MeanState, n_values: int) -> None:
    _mean_limit_check(st.low[0], st.high[0], n_values)


def _mean_finish(st: MeanState, g: int, fill_value: float, ddof: int) -> Any:
    if st.count[g] > ddof:
        return st.out[g] / st.count[g]
    return fill_value


mean = build_kernel(
    "mean",
    init=_mean_init,
    update=_mean_update,
    seed=_mean_seed,
    gate=_mean_gate,
    finish=_mean_finish,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_float,
    doc="Arithmetic mean of the values of every group",
)


class MomentState(NamedTuple):
    """The running central moments of the skewness and the kurtosis kernel.

    ``out`` is the result the driver writes into, ``m1`` is the running mean, ``m2``/``m3``/``m4`` the sums of the squared,
    cubed and fourth powers of the deviations from it, ``count`` the number of
    values seen, and ``low``/``high`` the global extrema that feed the
    exactness gate.  Kept flat like the other states - numba cannot type
    inherited NamedTuples."""

    out: npt.NDArray[np.floating]
    m1: npt.NDArray[np.floating]
    m2: npt.NDArray[np.floating]
    m3: npt.NDArray[np.floating]
    m4: npt.NDArray[np.floating]
    count: npt.NDArray[np.int64]  # type: ignore[assignment]  # the field name shadows tuple.count, which numba does not see
    low: npt.NDArray[np.number]
    high: npt.NDArray[np.number]


def _moment_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> MomentState:
    return MomentState(
        np.zeros(size, out_dtype),
        np.zeros(size, out_dtype),
        np.zeros(size, out_dtype),
        np.zeros(size, out_dtype),
        np.zeros(size, out_dtype),
        np.zeros(size, np.int64),
        np.empty(1, in_dtype),
        np.empty(1, in_dtype),
    )


def _moment_seed(st: MomentState, a: npt.NDArray[np.number]) -> None:
    # seed the global extrema scratch with the first value
    if len(a) > 0:
        st.low[0] = a[0]
        st.high[0] = a[0]


def _moment_update(st: MomentState, ri: int, i: int, val: float, fill_value: float, ddof: int) -> None:
    """Accumulate one value into the running central moments.

    The update is the one of Welford (and the one pandas uses): the moments
    are kept around the running mean, so that a large offset cannot erase
    the small deviations the skewness and the kurtosis are made of.
    """
    if isinstance(val, complex):
        # the moments of the deviations of a complex value would be complex,
        # and their ratio meaningless
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    st.count[ri] += 1
    n = st.count[ri]
    delta = val - st.m1[ri]
    delta_n = delta / n
    term1 = delta * delta_n * (n - 1)
    st.m4[ri] += (
        term1 * delta_n * delta_n * (n * n - 3 * n + 3) + 6 * delta_n * delta_n * st.m2[ri] - 4 * delta_n * st.m3[ri]
    )
    st.m3[ri] += term1 * delta_n * (n - 2) - 3 * delta_n * st.m2[ri]
    st.m2[ri] += term1
    st.m1[ri] += delta_n
    _track_extrema(val, st.low, st.high)  # type: ignore[arg-type]  # low/high are 1-element arrays; body uses low[0]


def _moment_update_nan(st: MomentState, ri: int, i: int, val: float, fill_value: float, ddof: int) -> None:
    """The same accumulation, with the nans left out.

    The body is a copy of the one above: a call would make the loop of the
    nan variant ten times slower than the one of the plain variant, and the
    accumulation is a handful of lines.
    """
    if val != val:
        return
    if isinstance(val, complex):
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    st.count[ri] += 1
    n = st.count[ri]
    delta = val - st.m1[ri]
    delta_n = delta / n
    term1 = delta * delta_n * (n - 1)
    st.m4[ri] += (
        term1 * delta_n * delta_n * (n * n - 3 * n + 3) + 6 * delta_n * delta_n * st.m2[ri] - 4 * delta_n * st.m3[ri]
    )
    st.m3[ri] += term1 * delta_n * (n - 2) - 3 * delta_n * st.m2[ri]
    st.m2[ri] += term1
    st.m1[ri] += delta_n
    _track_extrema(val, st.low, st.high)  # type: ignore[arg-type]  # low/high are 1-element arrays; body uses low[0]


def _moment_gate(st: MomentState, n_values: int) -> None:
    _mean_limit_check(st.low[0], st.high[0], n_values)


def _skew_finish(st: MomentState, g: int, fill_value: float, ddof: int) -> Any:
    n = st.count[g]
    if n == 0:
        return fill_value
    if n < 3:
        # fewer than three values have no skewness
        return np.nan
    if st.m2[g] == 0:
        # a group whose values are all the same has no spread - and no
        # skewness either, which pandas answers with a zero
        return 0.0
    return n * np.sqrt(n - 1) / (n - 2) * st.m3[g] / st.m2[g] ** 1.5


def _kurt_finish(st: MomentState, g: int, fill_value: float, ddof: int) -> Any:
    n = st.count[g]
    if n == 0:
        return fill_value
    if n < 4:
        # fewer than four values have no kurtosis
        return np.nan
    if st.m2[g] == 0:
        return 0.0
    return n * (n + 1) * (n - 1) * st.m4[g] / ((n - 2) * (n - 3) * st.m2[g] ** 2) - 3 * (n - 1) ** 2 / (
        (n - 2) * (n - 3)
    )


skew = build_kernel(
    "skew",
    init=_moment_init,
    update=_moment_update,
    update_nan=_moment_update_nan,
    seed=_moment_seed,
    gate=_moment_gate,
    finish=_skew_finish,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_float,
    doc="Skewness of the values of every group",
)


kurt = build_kernel(
    "kurt",
    init=_moment_init,
    update=_moment_update,
    update_nan=_moment_update_nan,
    seed=_moment_seed,
    gate=_moment_gate,
    finish=_kurt_finish,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_float,
    doc="Excess kurtosis of the values of every group",
)


class StdState(NamedTuple):
    """The running sums of the dispersion kernels: ``out`` accumulates the
    squared deviations in the real output dtype, ``dev`` the deviations
    themselves in the widened auxiliary dtype (complex stays complex),
    ``first`` anchors the integer centering and ``low``/``high`` feed the
    exactness gate.  Kept flat instead of extending ``MeanState`` - numba
    cannot type inherited NamedTuples."""

    out: npt.NDArray[np.floating]
    count: npt.NDArray[np.int64]  # type: ignore[assignment]  # the field name shadows tuple.count, which numba does not see
    dev: npt.NDArray[np.inexact]
    first: npt.NDArray[np.number]
    low: npt.NDArray[np.number]
    high: npt.NDArray[np.number]


def _accumulate_std(val: float, st: StdState, ri: int) -> None:
    """Accumulate one value into the running var/std sums.

    Replaced per dtype by the overload below: integer input is centered on
    the first value of its group so that a large common offset cannot erase
    small deviations, inexact input keeps the plain sums."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_accumulate_std)
def _accumulate_std_overload(val: float, st: StdState, ri: int) -> Callable[..., None]:
    if isinstance(val, nb.types.Integer):

        def _impl(val: float, st: StdState, ri: int) -> None:
            if st.count[ri] == 1:
                st.first[ri] = val
            # two's-complement deviation, exact in int64 because the gate
            # check between the passes verified that the values span less
            # than +/-2**63; uint64 input wraps into the correct signed bit
            # pattern via the np.int64 casts
            dev = np.int64(val) - np.int64(st.first[ri])
            st.dev[ri] += dev
            dev_sq = np.float64(dev)
            st.out[ri] += dev_sq * dev_sq
            _track_extrema(val, st.low, st.high)

        return _impl

    def _impl(val: float, st: StdState, ri: int) -> None:  # type: ignore[no-redef]  # the overload hands numba one of the two same-named alternatives
        st.dev[ri] += val
        # the squared magnitude of the complex deviation is real, so out stays
        # real for complex input - matching np.var
        st.out[ri] += val.real * val.real + val.imag * val.imag

    return _impl


def _std_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> StdState:
    return StdState(
        np.zeros(size, out_dtype),
        np.zeros(size, np.int64),
        np.zeros(size, aux_dtype),
        np.empty(size, in_dtype),
        np.empty(1, in_dtype),
        np.empty(1, in_dtype),
    )


def _std_seed(st: StdState, a: npt.NDArray[np.number]) -> None:
    # seed the global extrema scratch with the first value
    if len(a) > 0:
        st.low[0] = a[0]
        st.high[0] = a[0]


def _std_update(st: StdState, ri: int, i: int, val: float, fill_value: float, ddof: int) -> None:
    st.count[ri] += 1
    if isinstance(val, bool):
        st.dev[ri] += val
        # bool has no .real in numba and keeps the plain val * val product
        st.out[ri] += val * val
    else:
        _accumulate_std(val, st, ri)


def _std_gate(st: StdState, n_values: int) -> None:
    _span_limit_check(st.low[0], st.high[0], n_values)


def _std_finish(st: StdState, g: int, fill_value: float, ddof: int) -> Any:
    if st.count[g] > ddof:
        m = st.dev[g]
        mean2 = m.real * m.real + m.imag * m.imag
        return np.sqrt((st.out[g] - mean2 / st.count[g]) / (st.count[g] - ddof))
    return fill_value


std = build_kernel(
    "std",
    init=_std_init,
    update=_std_update,
    seed=_std_seed,
    gate=_std_gate,
    finish=_std_finish,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_real_float,
    aux_widen=True,
    extras=(("ddof", "int", 0, "ddof"),),
    doc="Standard deviation of the values of every group",
)


def _var_finish(st: StdState, g: int, fill_value: float, ddof: int) -> Any:
    if st.count[g] > ddof:
        m = st.dev[g]
        mean2 = m.real * m.real + m.imag * m.imag
        return (st.out[g] - mean2 / st.count[g]) / (st.count[g] - ddof)
    return fill_value


var = build_kernel(
    "var",
    init=_std_init,
    update=_std_update,
    seed=_std_seed,
    gate=_std_gate,
    finish=_var_finish,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_real_float,
    aux_widen=True,
    extras=(("ddof", "int", 0, "ddof"),),
    doc="Variance of the values of every group",
)
