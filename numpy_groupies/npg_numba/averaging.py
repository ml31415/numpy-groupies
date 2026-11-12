"""Two-pass averaging: mean, std and var, plus the Welford moments of the
skewness and the kurtosis.

Integer input is centered on its group's first value, the global min/max is
tracked in the same accumulation pass, and the exactness gates run between
the passes and raise ``OverflowError`` for the same inputs in every backend.
The answers are floats; where numpy's stubs know no element type -
``typing.cast`` is not compiled in nopython mode - a local states the type.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.npg_numba.compose import (
    TickKernel,
    build_kernel,
    complex_less,
    make_tick_call,
    rule_float,
    rule_int64,
    rule_real_float,
)
from numpy_groupies.npg_numba.jit import njit_stable, register_overload
from numpy_groupies.temporal import NAT_TICK
from numpy_groupies.validation import (
    EXACT_FLOAT64_INTEGER_LIMIT,
    INT64_DEVIATION_LIMIT,
    INT64_MIN,
    MOMENTS_NOT_FOR_COMPLEX,
    mean_overflow_message,
    variance_span_overflow_message,
)


# the building blocks and overload shims of this file are jitted: numba
# compiles them from the argument types and ignores these annotations, the
# type checker needs them.  One exception it does not see: numba compares the
# *signature* of an ``@overload`` function with that of the implementation it
# returns, annotations included, so the shim, its overload and every
# alternative implementation must be spelled alike - a lone ``int`` where the
# overload says ``float`` refuses the compile ("Typing and implementation
# arguments differ in argument names")
def _track_extrema(val: float, low: npt.NDArray[np.number], high: npt.NDArray[np.number]) -> None:
    """The global min/max tracking for the integer limit checks, folded into
    the accumulation pass - replaced per dtype by the overload below, inexact
    input never tracks anything."""
    raise NotImplementedError  # replaced by the overload below


@register_overload(_track_extrema)
def _track_extrema_overload(
    val: float, low: npt.NDArray[np.number], high: npt.NDArray[np.number]
) -> Callable[..., None]:
    if isinstance(val, nb.types.Integer):

        def _impl_integer(val: float, low: npt.NDArray[np.number], high: npt.NDArray[np.number]) -> None:
            low[0] = min(low[0], val)
            high[0] = max(high[0], val)

        return _impl_integer

    def _impl_inexact(val: float, low: npt.NDArray[np.number], high: npt.NDArray[np.number]) -> None:
        pass

    return _impl_inexact


# compiled twins of ``validation``'s message builders - numba cannot call a
# python function, but it can compile one
_mean_overflow_message: Callable[[int, int], str] = njit_stable(mean_overflow_message, name="mean_overflow_message")
_span_message: Callable[[str, int], str] = njit_stable(
    variance_span_overflow_message, name="variance_span_overflow_message"
)


@njit_stable
def _tick_mean_overflow_message(count: int, span: int) -> str:
    """The complaint about a mean of ticks whose sum would overflow int64."""
    return (
        "cannot compute an exact mean of "
        + str(count)
        + " ticks with a span of "
        + str(span)
        + ": their sum would leave the int64 range - the ticks of the values "
        + "are too far apart"
    )


@njit_stable
def _span_overflow_message(low: int, high: int) -> str:
    """The numba twin of ``check_integer_variance_span_overflow``."""
    # numba's str() negates int64-min internally, wraps, and then fails to
    # size the string buffer - spell that one value out
    low_text = "-9223372036854775808" if low == INT64_MIN else str(low)
    return _span_message(low_text, high)


def _mean_limit_check(low: float, high: float, count: int) -> None:
    """Refuse integer means float64 cannot compute exactly - replaced per
    dtype by the overload below, a no-op for inexact input."""
    raise NotImplementedError  # replaced by the overload below


@register_overload(_mean_limit_check)
def _mean_limit_check_overload(low: float, high: float, count: int) -> Callable[..., None]:
    if not isinstance(low, nb.types.Integer):

        def _impl_inexact(low: float, high: float, count: int) -> None:
            pass

        return _impl_inexact

    if not low.signed:

        def _impl_unsigned(low: float, high: float, count: int) -> None:
            # uint64 is non-negative: the largest magnitude is the maximum;
            # compare by division, count * high would wrap
            if count > 0 and high > EXACT_FLOAT64_INTEGER_LIMIT // count:
                raise OverflowError(_mean_overflow_message(count, high))

        return _impl_unsigned

    def _impl_signed(low: float, high: float, count: int) -> None:
        # -low wraps at int64-min and count * bound wraps - the former is the
        # out-of-range case, the latter compared by division
        if count > 0:
            if low == INT64_MIN:
                # |int64-min| is no int64 - the uint64 constant keeps the
                # reported magnitude from wrapping
                raise OverflowError(_mean_overflow_message(count, np.uint64(2**63)))
            if max(high, -low) > EXACT_FLOAT64_INTEGER_LIMIT // count:
                raise OverflowError(_mean_overflow_message(count, max(high, -low)))

    return _impl_signed


def _span_limit_check(low: float, high: float, count: int) -> None:
    """Refuse integer variances whose exact centering would overflow int64 -
    replaced per dtype by the overload below, a no-op for inexact input."""
    raise NotImplementedError  # replaced by the overload below


@register_overload(_span_limit_check)
def _span_limit_check_overload(low: float, high: float, count: int) -> Callable[..., None]:
    if not isinstance(low, nb.types.Integer):

        def _impl_inexact(low: float, high: float, count: int) -> None:
            pass

        return _impl_inexact

    if not low.signed:

        def _impl_unsigned(low: float, high: float, count: int) -> None:
            # uint64 values are non-negative, so high - low cannot wrap
            if count > 0 and high - low > INT64_DEVIATION_LIMIT:
                raise OverflowError(_span_overflow_message(low, high))

        return _impl_unsigned

    def _impl_signed(low: float, high: float, count: int) -> None:
        # signed: only the mixed-sign case can exceed the deviation limit,
        # and high - low itself would wrap there - compare without it
        if count > 0 and low < 0 and high >= 0 and high > INT64_DEVIATION_LIMIT + low:
            raise OverflowError(_span_overflow_message(low, high))

    return _impl_signed


class MeanState(NamedTuple):
    """``out`` doubles as the running sum until the finishing pass divides it
    by ``counts``; ``low``/``high`` are the single-element global extrema
    scratch that feed the integer exactness gate."""

    out: npt.NDArray[np.inexact]
    counts: npt.NDArray[np.int64]
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


def _mean_update(st: MeanState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    st.counts[ri] += 1
    st.out[ri] += val
    _track_extrema(val, st.low, st.high)


def _mean_gate(st: MeanState, n_values: int) -> None:
    _mean_limit_check(st.low[0], st.high[0], n_values)


def _mean_finish(st: MeanState, g: int, fill_value: float, extra: Any) -> float:
    # the mean takes nothing of its own, so the driver reaches this with the
    # zero of an operation without an argument and the test is the plain one of
    # a group that received a value at all
    if st.counts[g] > extra:
        # the mean of the group
        mean: float = st.out[g] / st.counts[g]
        return mean
    return fill_value


def _mean_combine(acc: MeanState, other: MeanState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one: the sums and
    the counts add up, the extrema are the ones of both chunks."""
    out: np.ndarray = acc.out
    out += other.out
    counts: np.ndarray = acc.counts
    counts += other.counts
    # the extrema compare the way the kernel compares - min()/max() have no
    # meaning for complex values
    low: np.ndarray = acc.low
    if complex_less(other.low[0], low[0]):
        low[0] = other.low[0]
    high: np.ndarray = acc.high
    if complex_less(high[0], other.high[0]):
        high[0] = other.high[0]


class TickMeanState(NamedTuple):
    """The state of the mean of the ticks of a temporal input: the sum of the
    deviations from the group's origin, the counts, the origin, the group's
    extrema and its poison flag.

    The deviation sum is exact in int64 (a float64 window would run out long
    before the group is done); the origin is the group's first value in the
    sequential path and the chunk's in the parallel one - the deviation sum
    does not depend on it.
    """

    out: npt.NDArray[np.int64]
    counts: npt.NDArray[np.int64]
    origin: npt.NDArray[np.int64]
    low: npt.NDArray[np.int64]
    high: npt.NDArray[np.int64]
    poisoned: npt.NDArray[np.bool_]


def _tick_mean_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: int,
    forced_fill: int,
) -> TickMeanState:
    return TickMeanState(
        np.zeros(size, np.int64),
        np.zeros(size, np.int64),
        np.zeros(size, np.int64),
        np.full(size, np.iinfo(np.int64).max, np.int64),
        np.full(size, INT64_MIN, np.int64),
        np.zeros(size, np.bool_),
    )


def _make_tick_mean_update(nan: bool) -> Callable[..., None]:
    """The update of the tick mean, in its plain and its nan form.

    The NaT is the tick -2**63, and the kernel sees ticks alone: the nan
    variant leaves it out, the plain one takes its group over (a value of
    exactly that tick counts as one).  The skip is a branch, measured: on
    NaT-free data it never runs, while a branchless skip of four selects
    costs a fifth of the kernel even on clean data.
    """

    def _tick_mean_update(st: TickMeanState, ri: int, i: int, val: int, fill_value: int, extra: Any) -> None:
        if nan and val == INT64_MIN:
            return
        if val == INT64_MIN:
            st.poisoned[ri] = True
        if st.counts[ri] == 0:
            # the first value is the origin its deviations are measured from
            # - the values of a group are close, their sum is what overflows
            st.origin[ri] = val
        st.counts[ri] += 1
        st.out[ri] += val - st.origin[ri]
        # the extrema of the group, for the check of the sum of the deviations
        st.low[ri] = min(st.low[ri], val)
        st.high[ri] = max(st.high[ri], val)

    return _tick_mean_update


_tick_mean_update = _make_tick_mean_update(False)
_tick_mean_update_nan = _make_tick_mean_update(True)


def _tick_mean_finish(st: TickMeanState, g: int, fill_value: int, extra: Any) -> int:
    """The mean of the ticks of a group, rounded half to even - the exact
    deviation sum makes quotient and remainder exact."""
    if st.poisoned[g]:
        # a single NaT poisons the mean of its whole group
        return NAT_TICK
    if st.counts[g] > extra:
        count = st.counts[g]
        # the deviation sum stays inside int64 while the span times the
        # count does - beyond it the sum would wrap around silently
        span = st.high[g] - st.low[g]
        if span > INT64_DEVIATION_LIMIT // count:
            raise OverflowError(_tick_mean_overflow_message(count, span))
        # total = origin * count + the sum of the deviations - the quotient
        # is the origin plus the quotient of the deviations
        quotient = st.out[g] // count
        remainder = st.out[g] - quotient * count
        # half to even looks at the parity of the total - the origin and the
        # quotient added up (the total itself does not fit into an int64)
        odd = (st.origin[g] & 1) + (quotient & 1)
        if 2 * remainder > count or (2 * remainder == count and odd % 2 == 1):
            quotient += 1
        # the mean of the ticks, rounded half to even
        mean: int = st.origin[g] + quotient
        return mean
    return fill_value


def _tick_mean_combine(acc: TickMeanState, other: TickMeanState, fill_value: int, extra: Any) -> None:
    """The states of two chunks of the data, folded into one.

    The deviations of the other chunk are moved to this one's origin by the
    difference of the origins; the origin of the chunk that saw the first
    value does not win - the deviation sum does not depend on it.
    """
    out: np.ndarray = acc.out
    counts: np.ndarray = acc.counts
    origin: np.ndarray = acc.origin
    low: np.ndarray = acc.low
    high: np.ndarray = acc.high
    other_count: np.ndarray = other.counts
    other_out: np.ndarray = other.out
    other_origin: np.ndarray = other.origin
    for g in range(out.size):
        if counts[g] == 0:
            # no value in this chunk - the other chunk's state is the group's
            out[g] = other_out[g]
            counts[g] = other_count[g]
            origin[g] = other_origin[g]
        elif other_count[g] > 0:
            # the other chunk's deviations move to this origin - the sum
            # stays inside the range the check of the finish allows
            out[g] += other_out[g] + other_count[g] * (other_origin[g] - origin[g])
            counts[g] += other_count[g]
        low[g] = min(low[g], other.low[g])
        high[g] = max(high[g], other.high[g])
        acc.poisoned[g] = acc.poisoned[g] or other.poisoned[g]


tick_mean = build_kernel(
    "tick_mean",
    init=_tick_mean_init,
    update=_tick_mean_update,
    update_nan=_tick_mean_update_nan,
    finish=_tick_mean_finish,
    combine=_tick_mean_combine,
    rule=rule_int64,
    fill_default=0,
    forced_fill=0,
    doc="Mean of the ticks of every group, rounded half to even - the tick of a NaT poisons its group, and the nan form leaves it out",
)


mean = build_kernel(
    "mean",
    init=_mean_init,
    update=_mean_update,
    combine=_mean_combine,
    # the state of the mean holds four arrays (the sum, the count and the two
    # extrema of the exactness gate) against the two of the sum
    state_size=2,
    seed=_mean_seed,
    gate=_mean_gate,
    finish=_mean_finish,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_float,
    doc="Arithmetic mean of the values of every group",
    # the mean of the temporal input is the mean of the ticks: the stub hands
    # the ticks to the tick kernel and views the answer back
    tick=TickKernel(
        plain=make_tick_call("tick_mean_call", tick_mean.plain),
        nan=make_tick_call("nantick_mean_call", tick_mean.nan),
    ),
)


class MomentState(NamedTuple):
    """The running central moments of the skew/kurt kernel: ``out``, the
    running mean ``m1``, the deviation powers ``m2``/``m3``/``m4``, the counts
    and the gate extrema.  Kept flat - numba cannot type inherited
    NamedTuples."""

    out: npt.NDArray[np.floating]
    m1: npt.NDArray[np.floating]
    m2: npt.NDArray[np.floating]
    m3: npt.NDArray[np.floating]
    m4: npt.NDArray[np.floating]
    counts: npt.NDArray[np.int64]
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


def _moment_update(st: MomentState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    """Accumulate one value into the running central moments - Welford, so a
    large offset cannot erase the small deviations the higher moments are
    made of."""
    if isinstance(val, complex):
        # the moments of complex deviations would be complex, their ratio
        # meaningless
        raise TypeError(MOMENTS_NOT_FOR_COMPLEX)
    st.counts[ri] += 1
    n = st.counts[ri]
    delta = val - st.m1[ri]
    delta_n = delta / n
    term1 = delta * delta_n * (n - 1)
    st.m4[ri] += (
        term1 * delta_n * delta_n * (n * n - 3 * n + 3) + 6 * delta_n * delta_n * st.m2[ri] - 4 * delta_n * st.m3[ri]
    )
    st.m3[ri] += term1 * delta_n * (n - 2) - 3 * delta_n * st.m2[ri]
    st.m2[ri] += term1
    st.m1[ri] += delta_n
    _track_extrema(val, st.low, st.high)


def _moment_update_nan(st: MomentState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    """The same accumulation, with the nans left out - a copy on purpose: a
    call would make the nan loop ten times slower than the plain one."""
    if val != val:
        return
    if isinstance(val, complex):
        raise TypeError(MOMENTS_NOT_FOR_COMPLEX)
    st.counts[ri] += 1
    n = st.counts[ri]
    delta = val - st.m1[ri]
    delta_n = delta / n
    term1 = delta * delta_n * (n - 1)
    st.m4[ri] += (
        term1 * delta_n * delta_n * (n * n - 3 * n + 3) + 6 * delta_n * delta_n * st.m2[ri] - 4 * delta_n * st.m3[ri]
    )
    st.m3[ri] += term1 * delta_n * (n - 2) - 3 * delta_n * st.m2[ri]
    st.m2[ri] += term1
    st.m1[ri] += delta_n
    _track_extrema(val, st.low, st.high)


def _moment_gate(st: MomentState, n_values: int) -> None:
    _mean_limit_check(st.low[0], st.high[0], n_values)


def _moment_combine(acc: MomentState, other: MomentState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one.

    The Welford moments are about their own chunk's mean, so the merge of
    Chan et al. adds the correction of the distance between the means; groups
    that one chunk saw alone carry a zero count in every correction term.
    The merge is not the accumulation's arithmetic - the parallel result
    differs from the sequential one in the last bits, like float sums do.
    """
    n_a = acc.counts
    n_b = other.counts
    total = n_a + n_b
    # whether the accumulator has seen a value at all - an empty chunk's
    # extrema scratch is never seeded; asked before the counts merge
    acc_empty = acc.counts.sum() == 0
    # the counts as floats - the arithmetic below divides by them
    fa = n_a.astype(np.float64)
    fb = n_b.astype(np.float64)
    # the divisor of the groups that no chunk received a value for - their
    # computed values are thrown away by the counts
    safe = np.where(total > 0, total.astype(np.float64), 1.0)

    delta = other.m1 - acc.m1
    d2 = delta * delta
    m1 = acc.m1 + delta * fb / safe
    m2 = acc.m2 + other.m2 + d2 * fa * fb / safe
    m3 = (
        acc.m3
        + other.m3
        + delta * d2 * fa * fb * (fa - fb) / (safe * safe)
        + 3.0 * delta * (fa * other.m2 - fb * acc.m2) / safe
    )
    m4 = (
        acc.m4
        + other.m4
        + d2 * d2 * fa * fb * (fa * fa - fa * fb + fb * fb) / (safe * safe * safe)
        + 6.0 * d2 * (fa * fa * other.m2 + fb * fb * acc.m2) / (safe * safe)
        + 4.0 * delta * (fa * other.m3 - fb * acc.m3) / safe
    )
    acc.m1[:] = m1
    acc.m2[:] = m2
    acc.m3[:] = m3
    acc.m4[:] = m4
    acc.counts[:] = total

    # the gate extrema of both chunks; an empty chunk's scratch is never
    # seeded, so its extrema are left out - a zero count means the chunk was
    # empty (the extrema are tracked for the nan-free integer input only)
    if other.counts.sum() > 0:
        if acc_empty:
            # the accumulator has no extrema of its own: take the other
            # chunk's as they are
            acc.low[0] = other.low[0]
            acc.high[0] = other.high[0]
        else:
            # min()/max() have no meaning for complex values - compared the
            # way the kernel compares
            if complex_less(other.low[0], acc.low[0]):
                acc.low[0] = other.low[0]
            if complex_less(acc.high[0], other.high[0]):
                acc.high[0] = other.high[0]


def _skew_finish(st: MomentState, g: int, fill_value: float, extra: Any) -> float:
    n = st.counts[g]
    if n == 0:
        return fill_value
    if n < 3:
        # fewer than three values have no skewness
        return np.nan
    if st.m2[g] == 0:
        # no spread, no skewness - pandas answers a zero
        return 0.0
    skew: float = n * np.sqrt(n - 1) / (n - 2) * st.m3[g] / st.m2[g] ** 1.5
    return skew


def _kurt_finish(st: MomentState, g: int, fill_value: float, extra: Any) -> float:
    n = st.counts[g]
    if n == 0:
        return fill_value
    if n < 4:
        # fewer than four values have no kurtosis
        return np.nan
    if st.m2[g] == 0:
        return 0.0
    # the kurtosis of the group
    kurt: float = n * (n + 1) * (n - 1) * st.m4[g] / ((n - 2) * (n - 3) * st.m2[g] ** 2) - 3 * (n - 1) ** 2 / (
        (n - 2) * (n - 3)
    )
    return kurt


skew = build_kernel(
    "skew",
    init=_moment_init,
    update=_moment_update,
    update_nan=_moment_update_nan,
    seed=_moment_seed,
    gate=_moment_gate,
    finish=_skew_finish,
    combine=_moment_combine,
    # eight arrays of state against the sum's two, but the fill and the fold
    # of the partial states cost more than that says - measured, the parallel
    # path is 2.35x faster at 3000 groups/a million values and slower at 8000
    # (six threads), which puts the number at five
    state_size=5,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_float,
    # the refusal stands ahead of the kernel: the compiled one would raise
    # from inside its accumulation loop, in the parallel region where an
    # exception cannot leave
    refusal=MOMENTS_NOT_FOR_COMPLEX,
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
    combine=_moment_combine,
    # eight arrays of state against the sum's two, but the fill and the fold
    # of the partial states cost more than that says - measured, the parallel
    # path is 2.35x faster at 3000 groups/a million values and slower at 8000
    # (six threads), which puts the number at five
    state_size=5,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_float,
    # the refusal stands ahead of the kernel, like the skewness's
    refusal=MOMENTS_NOT_FOR_COMPLEX,
    doc="Excess kurtosis of the values of every group",
)


class StdState(NamedTuple):
    """The running sums of the dispersion kernels: ``out`` the squared
    deviations in the real output dtype, ``dev`` the deviations in the
    widened aux dtype (complex stays complex), ``first`` the centering anchor,
    ``low``/``high`` the gate extrema.  Kept flat - numba cannot type
    inherited NamedTuples."""

    out: npt.NDArray[np.floating]
    counts: npt.NDArray[np.int64]
    dev: npt.NDArray[np.inexact]
    first: npt.NDArray[np.number]
    low: npt.NDArray[np.number]
    high: npt.NDArray[np.number]


def _accumulate_std(val: float, st: StdState, ri: int) -> None:
    """Accumulate one value into the running var/std sums - replaced per dtype
    by the overload below: integer input is centered on its group's first
    value so a large common offset cannot erase small deviations."""
    raise NotImplementedError  # replaced by the overload below


@register_overload(_accumulate_std)
def _accumulate_std_overload(val: float, st: StdState, ri: int) -> Callable[..., None]:
    if isinstance(val, nb.types.Integer):

        def _impl_integer(val: float, st: StdState, ri: int) -> None:
            if st.counts[ri] == 1:
                st.first[ri] = val
            # two's-complement deviation, exact in int64 (the gate verified
            # the span below 2**63); uint64 wraps into the correct signed bit
            # pattern via the np.int64 casts
            dev = np.int64(val) - np.int64(st.first[ri])
            st.dev[ri] += dev
            dev_sq = np.float64(dev)
            st.out[ri] += dev_sq * dev_sq
            _track_extrema(val, st.low, st.high)

        return _impl_integer

    def _impl_inexact(val: float, st: StdState, ri: int) -> None:
        # centered on the first value as well: the plain sums of raw values
        # cancel - the spread of a constant group is lost in the rounding of
        # mean^2 - sum^2/n
        if st.counts[ri] == 1:
            st.first[ri] = val
        dev = val - st.first[ri]
        st.dev[ri] += dev
        # the squared magnitude is real, so out stays real for complex input
        # - matching np.var
        st.out[ri] += dev.real * dev.real + dev.imag * dev.imag

    return _impl_inexact


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


def _std_update(st: StdState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    st.counts[ri] += 1
    if isinstance(val, bool):
        st.dev[ri] += val
        # bool has no .real in numba and keeps the plain val * val product
        st.out[ri] += val * val
    else:
        _accumulate_std(val, st, ri)


def _std_gate(st: StdState, n_values: int) -> None:
    _span_limit_check(st.low[0], st.high[0], n_values)


def _std_combine(acc: StdState, other: StdState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one, with the anchor
    correction of the centered input.  The overload has to replace a function
    that is *called* - an overload of this function itself would never run.
    """
    _fold_std(acc, other, fill_value, extra)


def _fold_std(acc: StdState, other: StdState, fill_value: float, extra: Any) -> None:
    """The fold of the two states - the integer and the inexact input differ."""
    raise NotImplementedError  # replaced by the overload below


@register_overload(_fold_std)
def _fold_std_overload(acc: StdState, other: StdState, fill_value: float, extra: Any) -> Callable[..., None]:
    # the dtype of the anchor field (the fourth) tells the cases apart - the
    # fields are reached through their types in the typing of the overload,
    # and the argument annotations must match the implementation's (numba
    # checks the two signatures against each other, strict=True)
    # the state's type in the typing of the overload - the checker knows no
    # type of a numba type
    acc_type: Any = acc
    if isinstance(acc_type.types[3].dtype, nb.types.Integer):

        def _impl_integer(acc: StdState, other: StdState, fill_value: float, extra: Any) -> None:
            # every chunk is centered on its own first value - the deviations
            # and squares are corrected for the anchor difference:
            #   sum(v - a)     = sum(v - b) + (b - a) * n
            #   sum((v - a)^2) = sum((v - b)^2) + 2 (b - a) sum(v - b) + (b - a)^2 n
            counts: np.ndarray = acc.counts
            out: np.ndarray = acc.out
            dev: np.ndarray = acc.dev
            for g in range(counts.size):
                n_other = other.counts[g]
                if n_other == 0:
                    continue
                if counts[g] == 0:
                    # the first chunk has no value of this group: take the
                    # second chunk's anchor
                    acc.first[g] = other.first[g]
                else:
                    delta = np.int64(other.first[g]) - np.int64(acc.first[g])
                    out[g] += 2 * delta * other.dev[g] + delta * delta * n_other
                    dev[g] += delta * n_other
                out[g] += other.out[g]
                dev[g] += other.dev[g]
                counts[g] += n_other
            _std_combine_extrema(acc, other)

        return _impl_integer

    if isinstance(acc_type.types[3].dtype, nb.types.Complex):

        def _impl_complex(acc: StdState, other: StdState, fill_value: float, extra: Any) -> None:
            # the squared magnitudes are corrected with the conjugate of the
            # anchor difference:
            #   sum(v - a)    = sum(v - b) + (b - a) * n
            #   sum|v - a|^2 = sum|v - b|^2 + 2 Re(sum(v - b) conj(b - a)) + |b - a|^2 n
            counts: np.ndarray = acc.counts
            out: np.ndarray = acc.out
            dev: np.ndarray = acc.dev
            for g in range(counts.size):
                n_other = other.counts[g]
                if n_other == 0:
                    continue
                if counts[g] == 0:
                    acc.first[g] = other.first[g]
                else:
                    delta = other.first[g] - acc.first[g]
                    out[g] += 2 * (other.dev[g] * np.conj(delta)).real + (delta.real**2 + delta.imag**2) * n_other
                    dev[g] += delta * n_other
                out[g] += other.out[g]
                dev[g] += other.dev[g]
                counts[g] += n_other
            _std_combine_extrema(acc, other)

        return _impl_complex

    def _impl_inexact(acc: StdState, other: StdState, fill_value: float, extra: Any) -> None:
        # every chunk is centered on its own first value - the deviations
        # and squares are corrected for the anchor difference:
        #   sum(v - a)     = sum(v - b) + (b - a) * n
        #   sum((v - a)^2) = sum((v - b)^2) + 2 (b - a) sum(v - b) + (b - a)^2 n
        counts: np.ndarray = acc.counts
        out: np.ndarray = acc.out
        dev: np.ndarray = acc.dev
        for g in range(counts.size):
            n_other = other.counts[g]
            if n_other == 0:
                continue
            if counts[g] == 0:
                acc.first[g] = other.first[g]
            else:
                delta = other.first[g] - acc.first[g]
                out[g] += 2 * delta * other.dev[g] + delta * delta * n_other
                dev[g] += delta * n_other
            out[g] += other.out[g]
            dev[g] += other.dev[g]
            counts[g] += n_other
        _std_combine_extrema(acc, other)

    return _impl_inexact


@njit_stable
def _std_combine_extrema(acc: StdState, other: StdState) -> None:
    """The extrema of two chunks folded into one - they feed the exactness
    gate."""
    low: np.ndarray = acc.low
    if complex_less(other.low[0], low[0]):
        low[0] = other.low[0]
    high: np.ndarray = acc.high
    if complex_less(high[0], other.high[0]):
        high[0] = other.high[0]


def _std_finish(st: StdState, g: int, fill_value: float, ddof: int) -> float:
    if st.counts[g] > ddof:
        m = st.dev[g]
        mean2 = m.real * m.real + m.imag * m.imag
        # the standard deviation of the group
        deviation: float = np.sqrt(_variance_of(st, g, mean2, ddof))
        return deviation
    return fill_value


std = build_kernel(
    "std",
    init=_std_init,
    update=_std_update,
    seed=_std_seed,
    gate=_std_gate,
    finish=_std_finish,
    combine=_std_combine,
    # the state of the dispersion holds six arrays against the two of the sum
    state_size=3,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_real_float,
    aux_widen=True,
    variant="ddof",
    doc="Standard deviation of the values of every group",
)


@njit_stable
def _variance_of(st: StdState, g: int, mean2: float, ddof: int) -> float:
    """the variance of a group from the sums the kernel accumulated.

    The one-pass formula cancels: a constant group leaves a tiny number that
    is negative half the time, so it is clamped at zero; a nan-poisoned group
    keeps its nan (``nan < 0`` is false).
    """
    variance = float((st.out[g] - mean2 / st.counts[g]) / (st.counts[g] - ddof))
    if variance < 0:
        return 0.0
    return variance


def _var_finish(st: StdState, g: int, fill_value: float, ddof: int) -> float:
    if st.counts[g] > ddof:
        m = st.dev[g]
        mean2 = m.real * m.real + m.imag * m.imag
        # the variance of the group
        variance: float = _variance_of(st, g, mean2, ddof)
        return variance
    return fill_value


var = build_kernel(
    "var",
    init=_std_init,
    update=_std_update,
    seed=_std_seed,
    gate=_std_gate,
    finish=_var_finish,
    combine=_std_combine,
    # the state of the dispersion holds six arrays against the two of the sum
    state_size=3,
    forced_fill=0,
    fill_default=np.nan,
    rule=rule_real_float,
    aux_widen=True,
    variant="ddof",
    doc="Variance of the values of every group",
)
