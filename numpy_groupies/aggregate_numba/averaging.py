"""Two-pass averaging: mean, std and var, including the integer exactness policy.

Integer input is centered on the first value of each group and the global
min/max is tracked in the same accumulation pass; the exactness gates run
between the passes and raise ``OverflowError`` for the same inputs in every
backend (see ``aggregate_common`` for the policy constants).
"""

import typing

import numba as nb
import numpy as np
import numpy.typing as npt

from ..aggregate_common import EXACT_FLOAT64_INTEGER_LIMIT, INT64_DEVIATION_LIMIT, INT64_MIN
from .compose import build_kernel, rule_float, rule_real_float
from .jit import njit_stable


def _track_extrema(val, low, high):
    """Fold the global min/max tracking for the integer limit checks into the
    accumulation pass - replaced per dtype by the overload below, inexact
    input never tracks anything."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_track_extrema)
def _track_extrema_overload(val, low, high):
    if isinstance(val, nb.types.Integer):

        def _impl(val, low, high):
            low[0] = min(low[0], val)
            high[0] = max(high[0], val)

        return _impl

    def _impl(val, low, high):
        pass

    return _impl


@njit_stable
def _mean_overflow_message(count, bound):
    """The numba twin of ``raise_integer_mean_overflow`` - numba cannot call
    the python helper (aggregate_common must stay numpy-free), so the text is
    mirrored here and must be kept in sync with it."""
    return (
        "cannot compute an exact mean of "
        + str(count)
        + " integer values with magnitude up to "
        + str(bound)
        + ": their sums can exceed +/-"
        + str(EXACT_FLOAT64_INTEGER_LIMIT)
        + ", the range float64 represents exactly - cast the input to a float dtype to accept "
        + "rounding, or use sum and len"
    )


@njit_stable
def _span_overflow_message(low, high):
    """The numba twin of ``raise_integer_variance_span_overflow``."""
    # numba's str() negates int64-min internally, wraps, and then fails to
    # size the string buffer - spell that one value out
    low_text = "-9223372036854775808" if low == INT64_MIN else str(low)
    return (
        "cannot compute an exact variance: the integer values span ["
        + low_text
        + ", "
        + str(high)
        + "], wider than +/-"
        + str(INT64_DEVIATION_LIMIT)
        + ", so centering each group on its first value would overflow the signed 64-bit "
        + "deviation - cast the input to a float dtype first"
    )


def _mean_limit_check(low, high, count):
    """Refuse integer means float64 cannot compute exactly - replaced per
    dtype by the overload below, a no-op for inexact input."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_mean_limit_check)
def _mean_limit_check_overload(low, high, count):
    if not isinstance(low, nb.types.Integer):

        def _impl(low, high, count):
            pass

        return _impl

    if not low.signed:

        def _impl(low, high, count):
            # uint64 values are non-negative, so the largest magnitude is the
            # maximum itself; compare by division, count * high would wrap
            if count > 0 and high > EXACT_FLOAT64_INTEGER_LIMIT // count:
                raise OverflowError(_mean_overflow_message(count, high))

        return _impl

    def _impl(low, high, count):
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


def _span_limit_check(low, high, count):
    """Refuse integer variances whose exact centering would overflow int64 -
    replaced per dtype by the overload below, a no-op for inexact input."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_span_limit_check)
def _span_limit_check_overload(low, high, count):
    if not isinstance(low, nb.types.Integer):

        def _impl(low, high, count):
            pass

        return _impl

    if not low.signed:

        def _impl(low, high, count):
            # uint64 values are non-negative, so high - low cannot wrap
            if count > 0 and high - low > INT64_DEVIATION_LIMIT:
                raise OverflowError(_span_overflow_message(low, high))

        return _impl

    def _impl(low, high, count):
        # signed: only the mixed-sign case can exceed the deviation limit,
        # and high - low itself would wrap there - compare without it
        if count > 0 and low < 0 and high >= 0 and high > INT64_DEVIATION_LIMIT + low:
            raise OverflowError(_span_overflow_message(low, high))

    return _impl


class MeanState(typing.NamedTuple):
    """``out`` doubles as the running sum until the finishing pass divides it
    by ``count``; ``low``/``high`` are the single-element global extrema
    scratch that feed the integer exactness gate."""

    out: npt.NDArray[np.inexact]
    count: npt.NDArray[np.int64]
    low: npt.NDArray[np.generic]
    high: npt.NDArray[np.generic]


def _mean_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    return MeanState(
        np.zeros(size, out_dtype),
        np.zeros(size, np.int64),
        np.empty(1, in_dtype),
        np.empty(1, in_dtype),
    )


def _mean_seed(st, a):
    # seed the global extrema scratch with the first value
    if len(a) > 0:
        st.low[0] = a[0]
        st.high[0] = a[0]


def _mean_update(st, ri, i, val, fill_value, ddof):
    st.count[ri] += 1
    st.out[ri] += val
    _track_extrema(val, st.low, st.high)


def _mean_gate(st, n_values):
    _mean_limit_check(st.low[0], st.high[0], n_values)


def _mean_finish(st, g, fill_value, ddof):
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


def _accumulate_std(val, st, ri):
    """Accumulate one value into the running var/std sums.

    Replaced per dtype by the overload below: integer input is centered on
    the first value of its group so that a large common offset cannot erase
    small deviations, inexact input keeps the plain sums."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_accumulate_std)
def _accumulate_std_overload(val, st, ri):
    if isinstance(val, nb.types.Integer):

        def _impl(val, st, ri):
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

    def _impl(val, st, ri):
        st.dev[ri] += val
        # the squared magnitude of the complex deviation is real, so out stays
        # real for complex input - matching np.var
        st.out[ri] += val.real * val.real + val.imag * val.imag

    return _impl


class StdState(typing.NamedTuple):
    """The running sums of the dispersion kernels: ``out`` accumulates the
    squared deviations in the real output dtype, ``dev`` the deviations
    themselves in the widened auxiliary dtype (complex stays complex),
    ``first`` anchors the integer centering and ``low``/``high`` feed the
    exactness gate.  Kept flat instead of extending ``MeanState`` - numba
    cannot type inherited NamedTuples."""

    out: npt.NDArray[np.floating]
    count: npt.NDArray[np.int64]
    dev: npt.NDArray[np.inexact]
    first: npt.NDArray[np.generic]
    low: npt.NDArray[np.generic]
    high: npt.NDArray[np.generic]


def _std_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    return StdState(
        np.zeros(size, out_dtype),
        np.zeros(size, np.int64),
        np.zeros(size, aux_dtype),
        np.empty(size, in_dtype),
        np.empty(1, in_dtype),
        np.empty(1, in_dtype),
    )


def _std_seed(st, a):
    # seed the global extrema scratch with the first value
    if len(a) > 0:
        st.low[0] = a[0]
        st.high[0] = a[0]


def _std_update(st, ri, i, val, fill_value, ddof):
    st.count[ri] += 1
    if isinstance(val, bool):
        st.dev[ri] += val
        # bool has no .real in numba and keeps the plain val * val product
        st.out[ri] += val * val
    else:
        _accumulate_std(val, st, ri)


def _std_gate(st, n_values):
    _span_limit_check(st.low[0], st.high[0], n_values)


def _std_finish(st, g, fill_value, ddof):
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
    doc="Standard deviation of the values of every group",
)


def _var_finish(st, g, fill_value, ddof):
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
    doc="Variance of the values of every group",
)
