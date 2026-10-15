"""Two-pass averaging: mean, std and var, including the integer exactness policy.

Integer input is centered on the first value of each group and the global
min/max is tracked in the same accumulation pass; the exactness gates run
between the two passes and raise ``OverflowError`` for the same inputs in
every backend (see ``aggregate_common`` for the policy constants).
"""

import numba as nb
import numpy as np

from ..aggregate_common import EXACT_FLOAT64_INTEGER_LIMIT, INT64_DEVIATION_LIMIT, INT64_MIN
from .base import Aggregate2pass
from .jit import njit_stable


def _track_extrema(val, ret, origin):
    """Fold the global min/max tracking for the integer limit checks into the
    accumulation pass - replaced per dtype by the overload below, inexact
    input never tracks anything."""
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_track_extrema)
def _track_extrema_overload(val, ret, origin):
    if isinstance(val, nb.types.Integer):

        def _impl(val, ret, origin):
            # the two slots behind the per-group origins of the output hold
            # the running minimum and maximum
            size = len(ret)
            origin[size] = min(origin[size], val)
            origin[size + 1] = max(origin[size + 1], val)

        return _impl

    def _impl(val, ret, origin):
        pass

    return _impl


def _accumulate_mean(val, ret, counter, origin, ri):
    """Accumulate one value into the running mean sums.

    Replaced per dtype by the overload below: integer input also tracks the
    global min/max for the limit check between the passes.
    """
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_accumulate_mean)
def _accumulate_mean_overload(val, ret, counter, origin, ri):
    if isinstance(val, nb.types.Integer):

        def _impl(val, ret, counter, origin, ri):
            counter[ri] += 1
            ret[ri] += val
            _track_extrema(val, ret, origin)

        return _impl

    def _impl(val, ret, counter, origin, ri):
        counter[ri] += 1
        ret[ri] += val

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


class Mean(Aggregate2pass):
    forced_fill_value = 0
    counter_fill_value = 0
    counter_dtype = int
    # track the global min/max inside the accumulation pass and run the exact-
    # mean limit check between the passes - no extra data pass is spent
    integer_gate = True

    @staticmethod
    def _gate_check(low, high, count):
        _mean_limit_check(low, high, count)

    @staticmethod
    def _inner(ri, val, ret, counter, mean, origin, fill_value):
        _accumulate_mean(val, ret, counter, origin, ri)

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        return ret[ri] / counter[ri]


def _accumulate_std(val, ret, counter, mean, origin, ri):
    """Accumulate one value into the running var/std sums.

    Replaced per dtype by the overload below: integer input is centered on
    the first value of its group so that a large common offset cannot erase
    small deviations, inexact input keeps the plain sums.
    """
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(_accumulate_std)
def _accumulate_std_overload(val, ret, counter, mean, origin, ri):
    if isinstance(val, nb.types.Integer):

        def _impl(val, ret, counter, mean, origin, ri):
            if counter[ri] == 1:
                origin[ri] = val
            # two's-complement deviation, exact in int64 because the gate
            # check between the passes verified that the values span less
            # than +/-2**63; uint64 input wraps into the correct signed bit
            # pattern via the np.int64 casts
            dev = np.int64(val) - np.int64(origin[ri])
            mean[ri] += dev
            dev_sq = np.float64(dev)
            ret[ri] += dev_sq * dev_sq
            _track_extrema(val, ret, origin)

        return _impl

    def _impl(val, ret, counter, mean, origin, ri):
        mean[ri] += val
        # the squared magnitude of the complex deviation is real, so ret stays
        # real for complex input - matching np.var
        ret[ri] += val.real * val.real + val.imag * val.imag

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


class Std(Mean):
    mean_fill_value = 0
    # the running sums accumulate in float64 (complex input in complex128 via
    # result_type): integer deviations must not wrap in an int64 accumulator,
    # and for float32/complex64 input this matches the numpy accumulation
    # precision
    mean_dtype = np.float64
    integer_centering = True
    integer_gate = True

    @staticmethod
    def _gate_check(low, high, count):
        _span_limit_check(low, high, count)

    @staticmethod
    def _inner(ri, val, ret, counter, mean, origin, fill_value):
        counter[ri] += 1
        if isinstance(val, bool):
            mean[ri] += val
            # bool has no .real in numba and keeps the plain val * val product
            ret[ri] += val * val
        else:
            _accumulate_std(val, ret, counter, mean, origin, ri)

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        m = mean[ri]
        mean2 = m.real * m.real + m.imag * m.imag
        return np.sqrt((ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof))


class Var(Std):
    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        m = mean[ri]
        mean2 = m.real * m.real + m.imag * m.imag
        return (ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof)
