"""Streaming one-pass reductions: sums, products, lengths, logic, min/max."""

import typing

import numpy as np
import numpy.typing as npt

from .compose import complex_less, kernel, rule_bool, rule_int64, rule_same, rule_sum, rule_sumofsquares, rule_widen_int


class ReductionState(typing.NamedTuple):
    """The state of the streaming reductions that fold every value into a
    per-group accumulator: ``out`` accumulates the reduction and doubles as
    the result, ``untouched`` marks the groups that have not received a value
    yet (a custom fill value is applied to them after the loop)."""

    out: npt.NDArray[np.generic]
    untouched: npt.NDArray[np.bool_]


class SelectionState(typing.NamedTuple):
    """The state of the select-form reductions (first, min, max): ``out``
    holds the selected value or the running best and doubles as the result,
    ``pending`` marks the groups that have not seen a value yet - their first
    value always wins, a nan poisons the group."""

    out: npt.NDArray[np.generic]
    pending: npt.NDArray[np.bool_]


def _sum_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    # the accumulation starts at 0 (the forced default) - a custom fill value
    # is applied to the untouched groups after the loop
    return ReductionState(np.zeros(size, out_dtype), np.ones(size, np.bool_))


def _sum_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    st.out[ri] += val


def _sum_untouched(st):
    return st.untouched


sum = kernel(
    "sum",
    init=_sum_init,
    update=_sum_update,
    untouched=_sum_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_sum,
    doc="Sum of the values of every group",
)


def _prod_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ReductionState(np.ones(size, out_dtype), np.ones(size, np.bool_))


def _prod_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    st.out[ri] *= val


def _prod_untouched(st):
    return st.untouched


prod = kernel(
    "prod",
    init=_prod_init,
    update=_prod_update,
    untouched=_prod_untouched,
    forced_fill=1,
    fill_default=1,
    rule=rule_widen_int,
    doc="Product of the values of every group",
)


def _len_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ReductionState(np.zeros(size, out_dtype), np.ones(size, np.bool_))


def _len_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    st.out[ri] += 1


def _len_untouched(st):
    return st.untouched


len = kernel(
    "len",
    init=_len_init,
    update=_len_update,
    untouched=_len_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_int64,
    doc="Amount of values of every group",
)


def _all_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ReductionState(np.ones(size, np.bool_), np.ones(size, np.bool_))


def _all_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    st.out[ri] &= bool(val)


def _all_untouched(st):
    return st.untouched


all = kernel(
    "all",
    init=_all_init,
    update=_all_update,
    untouched=_all_untouched,
    forced_fill=1,
    fill_default=False,
    rule=rule_bool,
    doc="Whether all values of every group are truthy",
)


def _any_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ReductionState(np.zeros(size, np.bool_), np.ones(size, np.bool_))


def _any_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    st.out[ri] |= bool(val)


def _any_untouched(st):
    return st.untouched


any = kernel(
    "any",
    init=_any_init,
    update=_any_update,
    untouched=_any_untouched,
    forced_fill=0,
    fill_default=False,
    rule=rule_bool,
    doc="Whether any value of every group is truthy",
)


class LastState(typing.NamedTuple):
    """The output array carries the last value of each group itself."""

    out: npt.NDArray[np.generic]


def _last_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    # untouched groups keep the fill value
    return LastState(np.full(size, fill_value, out_dtype))


def _last_update(st, ri, i, val, fill_value, ddof):
    st.out[ri] = val


last = kernel(
    "last",
    init=_last_init,
    update=_last_update,
    rule=rule_same,
    fill_default=np.nan,
    doc="Last value of every group",
)


def _first_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return SelectionState(np.full(size, fill_value, out_dtype), np.ones(size, np.bool_))


def _first_update(st, ri, i, val, fill_value, ddof):
    if st.pending[ri]:
        st.pending[ri] = False
        st.out[ri] = val


first = kernel(
    "first",
    init=_first_init,
    update=_first_update,
    rule=rule_same,
    fill_default=np.nan,
    doc="First value of every group",
)


def _allnan_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ReductionState(np.ones(size, np.bool_), np.ones(size, np.bool_))


def _allnan_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    st.out[ri] &= val != val


def _allnan_untouched(st):
    return st.untouched


allnan = kernel(
    "allnan",
    init=_allnan_init,
    update=_allnan_update,
    untouched=_allnan_untouched,
    forced_fill=1,
    fill_default=False,
    rule=rule_bool,
    nans=False,
    doc="Whether all values of every group are nan",
)


def _anynan_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ReductionState(np.zeros(size, np.bool_), np.ones(size, np.bool_))


def _anynan_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    st.out[ri] |= val != val


def _anynan_untouched(st):
    return st.untouched


anynan = kernel(
    "anynan",
    init=_anynan_init,
    update=_anynan_update,
    untouched=_anynan_untouched,
    forced_fill=0,
    fill_default=False,
    rule=rule_bool,
    nans=False,
    doc="Whether any value of every group is nan",
)


def _max_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return SelectionState(np.full(size, fill_value, out_dtype), np.ones(size, np.bool_))


def _max_update(st, ri, i, val, fill_value, ddof):
    # select-form on purpose: the branchy equivalent compiles ~15x slower
    # under numba >= 0.66 when reached as a separate njit call.  Note: out[ri]
    # must be read into a local first - repeating the subscript expression
    # keeps the slow codegen (aliasing).  val != val has to be tested
    # explicitly - a nan compares neither smaller nor greater, in numpy's
    # order for complex values as well as for floats, so complex_less alone
    # would leave a group whose nan arrives after its first value at the
    # finite value, unlike np.max (and unlike the arg kernels, which report
    # such a group as invalid).  The nan variants are unaffected, they filter
    # the nans out before the kernel runs.  The extra comparison is the
    # cheapest way to get there: np.maximum in the non-first branch (which
    # propagates nans on its own, but does not order complex values the way
    # numpy does) and the branchy form both measured ~45% slower, while for
    # integer dtypes LLVM folds the test away entirely.
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.out[ri]
    st.out[ri] = val if (first or val != val or complex_less(cur, val)) else cur


max = kernel(
    "max",
    init=_max_init,
    update=_max_update,
    rule=rule_same,
    fill_default=np.nan,
    doc="Maximum value of every group",
)


def _min_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return SelectionState(np.full(size, fill_value, out_dtype), np.ones(size, np.bool_))


def _min_update(st, ri, i, val, fill_value, ddof):
    # select-form on purpose, see _max_update (including why val != val is
    # part of the condition and what it costs)
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.out[ri]
    st.out[ri] = val if (first or val != val or complex_less(val, cur)) else cur


min = kernel(
    "min",
    init=_min_init,
    update=_min_update,
    rule=rule_same,
    fill_default=np.nan,
    doc="Minimum value of every group",
)


def _sumofsquares_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ReductionState(np.zeros(size, out_dtype), np.ones(size, np.bool_))


def _sumofsquares_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    # the squared magnitude of val - complex values give a real, sqrt-free
    # result like np.var; bool has no .real in numba and keeps the plain
    # val * val product (isinstance resolves at compile time per dtype)
    if isinstance(val, bool):
        st.out[ri] += val * val
    else:
        st.out[ri] += val.real * val.real + val.imag * val.imag


def _sumofsquares_untouched(st):
    return st.untouched


sumofsquares = kernel(
    "sumofsquares",
    init=_sumofsquares_init,
    update=_sumofsquares_update,
    untouched=_sumofsquares_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_sumofsquares,
    doc="Sum of the squared magnitudes of the values of every group",
)
