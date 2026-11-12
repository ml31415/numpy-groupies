"""Cumulative (N-to-N) operations: every input position gets an output.

``running`` accumulates per group, ``record`` copies it to the output position
of the value that just arrived - one pass in input order, hence order-sensitive
(unlike the reductions).
"""

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from numpy_groupies.npg_numba.compose import build_kernel, complex_less, rule_same, rule_widen_int


class CumulativeState(NamedTuple):
    """``running`` accumulates per group, ``out`` records it at every input position."""

    running: npt.NDArray[np.number]
    out: npt.NDArray[np.number]


def _cumsum_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> CumulativeState:
    # every position of out is written in the same pass that fills running
    return CumulativeState(np.zeros(size, out_dtype), np.full(n_input, fill_value, out_dtype))


def _cumsum_update(st: CumulativeState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    st.running[ri] += val


def _cumsum_update_nan(st: CumulativeState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form twin of _cumsum_update - a nan leaves the running value
    # unchanged, the position still records it; integer input has no nan and
    # keeps the plain path (a literal 0 would unify uint64 and int64 into float64)
    if isinstance(val, (float, complex, np.float32, np.complex64)):
        st.running[ri] += val if val == val else 0.0
    else:
        # integer and bool input have no nan, see _sum_update_nan
        st.running[ri] += val


def _cumsum_record(st: CumulativeState, ri: int, i: int) -> None:
    st.out[i] = st.running[ri]


cumsum = build_kernel(
    "cumsum",
    init=_cumsum_init,
    update=_cumsum_update,
    update_nan=_cumsum_update_nan,
    record=_cumsum_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running sum within every group",
)


def _cumprod_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> CumulativeState:
    return CumulativeState(np.ones(size, out_dtype), np.full(n_input, fill_value, out_dtype))


def _cumprod_update(st: CumulativeState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    st.running[ri] *= val


def _cumprod_update_nan(st: CumulativeState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if isinstance(val, (float, complex, np.float32, np.complex64)):
        st.running[ri] *= val if val == val else 1.0
    else:
        # integer and bool input have no nan, see _sum_update_nan
        st.running[ri] *= val


def _cumprod_record(st: CumulativeState, ri: int, i: int) -> None:
    st.out[i] = st.running[ri]


cumprod = build_kernel(
    "cumprod",
    init=_cumprod_init,
    update=_cumprod_update,
    update_nan=_cumprod_update_nan,
    record=_cumprod_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running product within every group",
)


class CumExtremaState(NamedTuple):
    """The running extrema: ``running`` the best value seen so far per group,
    ``out`` its record at every input position, ``pending`` the groups that have
    not seen a value yet (their first value always wins, a nan poisons the
    group).  Flat, not extending ``CumulativeState`` - numba cannot type
    inherited NamedTuples."""

    running: npt.NDArray[np.number]
    out: npt.NDArray[np.number]
    pending: npt.NDArray[np.bool_]


def _cummax_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> CumExtremaState:
    return CumExtremaState(
        np.full(size, fill_value, out_dtype),
        np.full(n_input, fill_value, out_dtype),
        np.ones(size, np.bool_),
    )


def _cummax_update(st: CumExtremaState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form on purpose, see reductions._max_update
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.running[ri]
    # the comparison is resolved into a local before the select expression -
    # reaching it as a call defeats the branchless form (2.5ms vs 0.8ms per 500k)
    best = complex_less(cur, val)
    # temporal input: a NaT poisons the rest of its group (numpy carries it
    # through the running maximum) - the comparison alone would order NaT as
    # the smallest tick and let the next value win; a NaT is the value not
    # equal to itself
    # no short circuit: python's or is a branch, and a branch a fifth of the
    # values take mispredicts (temporal NaTs sit at random places)
    st.running[ri] = val if (first | (val != val) | (best & (cur == cur))) else cur


def _cummax_update_nan(st: CumExtremaState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form twin of _cummax_update - nans neither win nor clear pending
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.running[ri]
    # the comparison is resolved into a local before the select expression -
    # reaching it as a call defeats the branchless form (2.5ms vs 0.8ms per 500k)
    best = complex_less(cur, val)
    st.running[ri] = val if (valid & (first | best)) else cur


def _cummax_record(st: CumExtremaState, ri: int, i: int) -> None:
    st.out[i] = st.running[ri]


cummax = build_kernel(
    "cummax",
    init=_cummax_init,
    update=_cummax_update,
    update_nan=_cummax_update_nan,
    record=_cummax_record,
    rule=rule_same,
    fill_default=0,
    nton=True,
    doc="Running maximum within every group",
)


def _cummin_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> CumExtremaState:
    return CumExtremaState(
        np.full(size, fill_value, out_dtype),
        np.full(n_input, fill_value, out_dtype),
        np.ones(size, np.bool_),
    )


def _cummin_update(st: CumExtremaState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form on purpose, see reductions._min_update
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.running[ri]
    # the comparison is resolved into a local before the select expression -
    # reaching it as a call defeats the branchless form (2.5ms vs 0.8ms per 500k)
    best = complex_less(val, cur)
    # the short circuit stays: without it this kernel is a third slower
    # (2.0ms vs 3.0ms per million values) - it saves the comparison branch of
    # a value that does not win the group
    st.running[ri] = val if (first or val != val or best) else cur


def _cummin_update_nan(st: CumExtremaState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.running[ri]
    # the comparison is resolved into a local before the select expression -
    # reaching it as a call defeats the branchless form (2.5ms vs 0.8ms per 500k)
    best = complex_less(val, cur)
    st.running[ri] = val if (valid & (first | best)) else cur


def _cummin_record(st: CumExtremaState, ri: int, i: int) -> None:
    st.out[i] = st.running[ri]


cummin = build_kernel(
    "cummin",
    init=_cummin_init,
    update=_cummin_update,
    update_nan=_cummin_update_nan,
    record=_cummin_record,
    rule=rule_same,
    fill_default=0,
    nton=True,
    doc="Running minimum within every group",
)
