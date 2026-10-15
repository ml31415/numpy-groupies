"""Cumulative (N-to-N) operations: every input position gets an output.

The running value lives in ``running``, ``record`` copies it to the output
position of the value that just arrived - one pass over the data in input
order (these operations are order-sensitive, unlike the reductions).
"""

import typing

import numpy as np
import numpy.typing as npt

from .compose import build_kernel, complex_less, rule_widen_int


class CumulativeState(typing.NamedTuple):
    """``running`` accumulates per group, ``out`` records the running value
    at every input position."""

    running: npt.NDArray[np.generic]
    out: npt.NDArray[np.generic]


def _cumsum_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    # every position of out is written in the same pass that fills running
    return CumulativeState(np.zeros(size, out_dtype), np.full(n_input, fill_value, out_dtype))


def _cumsum_update(st, ri, i, val, fill_value, ddof):
    st.running[ri] += val


def _cumsum_update_nan(st, ri, i, val, fill_value, ddof):
    # select-form twin of _cumsum_update - a nan leaves the running value
    # unchanged, the position still records it.  integer input has no nan
    # and keeps the plain path (a python literal 0 would unify uint64 with
    # int64 into float64)
    if isinstance(val, (float, complex, np.float32, np.complex64)):
        st.running[ri] += val if val == val else 0.0
    else:
        # integer and bool input have no nan, see _sum_update_nan
        st.running[ri] += val


def _cumsum_record(st, ri, i):
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


def _cumprod_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    return CumulativeState(np.ones(size, out_dtype), np.full(n_input, fill_value, out_dtype))


def _cumprod_update(st, ri, i, val, fill_value, ddof):
    st.running[ri] *= val


def _cumprod_update_nan(st, ri, i, val, fill_value, ddof):
    if isinstance(val, (float, complex, np.float32, np.complex64)):
        st.running[ri] *= val if val == val else 1.0
    else:
        # integer and bool input have no nan, see _sum_update_nan
        st.running[ri] *= val


def _cumprod_record(st, ri, i):
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


class CumExtremaState(typing.NamedTuple):
    """The state of the running extrema: ``running`` holds the best value
    seen so far per group, ``out`` records it at every input position and
    ``pending`` marks the groups that have not seen a value yet (their first
    value always wins, a nan poisons the group).

    Kept flat instead of extending ``CumulativeState`` - numba cannot type
    inherited NamedTuples."""

    running: npt.NDArray[np.generic]
    out: npt.NDArray[np.generic]
    pending: npt.NDArray[np.bool_]


def _cummax_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    return CumExtremaState(
        np.full(size, fill_value, out_dtype),
        np.full(n_input, fill_value, out_dtype),
        np.ones(size, np.bool_),
    )


def _cummax_update(st, ri, i, val, fill_value, ddof):
    # select-form on purpose, see reductions._max_update
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.running[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
    best = complex_less(cur, val)
    st.running[ri] = val if (first or val != val or best) else cur


def _cummax_update_nan(st, ri, i, val, fill_value, ddof):
    # select-form twin of _cummax_update - nans neither win nor clear the
    # pending flag
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.running[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
    best = complex_less(cur, val)
    st.running[ri] = val if (valid & (first | best)) else cur


def _cummax_record(st, ri, i):
    st.out[i] = st.running[ri]


cummax = build_kernel(
    "cummax",
    init=_cummax_init,
    update=_cummax_update,
    update_nan=_cummax_update_nan,
    record=_cummax_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running maximum within every group",
)


def _cummin_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    return CumExtremaState(
        np.full(size, fill_value, out_dtype),
        np.full(n_input, fill_value, out_dtype),
        np.ones(size, np.bool_),
    )


def _cummin_update(st, ri, i, val, fill_value, ddof):
    # select-form on purpose, see reductions._min_update
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.running[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
    best = complex_less(val, cur)
    st.running[ri] = val if (first or val != val or best) else cur


def _cummin_update_nan(st, ri, i, val, fill_value, ddof):
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.running[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
    best = complex_less(val, cur)
    st.running[ri] = val if (valid & (first | best)) else cur


def _cummin_record(st, ri, i):
    st.out[i] = st.running[ri]


cummin = build_kernel(
    "cummin",
    init=_cummin_init,
    update=_cummin_update,
    update_nan=_cummin_update_nan,
    record=_cummin_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running minimum within every group",
)
