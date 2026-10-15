"""Cumulative (N-to-N) operations: every input position gets an output.

The running value lives in ``running``, ``record`` copies it to the output
position of the value that just arrived - one pass over the data in input
order (these operations are order-sensitive, unlike the reductions).
"""

import typing

import numpy as np

from .compose import complex_less, kernel, rule_widen_int


class CumSumState(typing.NamedTuple):
    """``running`` accumulates per group, ``out`` records the running value
    at every input position."""

    running: np.ndarray
    out: np.ndarray


def _cumsum_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    # every position of out is written in the same pass that fills running
    return CumSumState(np.zeros(size, out_dtype), np.full(n_input, fill_value, out_dtype))


def _cumsum_update(st, ri, i, val, fill_value, ddof):
    st.running[ri] += val


def _cumsum_record(st, ri, i):
    st.out[i] = st.running[ri]


cumsum = kernel(
    "cumsum",
    init=_cumsum_init,
    update=_cumsum_update,
    record=_cumsum_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running sum within every group",
)


class CumProdState(typing.NamedTuple):
    """``running`` accumulates per group, ``out`` records the running value
    at every input position."""

    running: np.ndarray
    out: np.ndarray


def _cumprod_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return CumProdState(np.ones(size, out_dtype), np.full(n_input, fill_value, out_dtype))


def _cumprod_update(st, ri, i, val, fill_value, ddof):
    st.running[ri] *= val


def _cumprod_record(st, ri, i):
    st.out[i] = st.running[ri]


cumprod = kernel(
    "cumprod",
    init=_cumprod_init,
    update=_cumprod_update,
    record=_cumprod_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running product within every group",
)


class CumMaxState(typing.NamedTuple):
    """``running`` holds the maximum seen so far per group, ``out`` records
    it at every input position and ``pending`` marks the groups that have not
    seen a value yet (their first value always wins, a nan poisons the
    group)."""

    running: np.ndarray
    out: np.ndarray
    pending: np.ndarray


def _cummax_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return CumMaxState(
        np.full(size, fill_value, out_dtype),
        np.full(n_input, fill_value, out_dtype),
        np.ones(size, np.bool_),
    )


def _cummax_update(st, ri, i, val, fill_value, ddof):
    # select-form on purpose, see reductions._max_update
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.running[ri]
    st.running[ri] = val if (first or val != val or complex_less(cur, val)) else cur


def _cummax_record(st, ri, i):
    st.out[i] = st.running[ri]


cummax = kernel(
    "cummax",
    init=_cummax_init,
    update=_cummax_update,
    record=_cummax_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running maximum within every group",
)


class CumMinState(typing.NamedTuple):
    """``running`` holds the minimum seen so far per group, ``out`` records
    it at every input position and ``pending`` marks the groups that have not
    seen a value yet (their first value always wins, a nan poisons the
    group)."""

    running: np.ndarray
    out: np.ndarray
    pending: np.ndarray


def _cummin_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return CumMinState(
        np.full(size, fill_value, out_dtype),
        np.full(n_input, fill_value, out_dtype),
        np.ones(size, np.bool_),
    )


def _cummin_update(st, ri, i, val, fill_value, ddof):
    # select-form on purpose, see reductions._min_update
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.running[ri]
    st.running[ri] = val if (first or val != val or complex_less(val, cur)) else cur


def _cummin_record(st, ri, i):
    st.out[i] = st.running[ri]


cummin = kernel(
    "cummin",
    init=_cummin_init,
    update=_cummin_update,
    record=_cummin_record,
    rule=rule_widen_int,
    fill_default=0,
    nton=True,
    doc="Running minimum within every group",
)
