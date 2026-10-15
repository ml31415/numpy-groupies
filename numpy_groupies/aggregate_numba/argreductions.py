"""Index-returning reductions: argmax and argmin."""

import typing

import numpy as np

from .compose import complex_less, kernel, rule_int64


class ArgMaxState(typing.NamedTuple):
    """``out`` holds the input position of the running maximum and doubles as
    the result, ``running`` is the maximum itself and ``pending`` marks the
    groups that have not seen a value.  A nan poisons its group: the running
    value becomes a nan no finite value can beat and the result falls back to
    the fill value."""

    out: np.ndarray
    running: np.ndarray
    pending: np.ndarray


def _argmax_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ArgMaxState(np.full(size, fill_value, out_dtype), np.empty(size, in_dtype), np.ones(size, np.bool_))


def _argmax_update(st, ri, i, val, fill_value, ddof):
    if val != val:
        st.pending[ri] = False
        st.running[ri] = val
        st.out[ri] = fill_value
    elif st.pending[ri]:
        st.pending[ri] = False
        st.running[ri] = val
        st.out[ri] = i
    elif complex_less(st.running[ri], val):
        st.running[ri] = val
        st.out[ri] = i


argmax = kernel(
    "argmax",
    init=_argmax_init,
    update=_argmax_update,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the maximum value of every group",
)


class ArgMinState(typing.NamedTuple):
    """``out`` holds the input position of the running minimum and doubles as
    the result, ``running`` is the minimum itself and ``pending`` marks the
    groups that have not seen a value.  A nan poisons its group: the running
    value becomes a nan no finite value can beat and the result falls back to
    the fill value."""

    out: np.ndarray
    running: np.ndarray
    pending: np.ndarray


def _argmin_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ArgMinState(np.full(size, fill_value, out_dtype), np.empty(size, in_dtype), np.ones(size, np.bool_))


def _argmin_update(st, ri, i, val, fill_value, ddof):
    if val != val:
        st.pending[ri] = False
        st.running[ri] = val
        st.out[ri] = fill_value
    elif st.pending[ri]:
        st.pending[ri] = False
        st.running[ri] = val
        st.out[ri] = i
    elif complex_less(val, st.running[ri]):
        st.running[ri] = val
        st.out[ri] = i


argmin = kernel(
    "argmin",
    init=_argmin_init,
    update=_argmin_update,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the minimum value of every group",
)
