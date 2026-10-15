"""Index-returning reductions: argmax and argmin."""

import typing

import numpy as np
import numpy.typing as npt

from .compose import build_kernel, complex_less, rule_int64


class ArgState(typing.NamedTuple):
    """``out`` holds the input position of the running best value and doubles
    as the result, ``running`` is the best value itself and ``pending`` marks
    the groups that have not seen a value.  A nan poisons its group: the
    running value becomes a nan no finite value can beat and the result falls
    back to the fill value."""

    out: npt.NDArray[np.int64]
    running: npt.NDArray[np.generic]
    pending: npt.NDArray[np.bool_]


def _argmax_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ArgState(np.full(size, fill_value, out_dtype), np.empty(size, in_dtype), np.ones(size, np.bool_))


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


argmax = build_kernel(
    "argmax",
    init=_argmax_init,
    update=_argmax_update,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the maximum value of every group",
)


def _argmin_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return ArgState(np.full(size, fill_value, out_dtype), np.empty(size, in_dtype), np.ones(size, np.bool_))


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


argmin = build_kernel(
    "argmin",
    init=_argmin_init,
    update=_argmin_update,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the minimum value of every group",
)
