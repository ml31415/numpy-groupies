"""Index-returning reductions: argmax and argmin."""

import numpy as np

from .compose import build_custom_kernel, complex_less, output_size, rule_int64
from .jit import njit_stable


def _argmax_driver_factory(nan_variant):
    # The loop body is inlined on purpose: with numba >= 0.67 the composed
    # driver plus a separate njit update call compiles ~20x slower for this
    # reduction - the branching body defeats the automatic inliner (and
    # numba's inline="always" does not help either, measured against 0.68).
    @njit_stable
    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        running = np.empty(size, a.dtype)
        pending = np.ones(size, np.bool_)
        for i in range(len(group_idx)):
            ri = group_idx[i]
            if ri < 0:
                raise ValueError("negative indices not supported")
            if ri >= size:
                raise ValueError("one or more indices in group_idx are too large")
            val = a[i]
            valid = val == val
            best = complex_less(running[ri], val)
            if nan_variant:
                # select-form: nans neither win nor clear the pending flag,
                # the loop stays vectorizable (folded away for the plain one)
                win = valid & (pending[ri] | best)
                pending[ri] = pending[ri] & (val != val)
                running[ri] = val if win else running[ri]
                out[ri] = i if win else out[ri]
            else:
                # a nan poisons its group: it wins outright and the result
                # falls back to the fill value
                win = (val != val) | pending[ri] | best
                pending[ri] = False
                running[ri] = val if win else running[ri]
                out[ri] = (i if valid else fill_value) if win else out[ri]
        return out

    return driver


argmax = build_custom_kernel(
    "argmax",
    driver_factory=_argmax_driver_factory,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the maximum value of every group",
)


def _argmin_driver_factory(nan_variant):
    # inlined on purpose, see _argmax_driver_factory
    @njit_stable
    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        running = np.empty(size, a.dtype)
        pending = np.ones(size, np.bool_)
        for i in range(len(group_idx)):
            ri = group_idx[i]
            if ri < 0:
                raise ValueError("negative indices not supported")
            if ri >= size:
                raise ValueError("one or more indices in group_idx are too large")
            val = a[i]
            valid = val == val
            best = complex_less(val, running[ri])
            if nan_variant:
                # select-form, see _argmax_driver_factory
                win = valid & (pending[ri] | best)
                pending[ri] = pending[ri] & (val != val)
                running[ri] = val if win else running[ri]
                out[ri] = i if win else out[ri]
            else:
                # a nan poisons its group, see _argmax_driver_factory
                win = (val != val) | pending[ri] | best
                pending[ri] = False
                running[ri] = val if win else running[ri]
                out[ri] = (i if valid else fill_value) if win else out[ri]
        return out

    return driver


argmin = build_custom_kernel(
    "argmin",
    driver_factory=_argmin_driver_factory,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the minimum value of every group",
)
