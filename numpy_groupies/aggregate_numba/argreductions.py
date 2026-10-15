"""Index-returning reductions: argmax and argmin."""

import numpy as np

from .compose import build_custom_kernel, complex_less, rule_int64
from .jit import njit_stable


def _argmax_driver_factory(nan_variant):
    # The loop body is inlined on purpose: with numba >= 0.67 the composed
    # driver plus a separate njit update call compiles ~20x slower for this
    # reduction - the branching body defeats the automatic inliner (and
    # numba's inline="always" does not help either, measured against 0.68).
    @njit_stable
    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        # numba's np.max reduction is several times faster than a scalar
        # loop here; negative indices slip through and are caught inside
        # the loop below
        size = np.max(group_idx) + 1
        if size_arg >= 0:
            # an explicit output size may be larger than the largest
            # index, but never smaller
            if size > size_arg:
                raise ValueError("one or more indices in group_idx are too large")
            size = size_arg
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
            if val != val:
                # a nan poisons its group - no finite value can beat it and
                # the result falls back to the fill value; the nan variant
                # skips the value instead (folded away for the plain one)
                if nan_variant:
                    continue
                pending[ri] = False
                running[ri] = val
                out[ri] = fill_value
            elif pending[ri]:
                pending[ri] = False
                running[ri] = val
                out[ri] = i
            elif complex_less(running[ri], val):
                running[ri] = val
                out[ri] = i
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
        size = np.max(group_idx) + 1
        if size_arg >= 0:
            if size > size_arg:
                raise ValueError("one or more indices in group_idx are too large")
            size = size_arg
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
            if val != val:
                if nan_variant:
                    continue
                pending[ri] = False
                running[ri] = val
                out[ri] = fill_value
            elif pending[ri]:
                pending[ri] = False
                running[ri] = val
                out[ri] = i
            elif complex_less(val, running[ri]):
                running[ri] = val
                out[ri] = i
        return out

    return driver


argmin = build_custom_kernel(
    "argmin",
    driver_factory=_argmin_driver_factory,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the minimum value of every group",
)
