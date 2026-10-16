"""Index-returning reductions: argmax and argmin."""

import numpy as np

from numpy_groupies.aggregate_numba.compose import (
    build_custom_kernel,
    complex_less,
    output_size,
    rule_int64,
)
from numpy_groupies.aggregate_numba.jit import njit_stable


def _argmax_driver_factory(nan_variant):
    # Neither of the two bodies reaches build_kernel: the per-element update
    # call of the composed driver costs more than these bodies save (measured
    # 1.06ms against 0.60ms on numba 0.68), so the loop lives here - but in a
    # function of its own, with the state arrays passed in: allocating them in
    # the looping function itself measures ~25% slower, numba reloads the data
    # pointers around every store instead of keeping them in registers.
    if nan_variant:

        @njit_stable
        def kernel_nan(group_idx, a, out, running, pending, fill_value):
            # select-form: nans neither win nor clear the pending flag, the
            # loop stays vectorizable
            bad = False
            for i in range(group_idx.shape[0]):
                ri = group_idx[i]
                if ri < 0:
                    bad = True
                    continue
                val = a[i]
                valid = val == val
                cur = running[ri]
                # the comparison is resolved into a local before the select
                # expression: reaching the expression as a call defeats the
                # branchless form entirely (2.5ms against 0.8ms per 500k)
                best = complex_less(cur, val)
                win = valid & (pending[ri] | best)
                pending[ri] = pending[ri] & (val != val)
                running[ri] = val if win else cur
                out[ri] = i if win else out[ri]
            if bad:
                raise ValueError("negative indices not supported")

        kernel = kernel_nan
    else:

        @njit_stable
        def kernel_plain(group_idx, a, out, running, pending, fill_value):
            # branchy on purpose, unlike the composed kernels: this loop writes
            # only when the group actually changed, and with a data dependent
            # index the select-form store could not vectorize anyway - the
            # plain scan of a 500k input with 1000 groups settles after a few
            # thousand values, so the branches predict perfectly
            #
            # the loop body is equivalent to
            #   win = (val != val) | pending[ri] | complex_less(running[ri], val)
            #   pending[ri] = False
            #   running[ri] = val if win else running[ri]
            #   out[ri] = (i if val == val else fill_value) if win else out[ri]
            # measured ~20% faster than that select-form spelling
            bad = False
            for i in range(group_idx.shape[0]):
                ri = group_idx[i]
                if ri < 0:
                    bad = True
                    continue
                val = a[i]
                cur = running[ri]
                # the comparison is resolved into a local before the
                # branch - reaching the test as a call costs more than
                # the loop around it (see the nan variant)
                best = complex_less(cur, val)
                if val != val:
                    # a nan poisons its group: it wins outright and the
                    # result falls back to the fill value
                    pending[ri] = False
                    running[ri] = val
                    out[ri] = fill_value
                elif pending[ri] or best:
                    pending[ri] = False
                    running[ri] = val
                    out[ri] = i
            if bad:
                raise ValueError("negative indices not supported")

        kernel = kernel_plain

    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        # once this resolved, every index is within [0, size) - the loop does
        # not need an upper bounds check (negative ones are flagged and raised
        # after the loop, a raise inside it would be a call barrier)
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        running = np.empty(size, a.dtype)
        pending = np.ones(size, np.bool_)
        kernel(group_idx, a, out, running, pending, fill_value)
        return out

    return njit_stable(driver, name=f"{'nan' if nan_variant else ''}argmax_driver")


argmax = build_custom_kernel(
    "argmax",
    driver_factory=_argmax_driver_factory,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the maximum value of every group",
)


def _argmin_driver_factory(nan_variant):
    # split loop/driver and body forms as in _argmax_driver_factory
    if nan_variant:

        @njit_stable
        def kernel_nan(group_idx, a, out, running, pending, fill_value):
            bad = False
            for i in range(group_idx.shape[0]):
                ri = group_idx[i]
                if ri < 0:
                    bad = True
                    continue
                val = a[i]
                valid = val == val
                cur = running[ri]
                # the comparison is resolved into a local first, see the
                # argmax kernel
                best = complex_less(val, cur)
                win = valid & (pending[ri] | best)
                pending[ri] = pending[ri] & (val != val)
                running[ri] = val if win else cur
                out[ri] = i if win else out[ri]
            if bad:
                raise ValueError("negative indices not supported")

        kernel = kernel_nan
    else:

        @njit_stable
        def kernel_plain(group_idx, a, out, running, pending, fill_value):
            bad = False
            for i in range(group_idx.shape[0]):
                ri = group_idx[i]
                if ri < 0:
                    bad = True
                    continue
                val = a[i]
                cur = running[ri]
                # the comparison is resolved into a local before the
                # branch - reaching the test as a call costs more than
                # the loop around it (see the nan variant)
                best = complex_less(val, cur)
                if val != val:
                    pending[ri] = False
                    running[ri] = val
                    out[ri] = fill_value
                elif pending[ri] or best:
                    pending[ri] = False
                    running[ri] = val
                    out[ri] = i
            if bad:
                raise ValueError("negative indices not supported")

        kernel = kernel_plain

    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        running = np.empty(size, a.dtype)
        pending = np.ones(size, np.bool_)
        kernel(group_idx, a, out, running, pending, fill_value)
        return out

    return njit_stable(driver, name=f"{'nan' if nan_variant else ''}argmin_driver")


argmin = build_custom_kernel(
    "argmin",
    driver_factory=_argmin_driver_factory,
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the minimum value of every group",
)
