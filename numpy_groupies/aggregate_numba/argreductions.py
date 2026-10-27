"""Index-returning reductions: argmax and argmin."""

from collections.abc import Callable
from typing import Any

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.compose import (
    build_custom_kernel,
    complex_less,
    output_size,
    rule_int64,
)
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numba.mt import chunk_count, should_use_mt
from numpy_groupies.validation import NEGATIVE_INDICES

# The state of the parallel path of the argmax and the argmin: the best
# value of every group, the index it was found at and the groups that a nan
# poisoned.  The state of a chunk holds three arrays of the size of the
# groups - the same weight for the cost model as the state of the sum, whose
# two arrays are one of them and a flag.
_MT_STATE_SIZE = 2


def _argmax_driver_factory(nan_variant: bool) -> Any:
    # Neither of the two bodies reaches build_kernel: the per-element update
    # call of the composed driver costs more than these bodies save (measured
    # 1.06ms against 0.60ms on numba 0.68), so the loop lives here - but in a
    # function of its own, with the state arrays passed in: allocating them in
    # the looping function itself measures ~25% slower, numba reloads the data
    # pointers around every store instead of keeping them in registers.
    if nan_variant:

        @njit_stable
        def kernel_nan(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            out: npt.NDArray[np.number],
            running: npt.NDArray[np.number],
            pending: npt.NDArray[np.bool_],
            fill_value: float,
        ) -> None:
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
                raise ValueError(NEGATIVE_INDICES)

        kernel = kernel_nan
    else:

        @njit_stable
        def kernel_plain(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            out: npt.NDArray[np.number],
            running: npt.NDArray[np.number],
            pending: npt.NDArray[np.bool_],
            fill_value: float,
        ) -> None:
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
                raise ValueError(NEGATIVE_INDICES)

        kernel = kernel_plain

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
    ) -> npt.NDArray[np.number]:
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


# ---------------------------------------------------------------------------
# the parallel path
#
# The argmax and the argmin cannot be reduced in chunks the way the sums
# are: the value of a group is the one that wins, and the winner of two
# chunks is not the sum of the two.  Their parallel path splits the data
# into chunks, lets every chunk look for the best value of every group, and
# folds the chunks in the order of their index - the chunk that has the
# better value wins, and a tie is won by the chunk that comes first, the
# way the sequential kernel keeps the first of two equal values.
#
# The result is therefore the one of the sequential path bit for bit.


def _arg_mt_factory(greater: bool) -> Callable[[bool, Callable[..., Any]], Any]:
    """The parallel passes of the argmax (the greater one) and the argmin.

    ``greater`` says which of the two the factory builds: the maximum of a
    group or the minimum of it.
    """

    def factory(nans: bool, driver: Callable[..., npt.NDArray[np.integer]]) -> Any:
        name = "argmax" if greater else "argmin"

        @njit_stable(name=f"{'nan' if nans else ''}{name}_chunk_scan")
        def scan_chunk(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            best: npt.NDArray[np.number],
            best_idx: npt.NDArray[np.int64],
            poisoned: npt.NDArray[np.bool_],
            start: int,
        ) -> int:
            """The best value of every group within one chunk of the data.

            The index of the value is the one of the whole input, so that
            the fold of the chunks can tell the values of two chunks apart.

            The fault of the indices travels as a number and not as a flag in
            an array: an exception cannot leave the parallel region, so the
            complaint is made after it.
            """
            fault = 0
            for i in range(group_idx.shape[0]):
                ri = group_idx[i]
                if ri < 0:
                    fault = 1
                    continue
                val = a[i]
                if val != val:
                    if not nans:
                        # a nan poisons its group: it wins outright and the
                        # result falls back to the fill value
                        poisoned[ri] = True
                    # the nan variant leaves the nans out
                    continue
                if best_idx[ri] < 0:
                    # the first value of the group is the one that wins a tie
                    best[ri] = val
                    best_idx[ri] = start + i
                elif greater:
                    if complex_less(best[ri], val):
                        best[ri] = val
                        best_idx[ri] = start + i
                else:
                    if complex_less(val, best[ri]):
                        best[ri] = val
                        best_idx[ri] = start + i
            return fault

        def mt_passes(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: float,
            extra: Any,
            n_threads: int,
        ) -> npt.NDArray[np.integer]:
            size = output_size(group_idx, size_arg)
            n_values = group_idx.shape[0]
            if not should_use_mt(size, n_threads, n_values, _MT_STATE_SIZE):
                # not worth it, or only one thread to do it with: the
                # sequential driver, with its result bit for bit
                return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
            n_chunks = chunk_count(n_threads, n_values)
            chunk = (n_values + n_chunks - 1) // n_chunks
            # The state of every chunk: the best value of every group, the
            # index it was found at and the groups a nan poisoned.  The
            # states are the rows of three arrays and not a list of arrays: a
            # list of them inside the parallel region costs more than the
            # region saves.
            best = np.empty((n_chunks, size), a.dtype)
            best_idx = np.full((n_chunks, size), -1, np.int64)
            poisoned = np.zeros((n_chunks, size), np.bool_)
            # the fault of the indices of every chunk
            faults = np.zeros(n_chunks, np.int64)
            for c in nb.prange(n_chunks):
                start = c * chunk
                # np.minimum and not min: the module global 'min' is the min
                # kernel, not the builtin
                stop = np.minimum(start + chunk, n_values)
                faults[c] = scan_chunk(group_idx[start:stop], a[start:stop], best[c], best_idx[c], poisoned[c], start)
            # the fold in the order of the chunks: the chunk with the better
            # value wins, and a tie is won by the chunk that comes first
            acc_best: npt.NDArray[np.number] = best[0]
            acc_idx: npt.NDArray[np.int64] = best_idx[0]
            acc_poisoned: npt.NDArray[np.bool_] = poisoned[0]
            for c in range(1, n_chunks):
                other_best: npt.NDArray[np.number] = best[c]
                other_idx: npt.NDArray[np.int64] = best_idx[c]
                other_poisoned: npt.NDArray[np.bool_] = poisoned[c]
                for g in range(size):
                    if other_idx[g] >= 0:
                        if acc_idx[g] < 0:
                            acc_best[g] = other_best[g]
                            acc_idx[g] = other_idx[g]
                        elif greater:
                            if complex_less(acc_best[g], other_best[g]):
                                acc_best[g] = other_best[g]
                                acc_idx[g] = other_idx[g]
                        else:
                            if complex_less(other_best[g], acc_best[g]):
                                acc_best[g] = other_best[g]
                                acc_idx[g] = other_idx[g]
                    if other_poisoned[g]:
                        # a nan of any chunk poisons the group
                        acc_poisoned[g] = True
            if faults.any():
                raise ValueError(NEGATIVE_INDICES)
            # the groups that received a value and were not poisoned get the
            # index of their best value, the others keep the fill value
            out = np.full(size, fill_value, out_dtype)
            for g in range(size):
                if acc_idx[g] >= 0 and not acc_poisoned[g]:
                    out[g] = acc_idx[g]
            return out

        return njit_stable(mt_passes, name=f"{'nan' if nans else ''}{name}_mt_passes", parallel=True)

    return factory


argmax = build_custom_kernel(
    "argmax",
    driver_factory=_argmax_driver_factory,
    mt_driver_factory=_arg_mt_factory(True),
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the maximum value of every group",
)


def _argmin_driver_factory(nan_variant: bool) -> Any:
    # split loop/driver and body forms as in _argmax_driver_factory
    if nan_variant:

        @njit_stable
        def kernel_nan(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            out: npt.NDArray[np.number],
            running: npt.NDArray[np.number],
            pending: npt.NDArray[np.bool_],
            fill_value: float,
        ) -> None:
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
                raise ValueError(NEGATIVE_INDICES)

        kernel = kernel_nan
    else:

        @njit_stable
        def kernel_plain(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            out: npt.NDArray[np.number],
            running: npt.NDArray[np.number],
            pending: npt.NDArray[np.bool_],
            fill_value: float,
        ) -> None:
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
                raise ValueError(NEGATIVE_INDICES)

        kernel = kernel_plain

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
    ) -> npt.NDArray[np.number]:
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
    mt_driver_factory=_arg_mt_factory(False),
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the minimum value of every group",
)
