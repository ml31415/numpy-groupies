"""Index-returning reductions: argmax and argmin."""

import functools
from collections.abc import Callable
from typing import Any

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.npg_numba.compose import (
    build_custom_kernel,
    complex_less,
    output_size,
    rule_int64,
)
from numpy_groupies.npg_numba.jit import njit_stable
from numpy_groupies.npg_numba.mt import chunk_count, should_use_mt
from numpy_groupies.validation import NEGATIVE_INDICES

# the scan state per chunk: the best index, the value, the pending groups -
# three arrays of the size of the groups, counted against the sum's two
_MT_STATE_SIZE = 2


# The scan and the driver of the argmax and the argmin: neither body reaches
# build_kernel (the per-element update call costs more than the bodies save,
# 1.06ms against 0.60ms on numba 0.68), and the loop lives in a function of
# its own with the state arrays passed in - allocating them inside measures
# ~25% slower, numba reloads the data pointers around every store.  ONE scan
# serves both paths, so the tie-break (first of two equal values wins) and
# the nan poisoning are written down once.


@functools.lru_cache(maxsize=4)
def _arg_scan(nans: bool, greater: bool) -> Callable[..., int]:
    """The scan that finds the best value of every group; ``nans`` left out
    or poisoning, ``greater`` the direction.

    Branchy on purpose: it writes only when a group changes, and with a data
    dependent index the select form could not vectorize anyway - the groups
    settle after a few thousand values, so the branches predict perfectly
    (the select-form nan variant measured ~30% slower: the pointless stores
    cost more than the branches save).
    """

    @njit_stable(name=f"{'nan' if nans else ''}{'argmax' if greater else 'argmin'}_scan")
    def scan(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        out: npt.NDArray[np.int64],
        running: npt.NDArray[np.number],
        pending: npt.NDArray[np.bool_],
        fill_value: float,
        start: int,
    ) -> int:
        """The index of the best value of every group, and the fault of the indices.

        ``out`` is in whole-input coordinates (``start`` is the chunk
        offset), ``running`` carries the value, ``pending`` marks the unseen
        groups.  A nan-poisoned group says so in ``running`` - the nan
        compares neither way, so nothing wins it any more, no fourth flag
        array needed.  The fault travels as a number: an exception cannot
        leave the parallel region, the complaint is made after it.
        """
        fault = 0
        for i in range(group_idx.shape[0]):
            ri = group_idx[i]
            if ri < 0:
                fault = 1
                continue
            val = a[i]
            cur = running[ri]
            # resolved into a local before the branch, computed before it -
            # a call costs more than the loop around it (872us against 1004us
            # per million values).  NaT in the running value poisons its
            # group (numpy answers no position behind the first NaT), and it
            # is the value that is not equal to itself.
            live = cur == cur
            if greater:
                best = live and complex_less(cur, val)
            else:
                best = live and complex_less(val, cur)
            if val != val:
                if not nans:
                    pending[ri] = False
                    running[ri] = val
                    if isinstance(val, (np.datetime64, np.timedelta64)):
                        # temporal: NaT is a value, not missing - numpy
                        # answers the position of its first occurrence, and
                        # no tick beats the smallest
                        out[ri] = start + i
                    else:
                        # a nan poisons its group: it wins, the result
                        # falls back to the fill value
                        out[ri] = fill_value
                # the nan variant skips the NaT too, deliberately unlike
                # np.nanargmin/nanargmax, which answer the NaT position where
                # np.nanmin/nanmax skip it (numpy/numpy#5222, #14717, now
                # #33207) - the nan forms of the package skip the missing
                # value of every dtype, as their contract says
                continue
            if pending[ri] or best:
                # the first value of a group wins a tie
                pending[ri] = False
                running[ri] = val
                out[ri] = start + i
        return fault

    return scan


def _arg_driver_factory(nans: bool, greater: bool) -> Callable[..., npt.NDArray[np.int64]]:
    """The compiled driver of the argmax (the greater one) and the argmin:
    allocates the state, runs the scan, raises the fault complaint the
    parallel path cannot raise itself."""
    scan = _arg_scan(nans, greater)
    name = "argmax" if greater else "argmin"

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
    ) -> npt.NDArray[np.int64]:
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        # the state is allocated here, not in the scan itself: inside the
        # looping function it measures ~25% slower (numba reloads the data
        # pointers around every store)
        running = np.empty(size, a.dtype)
        pending = np.ones(size, np.bool_)
        fault = scan(group_idx, a, out, running, pending, fill_value, 0)
        if fault:
            raise ValueError(NEGATIVE_INDICES)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}{name}_driver")


# ---------------------------------------------------------------------------
# the parallel path - the same scan per chunk of the data, the chunk states
# folded in index order (the better value wins, a tie the earlier chunk),
# so the result is the sequential one bit for bit


def _arg_mt_factory(greater: bool) -> Callable[[bool, Callable[..., Any]], Any]:
    """The parallel passes of the argmax (the greater one) and the argmin."""

    def factory(nans: bool, driver: Callable[..., npt.NDArray[np.int64]]) -> Callable[..., npt.NDArray[np.int64]]:
        scan = _arg_scan(nans, greater)
        name = "argmax" if greater else "argmin"

        @njit_stable(name=f"{'nan' if nans else ''}{name}_fold")
        def fold(
            outs: npt.NDArray[np.int64],
            runs: npt.NDArray[np.number],
            pends: npt.NDArray[np.bool_],
        ) -> npt.NDArray[np.int64]:
            """The chunk states, folded in index order: the better value
            wins, a tie the earlier chunk; a nan says the group is poisoned -
            the fill value it wrote stands, nothing wins it any more."""
            acc_out: npt.NDArray[np.int64] = outs[0]
            acc_run = runs[0]
            acc_pend = pends[0]
            for c in range(1, outs.shape[0]):
                other_out = outs[c]
                other_run = runs[c]
                other_pend = pends[c]
                for g in range(acc_out.size):
                    if other_pend[g]:
                        # the other chunk has no value for this group
                        continue
                    if acc_pend[g]:
                        # the accumulator has no value yet: the other
                        # chunk's value is the group's
                        acc_out[g] = other_out[g]
                        acc_run[g] = other_run[g]
                        acc_pend[g] = False
                        continue
                    if other_run[g] != other_run[g]:
                        # a nan of the other chunk poisons the group - its
                        # fill value stands
                        acc_out[g] = other_out[g]
                        acc_run[g] = other_run[g]
                        continue
                    if acc_run[g] != acc_run[g]:
                        # the group is poisoned already: nothing wins it any
                        # more
                        continue
                    if greater:
                        if complex_less(acc_run[g], other_run[g]):
                            acc_out[g] = other_out[g]
                            acc_run[g] = other_run[g]
                    else:
                        if complex_less(other_run[g], acc_run[g]):
                            acc_out[g] = other_out[g]
                            acc_run[g] = other_run[g]
            # the result is a copy of the first chunk's row - the chunk
            # states are released with the array they live in
            return acc_out.copy()

        def mt_passes(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: float,
            extra: Any,
            n_threads: int,
        ) -> npt.NDArray[np.int64]:
            size = output_size(group_idx, size_arg)
            n_values = group_idx.shape[0]
            if not should_use_mt(size, n_threads, n_values, _MT_STATE_SIZE):
                # not worth it, or one thread only: the sequential driver,
                # with its result bit for bit
                return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
            n_chunks = chunk_count(n_threads, n_values)
            chunk = (n_values + n_chunks - 1) // n_chunks
            # the chunk states are the rows of four arrays, not a list of
            # arrays - a list inside the parallel region costs more than the
            # region saves (160us against 96us per 200k values on four threads)
            outs = np.full((n_chunks, size), fill_value, out_dtype)
            runs = np.empty((n_chunks, size), a.dtype)
            pends = np.ones((n_chunks, size), np.bool_)
            # the fault of the indices of every chunk
            faults = np.zeros(n_chunks, np.int64)
            for c in nb.prange(n_chunks):
                start = c * chunk
                # np.minimum and not min: the module global 'min' is the min
                # kernel, not the builtin
                stop = np.minimum(start + chunk, n_values)
                faults[c] = scan(
                    group_idx[start:stop],
                    a[start:stop],
                    outs[c],
                    runs[c],
                    pends[c],
                    fill_value,
                    start,
                )
            if faults.any():
                raise ValueError(NEGATIVE_INDICES)
            return fold(outs, runs, pends)

        return njit_stable(mt_passes, name=f"{'nan' if nans else ''}{name}_mt_passes", parallel=True)

    return factory


argmax = build_custom_kernel(
    "argmax",
    driver_factory=functools.partial(_arg_driver_factory, greater=True),
    mt_driver_factory=_arg_mt_factory(True),
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the maximum value of every group",
)


argmin = build_custom_kernel(
    "argmin",
    driver_factory=functools.partial(_arg_driver_factory, greater=False),
    mt_driver_factory=_arg_mt_factory(False),
    rule=rule_int64,
    fill_default=-1,
    doc="Position of the minimum value of every group",
)
