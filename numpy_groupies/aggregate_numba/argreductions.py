"""Index-returning reductions: argmax and argmin."""

import functools
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

# The state of the scan: the index of the best value of every group, the
# value itself and the groups that have not seen a value yet.  The state of
# a chunk is one such state per chunk - three arrays of the size of the
# groups - and the cost model counts them against the two of the sum.
_MT_STATE_SIZE = 2


# The scan and the driver of the argmax and the argmin
#
# Neither of the two bodies reaches build_kernel: the per-element update
# call of the composed driver costs more than these bodies save (measured
# 1.06ms against 0.60ms on numba 0.68), so the loop lives here - but in a
# function of its own, with the state arrays passed in: allocating them in
# the looping function itself measures ~25% slower, numba reloads the data
# pointers around every store instead of keeping them in registers.
#
# ONE scan serves both paths: the sequential driver runs it over the whole
# input, the parallel path runs the very same scan over every chunk of the
# data and folds the states of the chunks afterwards.  The two paths cannot
# drift apart then - the tie-break (the first of two equal values wins) and
# the poisoning of a group by a nan are the delicate parts of it, and they
# are written down once.


@functools.lru_cache(maxsize=4)
def _arg_scan(nans: bool, greater: bool) -> Callable[..., int]:
    """The scan that finds the best value of every group.

    ``nans`` says whether the nans are left out (the nan variants) or
    whether they poison their group (the plain ones); ``greater`` says
    whether the best value is the largest (the argmax) or the smallest (the
    argmin) one.

    The scan is branchy on purpose: it writes only when the group actually
    changes, and with a data dependent index the select-form store could not
    vectorize anyway - the scan of a 500k input with 1000 groups settles
    after a few thousand values, so the branches predict perfectly.  The
    select-form spelling of the nan variants measured ~30% slower
    (2.19ms against 3.14ms per million values) - the branchless form writes
    the value of every group at every value, and the stores it does not have
    to do cost more than the branches it saves.
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

        ``out`` carries the index of the best value of every group, in the
        index of the whole input - ``start`` is the offset of the chunk the
        scan runs over.  ``running`` carries the value itself, and
        ``pending`` marks the groups that have not seen a value yet.

        A group that a nan poisoned says so in its ``running``: the nan
        stands there, and a nan compares neither smaller nor greater - so
        nothing wins the group any more.  A flag of its own would be a
        fourth array in the state of every chunk - and the nan says the same
        thing for free.

        The fault of the indices travels as a number and not as a flag in an
        array: an exception cannot leave the parallel region, so the
        complaint is made after it.
        """
        fault = 0
        for i in range(group_idx.shape[0]):
            ri = group_idx[i]
            if ri < 0:
                fault = 1
                continue
            val = a[i]
            cur = running[ri]
            # The comparison is resolved into a local before the branch:
            # reaching the expression as a call costs more than the loop
            # around it.  It is computed before the branch and not in it -
            # the comparison of a value that does not win the group is
            # cheap, and the loop is faster with it out of the way
            # (measured 872us against 1004us per million values).
            #
            # The direction of the comparison is the one of the operation:
            # the greater value wins for the argmax, the smaller one for
            # the argmin.
            if greater:
                best = complex_less(cur, val)
            else:
                best = complex_less(val, cur)
            if val != val:
                if not nans:
                    # a nan poisons its group: it wins outright and the
                    # result falls back to the fill value
                    pending[ri] = False
                    running[ri] = val
                    out[ri] = fill_value
                # the nan variant leaves the nans out
                continue
            if pending[ri] or best:
                # the first value of a group wins a tie
                pending[ri] = False
                running[ri] = val
                out[ri] = start + i
        return fault

    return scan


def _arg_driver_factory(nans: bool, greater: bool) -> Any:
    """The compiled driver of the argmax (the greater one) and the argmin.

    The driver is a thin wrapper around the scan: it allocates the state,
    runs the scan over the whole input and raises the complaint of a fault
    of the indices - a complaint the parallel path cannot raise itself.
    """
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
        # the state of the scan is allocated here and not in the scan itself:
        # allocating it in the looping function measures ~25% slower, numba
        # reloads the data pointers around every store instead of keeping
        # them in registers
        running = np.empty(size, a.dtype)
        pending = np.ones(size, np.bool_)
        fault = scan(group_idx, a, out, running, pending, fill_value, 0)
        if fault:
            raise ValueError(NEGATIVE_INDICES)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}{name}_driver")


# ---------------------------------------------------------------------------
# the parallel path
#
# The argmax and the argmin cannot be reduced in chunks the way the sums
# are: the value of a group is the one that wins, and the winner of two
# chunks is not the sum of the two.  Their parallel path splits the data
# into chunks and runs the very same scan over every chunk of them - the
# states of the chunks are folded afterwards, in the order of their index:
# the chunk with the better value wins, and a tie is won by the chunk that
# comes first, the way the scan keeps the first of two equal values.
#
# The result is therefore the one of the sequential path bit for bit.


def _arg_mt_factory(greater: bool) -> Callable[[bool, Callable[..., Any]], Any]:
    """The parallel passes of the argmax (the greater one) and the argmin."""

    def factory(nans: bool, driver: Callable[..., npt.NDArray[np.int64]]) -> Any:
        scan = _arg_scan(nans, greater)
        name = "argmax" if greater else "argmin"

        @njit_stable(name=f"{'nan' if nans else ''}{name}_fold")
        def fold(
            outs: npt.NDArray[np.int64],
            runs: npt.NDArray[np.number],
            pends: npt.NDArray[np.bool_],
        ) -> npt.NDArray[np.int64]:
            """The states of the chunks, folded in the order of their index.

            Every chunk holds the index of the best value of every group and
            the value itself; the chunk with the better value wins, and a tie
            is won by the chunk that comes first.  A nan in the value of a
            chunk says that the group is poisoned - the fill value the chunk
            wrote for it stands, and nothing wins the group any more.
            """
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
                        # the accumulator has no value yet: the one of the
                        # other chunk is the one of the group
                        acc_out[g] = other_out[g]
                        acc_run[g] = other_run[g]
                        acc_pend[g] = False
                        continue
                    if other_run[g] != other_run[g]:
                        # a nan of the other chunk poisons the group - the
                        # fill value it wrote for it stands
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
            # the states of the chunks are released with the array they live
            # in - the result is a copy of the row of the first chunk
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
                # not worth it, or only one thread to do it with: the
                # sequential driver, with its result bit for bit
                return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
            n_chunks = chunk_count(n_threads, n_values)
            chunk = (n_values + n_chunks - 1) // n_chunks
            # The state of every chunk: the index of the best value, the
            # value itself, the groups that have not seen a value yet and
            # the groups a nan poisoned.  The states are the rows of four
            # arrays and not a list of arrays: a list of them inside the
            # parallel region costs more than the region saves (measured
            # 160us against 96us per 200k values on four threads).
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
