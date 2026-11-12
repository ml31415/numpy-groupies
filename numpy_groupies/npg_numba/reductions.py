"""Streaming one-pass reductions: sums, products, lengths, logic, min/max."""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.npg_numba.compose import (
    _fill_differs,
    build_custom_kernel,
    build_kernel,
    complex_less,
    output_size,
    rule_bool,
    rule_int64,
    rule_same,
    rule_sum,
    rule_sumofsquares,
    rule_widen_int,
)
from numpy_groupies.npg_numba.jit import njit_stable
from numpy_groupies.npg_numba.mt import chunk_count, should_use_mt
from numpy_groupies.validation import NEGATIVE_INDICES


class ReductionState(NamedTuple):
    """The state of the streaming reductions: ``out`` accumulates and doubles
    as the result, ``untouched`` marks the groups without a value (a custom
    fill value is applied after the loop), and ``track`` says whether that
    bookkeeping runs - only a non-default fill needs it (the default saves a
    store per element, ~0.1ms per 500k)."""

    out: npt.NDArray[np.number]
    untouched: npt.NDArray[np.bool_]
    track: bool


class SelectionState(NamedTuple):
    """The state of min/max: ``out`` holds the running best, ``pending`` the
    groups without a value - the first value wins, a nan poisons."""

    out: npt.NDArray[np.number]
    pending: npt.NDArray[np.bool_]


# jitted building blocks: numba ignores these annotations, the type checker
# needs them
def _sum_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    # the accumulation starts at 0 - a custom fill value is applied after
    # the loop
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, out_dtype), untouched, track)


def _sum_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] += val


def _sum_update_nan(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # branchless select-form twin of _sum_update - nans contribute nothing
    # and do not ready their group for a custom fill value; the guarded
    # equivalent turns the loop scalar and compiles ~3x slower.  The 32 bit
    # numpy scalars are named explicitly: numba maps python float/complex to
    # the 64 bit variants only, and a float32/complex64 nan on the plain path
    # would poison the accumulator
    valid = val == val
    if st.track:
        st.untouched[ri] = st.untouched[ri] & (val != val)
    if isinstance(val, (float, complex, np.float32, np.complex64)):
        st.out[ri] += val if valid else 0.0
    elif isinstance(val, (np.datetime64, np.timedelta64)):
        # temporal: NaT contributes nothing (NaT - NaT is NaT, not zero) -
        # the plain variant adds it, which the addition propagates like numpy
        st.out[ri] += val if valid else st.out[ri] - st.out[ri]
    else:
        # integer/bool have no nan - the plain path keeps wide uint64 exact
        # (a literal 0 would unify uint64 with int64 into float64)
        st.out[ri] += val


def _sum_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


@njit_stable
def _combine_untouched(acc: ReductionState, other: ReductionState) -> None:
    """The groups that neither chunk touched stay untouched."""
    if acc.track:
        untouched: np.ndarray = acc.untouched
        untouched &= other.untouched


def _sum_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: the sums add up."""
    out: np.ndarray = acc.out
    out += other.out
    _combine_untouched(acc, other)


sum = build_kernel(
    "sum",
    init=_sum_init,
    update=_sum_update,
    update_nan=_sum_update_nan,
    combine=_sum_combine,
    untouched=_sum_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_sum,
    doc="Sum of the values of every group",
)


def _prod_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.ones(size, out_dtype), untouched, track)


def _prod_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] *= val


def _prod_update_nan(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    valid = val == val
    if st.track:
        st.untouched[ri] = st.untouched[ri] & (val != val)
    if isinstance(val, (float, complex, np.float32, np.complex64)):
        st.out[ri] *= val if valid else 1.0
    else:
        # integer and bool input have no nan, see _sum_update_nan
        st.out[ri] *= val


def _prod_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _prod_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: the products multiply."""
    out: np.ndarray = acc.out
    out *= other.out
    _combine_untouched(acc, other)


prod = build_kernel(
    "prod",
    init=_prod_init,
    update=_prod_update,
    update_nan=_prod_update_nan,
    combine=_prod_combine,
    untouched=_prod_untouched,
    forced_fill=1,
    fill_default=1,
    rule=rule_widen_int,
    doc="Product of the values of every group",
)


def _len_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, out_dtype), untouched, track)


def _len_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] += 1


def _len_update_nan(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    valid = val == val
    if st.track:
        st.untouched[ri] = st.untouched[ri] & (val != val)
    st.out[ri] += 1 if valid else 0


def _len_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _len_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: the counts add up."""
    out: np.ndarray = acc.out
    out += other.out
    _combine_untouched(acc, other)


len = build_kernel(
    "len",
    init=_len_init,
    update=_len_update,
    update_nan=_len_update_nan,
    combine=_len_combine,
    untouched=_len_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_int64,
    doc="Amount of values of every group",
)


def _all_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.ones(size, np.bool_), untouched, track)


def _all_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] &= bool(val)


def _all_update_nan(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = st.untouched[ri] & (val != val)
    st.out[ri] &= (val != val) | bool(val)


def _all_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _all_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: a group is true only if it is true
    in both chunks."""
    out: np.ndarray = acc.out
    out &= other.out
    _combine_untouched(acc, other)


all = build_kernel(
    "all",
    init=_all_init,
    update=_all_update,
    update_nan=_all_update_nan,
    combine=_all_combine,
    untouched=_all_untouched,
    forced_fill=1,
    fill_default=False,
    rule=rule_bool,
    doc="Whether all values of every group are truthy",
)


def _any_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, np.bool_), untouched, track)


def _any_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] |= bool(val)


def _any_update_nan(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    valid = val == val
    if st.track:
        st.untouched[ri] = st.untouched[ri] & (val != val)
    st.out[ri] |= bool(val) & valid


def _any_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _any_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: a group is true if it is true in one
    of the chunks."""
    out: np.ndarray = acc.out
    out |= other.out
    _combine_untouched(acc, other)


any = build_kernel(
    "any",
    init=_any_init,
    update=_any_update,
    update_nan=_any_update_nan,
    combine=_any_combine,
    untouched=_any_untouched,
    forced_fill=0,
    fill_default=False,
    rule=rule_bool,
    doc="Whether any value of every group is truthy",
)


# The parallel path of first/last: the chunks are folded in index order (the
# first chunk with a value wins for the first, the last for the last), so the
# result is the sequential one bit for bit.  A chunk state is one value array
# and one flag per group - the weight of the sum's state in the cost model.
_MT_STATE_SIZE = 1


# The scan and the driver of the first and the last: neither reaches
# build_kernel (the per-element update call costs more than the bodies save,
# 1.06ms against 0.60ms on numba 0.68), and the loop lives in a function of
# its own with the state arrays passed in - allocating them inside measures
# ~25% slower, numba reloads the data pointers around every store.  ONE scan
# serves both paths, so the scan direction and the nan handling are written
# down once; the state is the value of every group and the groups that have
# one.
_MT_STATE_SIZE = 1


@functools.lru_cache(maxsize=4)
def _ordered_scan(nans: bool, last: bool) -> Callable[..., int]:
    """The scan that finds the first (or last) value of every group; ``nans``
    left out or taking the group over, ``last`` the direction."""

    if last:

        @njit_stable(name=f"{'nan' if nans else ''}last_scan")
        def scan(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            out: npt.NDArray[np.number],
            seen: npt.NDArray[np.bool_],
        ) -> int:
            """The last value of every group, and the fault of the indices.

            Walked backwards: a forward walk would scatter-write every value,
            which serializes the threads and is slower besides (813us against
            564us per million values).
            """
            fault = 0
            for i in range(group_idx.shape[0] - 1, -1, -1):
                ri = group_idx[i]
                if ri < 0:
                    fault = 1
                    continue
                if not seen[ri]:
                    val = a[i]
                    if nans and val != val:
                        # a nan does not take the group over: the last valid
                        # value is wanted
                        continue
                    seen[ri] = True
                    out[ri] = val
            return fault

    else:

        @njit_stable(name=f"{'nan' if nans else ''}first_scan")
        def scan(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            out: npt.NDArray[np.number],
            seen: npt.NDArray[np.bool_],
        ) -> int:
            """The first value of every group, and the fault of the indices -
            walked forwards, the seen flag short-circuiting the rest (the
            value is only read while the group is still pending)."""
            fault = 0
            for i in range(group_idx.shape[0]):
                ri = group_idx[i]
                if ri < 0:
                    fault = 1
                    continue
                if not seen[ri]:
                    val = a[i]
                    if nans and val != val:
                        # a nan does not take the group over: the first
                        # valid value is wanted
                        continue
                    seen[ri] = True
                    out[ri] = val
            return fault

    return scan


def _ordered_driver_factory(nans: bool, last: bool) -> Callable[..., npt.NDArray[np.number]]:
    """The compiled driver of the first (or the last): allocates the state,
    runs the scan, raises the fault complaint the parallel path cannot raise
    itself."""
    scan = _ordered_scan(nans, last)
    name = "last" if last else "first"

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
        # the state is allocated here, not in the scan itself: inside the
        # looping function it measures ~25% slower (numba reloads the data
        # pointers around every store)
        seen = np.zeros(size, np.bool_)
        fault = scan(group_idx, a, out, seen)
        if fault:
            raise ValueError(NEGATIVE_INDICES)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}{name}_driver")


def _ordered_mt_factory(last: bool) -> Callable[[bool, Callable[..., Any]], Any]:
    """The parallel passes of the first (or the last): the same scan per chunk
    of the data, the chunk states folded in index order (first wins for
    first, last for last)."""

    def factory(nans: bool, driver: Callable[..., npt.NDArray[np.number]]) -> Callable[..., npt.NDArray[np.number]]:
        scan = _ordered_scan(nans, last)
        name = "last" if last else "first"

        @njit_stable(name=f"{'nan' if nans else ''}{name}_fold")
        def fold(
            outs: npt.NDArray[np.number],
            flags: npt.NDArray[np.bool_],
        ) -> npt.NDArray[np.number]:
            """The states of the chunks, folded in the order of their index."""
            acc: npt.NDArray[np.number] = outs[0]
            if last:
                # the last chunk that has a value for a group wins
                for c in range(1, outs.shape[0]):
                    other_flag = flags[c]
                    for g in range(acc.size):
                        if other_flag[g]:
                            acc[g] = outs[c][g]
            else:
                # the first chunk that has a value for a group wins
                acc_flag = flags[0]
                for c in range(1, outs.shape[0]):
                    other_flag = flags[c]
                    for g in range(acc.size):
                        if other_flag[g] and not acc_flag[g]:
                            acc[g] = outs[c][g]
                            acc_flag[g] = True
            # the result is a copy of the first chunk's row - the chunk
            # states are released with the array they live in
            return acc.copy()

        def mt_passes(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: float,
            extra: Any,
            n_threads: int,
        ) -> npt.NDArray[np.number]:
            size = output_size(group_idx, size_arg)
            n_values = group_idx.shape[0]
            if not should_use_mt(size, n_threads, n_values, _MT_STATE_SIZE):
                # not worth it, or one thread only: the sequential driver,
                # with its result bit for bit
                return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
            n_chunks = chunk_count(n_threads, n_values)
            chunk = (n_values + n_chunks - 1) // n_chunks
            # the chunk states are the rows of two arrays, not a list of
            # arrays - a list inside the parallel region costs more than the
            # region saves (160us against 96us per 200k values on four threads)
            outs = np.full((n_chunks, size), fill_value, out_dtype)
            seen = np.zeros((n_chunks, size), np.bool_)
            # the fault of the indices of every chunk
            faults = np.zeros(n_chunks, np.int64)
            for c in nb.prange(n_chunks):
                start = c * chunk
                # np.minimum and not min: the module global 'min' is the min
                # kernel, not the builtin
                stop = np.minimum(start + chunk, n_values)
                faults[c] = scan(group_idx[start:stop], a[start:stop], outs[c], seen[c])
            if faults.any():
                raise ValueError(NEGATIVE_INDICES)
            return fold(outs, seen)

        return njit_stable(mt_passes, name=f"{'nan' if nans else ''}{name}_mt_passes", parallel=True)

    return factory


last = build_custom_kernel(
    "last",
    driver_factory=functools.partial(_ordered_driver_factory, last=True),
    mt_driver_factory=_ordered_mt_factory(True),
    rule=rule_same,
    fill_default=np.nan,
    doc="Last value of every group",
)


first = build_custom_kernel(
    "first",
    driver_factory=functools.partial(_ordered_driver_factory, last=False),
    mt_driver_factory=_ordered_mt_factory(False),
    rule=rule_same,
    fill_default=np.nan,
    doc="First value of every group",
)


def _allnan_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.ones(size, np.bool_), untouched, track)


def _allnan_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] &= val != val


def _allnan_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _allnan_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: a group is all-nan only if both
    chunks say so."""
    out: np.ndarray = acc.out
    out &= other.out
    _combine_untouched(acc, other)


allnan = build_kernel(
    "allnan",
    init=_allnan_init,
    update=_allnan_update,
    combine=_allnan_combine,
    untouched=_allnan_untouched,
    forced_fill=1,
    fill_default=False,
    rule=rule_bool,
    nans=False,
    doc="Whether all values of every group are nan",
)


def _anynan_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, np.bool_), untouched, track)


def _anynan_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] |= val != val


def _anynan_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _anynan_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: a group holds a nan if one of the
    chunks holds one."""
    out: np.ndarray = acc.out
    out |= other.out
    _combine_untouched(acc, other)


anynan = build_kernel(
    "anynan",
    init=_anynan_init,
    update=_anynan_update,
    combine=_anynan_combine,
    untouched=_anynan_untouched,
    forced_fill=0,
    fill_default=False,
    rule=rule_bool,
    nans=False,
    doc="Whether any value of every group is nan",
)


def _max_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> SelectionState:
    return SelectionState(np.full(size, fill_value, out_dtype), np.ones(size, np.bool_))


def _max_update(st: SelectionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form on purpose: the branchy equivalent compiles ~15x slower
    # under numba >= 0.66 as a separate njit call, and out[ri] must be read
    # into a local first (repeating the subscript keeps the slow codegen).
    # val != val is tested explicitly: a nan compares neither smaller nor
    # greater, so complex_less alone would leave a group whose nan arrives
    # late at the finite value, unlike np.max.  The nan variants run their
    # own twin body instead (see _max_update_nan).  The extra comparison is
    # the cheapest way: np.maximum (no numpy complex ordering) and the
    # branchy form both measured ~45% slower, and LLVM folds it away for
    # the integer dtypes
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.out[ri]
    # the comparison becomes a local before the select - a call defeats the
    # branchless form (2.5ms against 0.8ms per 500k values)
    best = complex_less(cur, val)
    # temporal: NaT takes its group over (ordered as the smallest tick, kept
    # by the guard), and it is the value that is not equal to itself.  The
    # condition is built without a short circuit: a branch that a fifth of
    # the values take mispredicts
    st.out[ri] = val if (first | (val != val) | (best & (cur == cur))) else cur


def _max_update_nan(st: SelectionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form twin of _max_update - nans neither win nor clear the
    # pending flag
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.out[ri]
    # the comparison becomes a local before the select - a call defeats the
    # branchless form (2.5ms against 0.8ms per 500k values)
    best = complex_less(cur, val)
    st.out[ri] = val if (valid & (first | best)) else cur


def _max_combine(acc: SelectionState, other: SelectionState, fill_value: float, extra: Any) -> None:
    """The states of two chunks, folded in the order of the data - the result
    of the sequential kernel, bit for bit.  The body repeats the select of
    ``_max_update`` instead of calling it: a call defeats the branchless
    form."""
    out: np.ndarray = acc.out
    pending = acc.pending
    other_out: np.ndarray = other.out
    other_pending: np.ndarray = other.pending
    for g in range(out.size):
        if not other_pending[g]:
            if pending[g]:
                out[g] = other_out[g]
                pending[g] = False
            else:
                val = other_out[g]
                # the temporal guard is dead code for the numeric dtypes - a
                # NaT of the running value poisons its group like a nan does
                if val != val or (out[g] == out[g] and complex_less(out[g], val)):
                    out[g] = val


def _max_combine_nan(acc: SelectionState, other: SelectionState, fill_value: float, extra: Any) -> None:
    """The twin of ``_max_combine`` for the nan variant - nans neither win nor
    clear the pending flag."""
    out: np.ndarray = acc.out
    pending = acc.pending
    other_out: np.ndarray = other.out
    other_pending: np.ndarray = other.pending
    for g in range(out.size):
        if not other_pending[g]:
            if pending[g]:
                out[g] = other_out[g]
                pending[g] = False
            elif complex_less(out[g], other_out[g]):
                out[g] = other_out[g]


def _min_combine(acc: SelectionState, other: SelectionState, fill_value: float, extra: Any) -> None:
    """The twin of ``_max_combine`` for the minimum."""
    out: np.ndarray = acc.out
    pending = acc.pending
    other_out: np.ndarray = other.out
    other_pending: np.ndarray = other.pending
    for g in range(out.size):
        if not other_pending[g]:
            if pending[g]:
                out[g] = other_out[g]
                pending[g] = False
            else:
                val = other_out[g]
                if val != val or complex_less(val, out[g]):
                    out[g] = val


def _min_combine_nan(acc: SelectionState, other: SelectionState, fill_value: float, extra: Any) -> None:
    """The twin of ``_max_combine_nan`` for the minimum."""
    for g in range(acc.out.size):
        if not other.pending[g]:
            if acc.pending[g]:
                acc.out[g] = other.out[g]
                acc.pending[g] = False
            elif complex_less(other.out[g], acc.out[g]):
                acc.out[g] = other.out[g]


max = build_kernel(
    "max",
    init=_max_init,
    update=_max_update,
    update_nan=_max_update_nan,
    combine=_max_combine,
    combine_nan=_max_combine_nan,
    rule=rule_same,
    fill_default=np.nan,
    doc="Maximum value of every group",
)


def _min_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> SelectionState:
    return SelectionState(np.full(size, fill_value, out_dtype), np.ones(size, np.bool_))


def _min_update(st: SelectionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form on purpose, see _max_update (why val != val is part of the
    # condition and what it costs)
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.out[ri]
    # the comparison becomes a local before the select - a call defeats the
    # branchless form (2.5ms against 0.8ms per 500k values)
    best = complex_less(val, cur)
    st.out[ri] = val if (first or val != val or best) else cur


def _min_update_nan(st: SelectionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form twin of _min_update, see _max_update_nan
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.out[ri]
    # the comparison becomes a local before the select - a call defeats the
    # branchless form (2.5ms against 0.8ms per 500k values)
    best = complex_less(val, cur)
    st.out[ri] = val if (valid & (first | best)) else cur


min = build_kernel(
    "min",
    init=_min_init,
    update=_min_update,
    update_nan=_min_update_nan,
    combine=_min_combine,
    combine_nan=_min_combine_nan,
    rule=rule_same,
    fill_default=np.nan,
    doc="Minimum value of every group",
)


def _sumofsquares_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    track = _fill_differs(fill_value, forced_fill)
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, out_dtype), untouched, track)


def _sumofsquares_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    # squared magnitude - real for complex like np.var; bool keeps the plain
    # val * val product (isinstance resolves at compile time per dtype)
    if isinstance(val, bool):
        st.out[ri] += val * val
    else:
        st.out[ri] += val.real * val.real + val.imag * val.imag


def _sumofsquares_update_nan(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    valid = val == val
    if st.track:
        st.untouched[ri] = st.untouched[ri] & (val != val)
    if isinstance(val, bool):
        st.out[ri] += val * val
    elif isinstance(val, (float, complex, np.float32, np.complex64)):
        st.out[ri] += (val.real * val.real + val.imag * val.imag) if valid else 0.0
    else:
        # integer input has no nan, see _sum_update_nan
        st.out[ri] += val.real * val.real + val.imag * val.imag


def _sumofsquares_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _sumofsquares_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """Two chunk states folded into one: the sums of the squared magnitudes
    add up."""
    out: np.ndarray = acc.out
    out += other.out
    _combine_untouched(acc, other)


sumofsquares = build_kernel(
    "sumofsquares",
    init=_sumofsquares_init,
    update=_sumofsquares_update,
    update_nan=_sumofsquares_update_nan,
    combine=_sumofsquares_combine,
    untouched=_sumofsquares_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_sumofsquares,
    doc="Sum of the squared magnitudes of the values of every group",
)
