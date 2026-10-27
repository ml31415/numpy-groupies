"""Streaming one-pass reductions: sums, products, lengths, logic, min/max."""

from collections.abc import Callable
from typing import Any, NamedTuple

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.compose import (
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
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numba.mt import chunk_count, should_use_mt
from numpy_groupies.validation import NEGATIVE_INDICES


class ReductionState(NamedTuple):
    """The state of the streaming reductions that fold every value into a
    per-group accumulator: ``out`` accumulates the reduction and doubles as
    the result, ``untouched`` marks the groups that have not received a value
    yet (a custom fill value is applied to them after the loop) and ``track``
    says whether that bookkeeping runs at all - only a fill value differing
    from the forced default needs it, the default keeps the per-element loop
    free of the extra store (measured ~0.1ms per 500k values)."""

    out: npt.NDArray[np.number]
    untouched: npt.NDArray[np.bool_]
    track: bool


class SelectionState(NamedTuple):
    """The state of the select-form reductions (min, max): ``out``
    holds the selected value or the running best and doubles as the result,
    ``pending`` marks the groups that have not seen a value yet - their first
    value always wins, a nan poisons the group."""

    out: npt.NDArray[np.number]
    pending: npt.NDArray[np.bool_]


# the building blocks of this file are jitted: numba compiles them from the
# argument types and ignores these annotations, the type checker needs them
def _sum_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> ReductionState:
    # the accumulation starts at 0 (the forced default) - a custom fill value
    # is applied to the untouched groups after the loop
    track = fill_value != forced_fill
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, out_dtype), untouched, track)


def _sum_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] += val


def _sum_update_nan(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # branchless select-form twin of _sum_update - nans contribute nothing
    # and do not ready their group for a custom fill value; the guarded
    # equivalent turns the loop scalar and compiles ~3x slower
    # the tuple has to name the 32 bit numpy scalars explicitly - numba
    # maps python float/complex to the 64 bit variants only, and a float32
    # or complex64 nan taking the plain path would poison the accumulator
    valid = val == val
    if st.track:
        st.untouched[ri] = st.untouched[ri] & (val != val)
    if isinstance(val, (float, complex, np.float32, np.complex64)):
        st.out[ri] += val if valid else 0.0
    else:
        # integer and bool input have no nan - the plain path keeps wide
        # uint64 exact (a python literal 0 would unify uint64 with int64
        # into float64)
        st.out[ri] += val


def _sum_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


@njit_stable
def _combine_untouched(acc: ReductionState, other: ReductionState) -> None:
    """The groups that neither of the two chunks touched stay untouched.

    A group that one of the chunks put a value into is not untouched any
    more - the flag of the state of the first chunk says so.
    """
    if acc.track:
        untouched: np.ndarray = acc.untouched
        untouched &= other.untouched


def _sum_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one: the sums add up,
    and a group that neither of the chunks touched stays untouched."""
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
    track = fill_value != forced_fill
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
    """The states of two chunks of the data, folded into one: the products
    multiply, and a group that neither of the chunks touched stays untouched."""
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
    track = fill_value != forced_fill
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
    """The states of two chunks of the data, folded into one: the counts add
    up, and a group that neither of the chunks touched stays untouched."""
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
    track = fill_value != forced_fill
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
    """The states of two chunks of the data, folded into one: a group is true
    only if it is true in both of the chunks."""
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
    track = fill_value != forced_fill
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
    """The states of two chunks of the data, folded into one: a group is true
    if it is true in one of the chunks."""
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


# The state of the parallel path of the first and the last: the value of
# every group and the groups that have one.  The two operations cannot be
# reduced in chunks the way the sums are - the value of a group depends on
# the order of the values - so the chunks are folded in the order of their
# index: the first chunk that has a value for a group wins for the first,
# the last one for the last.
#
# The result is the one of the sequential path bit for bit: every chunk
# looks at its own values in the order they come, and the fold picks the
# value of the chunk that comes first (or last) in the input.
#
# The state of a chunk holds one array of the size of the groups and one
# flag per group - the same shape as the state of the sum, and the same
# weight for the cost model.
_MT_STATE_SIZE = 1


def _last_driver_factory(nan_variant: bool) -> Any:
    """The compiled driver of the last value of every group.

    The value of a group is the one that comes last in the input - the
    repeated indexing of the numpy implementation, in a loop of its own: the
    composed driver plus a separate njit update call compiles slower here
    (see the note at ``_first_driver_factory``).
    """

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
        # negative indices are flagged in the loop and raised after it: a
        # raise is a call, and a call inside the loop stops LLVM from sinking
        # the value loads into the short-circuit
        bad = False
        for i in range(group_idx.shape[0]):
            ri = group_idx[i]
            if ri < 0:
                bad = True
                continue
            val = a[i]
            if nan_variant and val != val:
                # a nan does not take the group over: the last valid value
                # is the one that is wanted
                continue
            out[ri] = val
        if bad:
            raise ValueError(NEGATIVE_INDICES)
        return out

    return njit_stable(driver, name=f"{'nan' if nan_variant else ''}last_driver")


def _last_mt_factory(nans: bool, driver: Callable[..., npt.NDArray[np.number]]) -> Any:
    """The parallel passes of the last: the data is split into chunks, every
    chunk looks for the last value of every group, and the chunks are folded
    in the order of their index - the last chunk that has a value for a group
    is the one that wins.

    The chunk looks for its value backwards: the last value of a group in a
    chunk is the first one the scan meets when it walks the chunk from its
    end.  A scan that walks it forwards would write the value of every value
    it meets, and a scatter write that the threads cannot keep apart is a
    write the threads do one after another - the chunk scan of the first is
    the other way round for the same reason.
    """

    @njit_stable(name=f"{'nan' if nans else ''}last_chunk_scan")
    def scan_chunk(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        out_c: npt.NDArray[np.number],
        seen_c: npt.NDArray[np.bool_],
    ) -> int:
        """The last value of every group within one chunk of the data.

        The fault of the indices travels as a number and not as a flag in an
        array: an exception cannot leave the parallel region, so the
        complaint is made after it.
        """
        fault = 0
        for i in range(group_idx.shape[0] - 1, -1, -1):
            ri = group_idx[i]
            if ri < 0:
                fault = 1
                continue
            if not seen_c[ri]:
                val = a[i]
                if nans and val != val:
                    # a nan does not take the group over: the last valid
                    # value is the one that is wanted
                    continue
                seen_c[ri] = True
                out_c[ri] = val
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
    ) -> npt.NDArray[np.number]:
        size = output_size(group_idx, size_arg)
        n_values = group_idx.shape[0]
        if not should_use_mt(size, n_threads, n_values, _MT_STATE_SIZE):
            # not worth it, or only one thread to do it with: the sequential
            # driver, with its result bit for bit
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
        n_chunks = chunk_count(n_threads, n_values)
        chunk = (n_values + n_chunks - 1) // n_chunks
        # The state of every chunk: the value of every group and the groups
        # that have one.  The states are the rows of two arrays and not a
        # list of arrays: a list of them inside the parallel region costs
        # more than the region saves (measured 160us against 96us per 200k
        # values on four threads).
        outs = np.full((n_chunks, size), fill_value, out_dtype)
        seen = np.zeros((n_chunks, size), np.bool_)
        # the fault of the indices of every chunk
        faults = np.zeros(n_chunks, np.int64)
        for c in nb.prange(n_chunks):
            start = c * chunk
            # np.minimum and not min: the module global 'min' is the min
            # kernel, not the builtin
            stop = np.minimum(start + chunk, n_values)
            faults[c] = scan_chunk(group_idx[start:stop], a[start:stop], outs[c], seen[c])
        # the fold in the order of the chunks: the last chunk that has a
        # value for a group wins
        acc: npt.NDArray[np.number] = outs[0]
        for c in range(1, n_chunks):
            acc[:] = np.where(seen[c], outs[c], acc)
        if faults.any():
            raise ValueError(NEGATIVE_INDICES)
        return acc

    return njit_stable(mt_passes, name=f"{'nan' if nans else ''}last_mt_passes", parallel=True)


last = build_custom_kernel(
    "last",
    driver_factory=_last_driver_factory,
    mt_driver_factory=_last_mt_factory,
    rule=rule_same,
    fill_default=np.nan,
    doc="Last value of every group",
)


def _first_driver_factory(nan_variant: bool) -> Any:
    # The loop body is inlined on purpose: with numba >= 0.67 the composed
    # driver plus a separate njit update call compiles ~12x slower here -
    # see the note in argreductions._argmax_driver_factory.
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
        pending = np.ones(size, np.bool_)
        # group_idx.shape[0] instead of len - the module global 'len' is the
        # len kernel, not the builtin
        # negative indices are flagged in the loop and raised after it: a
        # raise is a call, and a call inside the loop stops LLVM from sinking
        # the value loads into the short-circuit (measured ~0.1ms per 500k
        # values on nanfirst)
        bad = False
        for i in range(group_idx.shape[0]):
            ri = group_idx[i]
            if ri < 0:
                bad = True
                continue
            if nan_variant:
                # early-exit form, matching numbagg: after the first value of
                # a group the pending load short-circuits the rest, so the
                # value load only runs while groups are still pending (folded
                # away for the plain one)
                if pending[ri] and not np.isnan(a[i]):
                    pending[ri] = False
                    out[ri] = a[i]
            else:
                # the plain one takes the first value whatever it is
                if pending[ri]:
                    pending[ri] = False
                    out[ri] = a[i]
        if bad:
            raise ValueError(NEGATIVE_INDICES)
        return out

    return njit_stable(driver, name=f"{'nan' if nan_variant else ''}first_driver")


def _first_mt_factory(nans: bool, driver: Callable[..., npt.NDArray[np.number]]) -> Any:
    """The parallel passes of the first: the data is split into chunks, every
    chunk looks for the first value of every group, and the chunks are
    folded in the order of their index - the first chunk that has a value for
    a group is the one that wins.
    """

    @njit_stable(name=f"{'nan' if nans else ''}first_chunk_scan")
    def scan_chunk(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        out_c: npt.NDArray[np.number],
        seen_c: npt.NDArray[np.bool_],
    ) -> int:
        """The first value of every group within one chunk of the data.

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
            if not seen_c[ri]:
                val = a[i]
                if nans and val != val:
                    # a nan does not take the group over: the first valid
                    # value is the one that is wanted
                    continue
                seen_c[ri] = True
                out_c[ri] = val
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
    ) -> npt.NDArray[np.number]:
        size = output_size(group_idx, size_arg)
        n_values = group_idx.shape[0]
        if not should_use_mt(size, n_threads, n_values, _MT_STATE_SIZE):
            # not worth it, or only one thread to do it with: the sequential
            # driver, with its result bit for bit
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
        n_chunks = chunk_count(n_threads, n_values)
        chunk = (n_values + n_chunks - 1) // n_chunks
        # The state of every chunk: the value of every group and the groups
        # that have one.  The states are the rows of two arrays and not a
        # list of arrays: a list of them inside the parallel region costs
        # more than the region saves (measured 160us against 96us per 200k
        # values on four threads).
        outs = np.full((n_chunks, size), fill_value, out_dtype)
        seen = np.zeros((n_chunks, size), np.bool_)
        # the fault of the indices of every chunk
        faults = np.zeros(n_chunks, np.int64)
        for c in nb.prange(n_chunks):
            start = c * chunk
            # np.minimum and not min: the module global 'min' is the min
            # kernel, not the builtin
            stop = np.minimum(start + chunk, n_values)
            faults[c] = scan_chunk(group_idx[start:stop], a[start:stop], outs[c], seen[c])
        # the fold in the order of the chunks: the first chunk that has a
        # value for a group wins
        acc: npt.NDArray[np.number] = outs[0]
        acc_seen = seen[0]
        for c in range(1, n_chunks):
            take = np.logical_and(np.logical_not(acc_seen), seen[c])
            acc[:] = np.where(take, outs[c], acc)
            acc_seen[:] = np.logical_or(acc_seen, seen[c])
        if faults.any():
            raise ValueError(NEGATIVE_INDICES)
        return acc

    return njit_stable(mt_passes, name=f"{'nan' if nans else ''}first_mt_passes", parallel=True)


first = build_custom_kernel(
    "first",
    driver_factory=_first_driver_factory,
    mt_driver_factory=_first_mt_factory,
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
    track = fill_value != forced_fill
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.ones(size, np.bool_), untouched, track)


def _allnan_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] &= val != val


def _allnan_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _allnan_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one: a group holds
    only nans if it holds them in both of the chunks."""
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
    track = fill_value != forced_fill
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, np.bool_), untouched, track)


def _anynan_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    st.out[ri] |= val != val


def _anynan_untouched(st: ReductionState) -> npt.NDArray[np.bool_]:
    return st.untouched


def _anynan_combine(acc: ReductionState, other: ReductionState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one: a group holds a
    nan if it holds one in one of the chunks."""
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
    # under numba >= 0.66 when reached as a separate njit call.  Note: out[ri]
    # must be read into a local first - repeating the subscript expression
    # keeps the slow codegen (aliasing).  val != val has to be tested
    # explicitly - a nan compares neither smaller nor greater, in numpy's
    # order for complex values as well as for floats, so complex_less alone
    # would leave a group whose nan arrives after its first value at the
    # finite value, unlike np.max (and unlike the arg kernels, which report
    # such a group as invalid).  The nan variants are unaffected, they filter
    # the nans out before the kernel runs; they use their own select-form bodies instead (see
    # _max_update_nan).  The extra comparison is the
    # cheapest way to get there: np.maximum in the non-first branch (which
    # propagates nans on its own, but does not order complex values the way
    # numpy does) and the branchy form both measured ~45% slower, while for
    # integer dtypes LLVM folds the test away entirely.
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.out[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
    best = complex_less(cur, val)
    st.out[ri] = val if (first or val != val or best) else cur


def _max_update_nan(st: SelectionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form twin of _max_update - nans neither win nor clear the
    # pending flag
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.out[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
    best = complex_less(cur, val)
    st.out[ri] = val if (valid & (first | best)) else cur


def _max_combine(acc: SelectionState, other: SelectionState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one.

    The value of the other chunk is put into the state of the first one the
    way the kernel would have done it - the chunks are folded in the order of
    the data, so the result is the one of the sequential kernel, bit for bit.
    The body repeats the select of ``_max_update`` instead of calling it: the
    call would defeat the branchless form the update is written for.
    """
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
                if val != val or complex_less(out[g], val):
                    out[g] = val


def _max_combine_nan(acc: SelectionState, other: SelectionState, fill_value: float, extra: Any) -> None:
    """The states of two chunks of the data, folded into one - the twin of
    ``_max_combine`` for the nan variant, whose nans neither win nor clear
    the pending flag."""
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
    """The states of two chunks of the data, folded into one - the twin of
    ``_max_combine`` for the minimum."""
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
    """The states of two chunks of the data, folded into one - the twin of
    ``_max_combine_nan`` for the minimum."""
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
    # select-form on purpose, see _max_update (including why val != val is
    # part of the condition and what it costs)
    first = st.pending[ri]
    st.pending[ri] = False
    cur = st.out[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
    best = complex_less(val, cur)
    st.out[ri] = val if (first or val != val or best) else cur


def _min_update_nan(st: SelectionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    # select-form twin of _min_update, see _max_update_nan
    valid = val == val
    first = st.pending[ri]
    st.pending[ri] = first & (val != val)
    cur = st.out[ri]
    # the comparison is resolved into a local before the select
    # expression - reaching it as a call defeats the branchless
    # form (measured 2.5ms against 0.8ms per 500k values)
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
    track = fill_value != forced_fill
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return ReductionState(np.zeros(size, out_dtype), untouched, track)


def _sumofsquares_update(st: ReductionState, ri: int, i: int, val: float, fill_value: float, extra: Any) -> None:
    if st.track:
        st.untouched[ri] = False
    # the squared magnitude of val - complex values give a real, sqrt-free
    # result like np.var; bool has no .real in numba and keeps the plain
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
    """The states of two chunks of the data, folded into one: the sums of the
    squared magnitudes add up."""
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
