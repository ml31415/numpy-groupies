"""The temporal policy of the numpy based backends.

The functions a temporal dtype refuses (the ones numpy refuses itself),
the functions whose kernels take the values as they are, the NaT handling
of plain and nan reductions, and the fills of the answers.

``build_temporal_op`` composes the policy of this module with the kernels
of a backend: it takes the backend operation, its operation on the int64
ticks, and the kernel that takes temporal values as they are, and returns
the operation for temporal input. Nothing here reaches into a backend, so
both backends reach into this module.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.dtypes import TEMPORAL_KINDS
from numpy_groupies.fills import (
    DEFAULT_FILL_VALUE,
    resolve_fill_value,
)
from numpy_groupies.groups import exact_integer_group_mean
from numpy_groupies.registry import missing_impl_error
from numpy_groupies.typing import AFillValue, AResult, AValue, RavelOrder

# ---------------------------------------------------------------------------
# datetime64 and timedelta64 support
#
# NaT is the int64 tick ``-2**63``, so every tick-kernel result is a tick
# again and is viewed back into the dtype of the input.  The policy follows
# numpy: plain reductions propagate NaT, nan variants skip it; what numpy
# refuses, we refuse (numpy raises itself).  numpy does compute the quantile
# and percentile of a datetime64 (numpy/numpy#11620, closed with
# numpy/numpy#11627), and so do we - not the median, since it would need to
# add two middles.  The unit of the input is the unit of the answer; NaT is
# the fill value.

# functions whose result is viewed back into the dtype of the input

# the tick of NaT - the smallest tick of a group, skipped by the nan forms
NAT_TICK: int = -(2**63)

TEMPORAL_VALUE_FUNCS = frozenset(
    {
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "quantile",
        "percentile",
        "first",
        "last",
        "sort",
        "topk",
        "mode",
        "cumsum",
        "cummax",
        "cummin",
    }
)
# what numpy refuses for datetime64 - plain and nan variants alike (the
# quantile and percentile are not among them: numpy computes them)
TEMPORAL_DATETIME_REJECTED = frozenset(
    {
        "sum",
        "prod",
        "cumsum",
        "cumprod",
        "median",
        "var",
        "std",
        "skew",
        "kurt",
        "trapezoid",
        "sumofsquares",
    }
)
# what numpy refuses for timedelta64
TEMPORAL_TIMEDELTA_REJECTED = frozenset({"prod", "cumprod", "var", "std", "skew", "kurt", "trapezoid", "sumofsquares"})
# functions whose kernel takes temporal values as they are: compiled code
# orders the values and ``val != val`` detects NaT like a nan, so the kernel
# finds NaT, its position and empty groups on its own, in one pass.  The sum
# and cumsum of durations join in: duration addition is compiled and
# propagates NaT like numpy.  The mean stays out: numba's division of
# durations would truncate, where the package rounds half to even.
TEMPORAL_NATIVE_FUNCS = frozenset(
    {
        "min",
        "max",
        "nanmin",
        "nanmax",
        "first",
        "last",
        "nanfirst",
        "nanlast",
        "argmin",
        "argmax",
        "nanargmin",
        "nanargmax",
        "len",
        "nanlen",
        "allnan",
        "anynan",
        "array",
        "nunique",
        "nannunique",
        "mode",
        "sum",
        "nansum",
        "cumsum",
        "nancumsum",
        "cummax",
        "cummin",
    }
)
# plain variants poison their whole group with a single NaT.  The sum of
# durations joins in: the ticks would wrap around in the int64 accumulation.
TEMPORAL_NAT_POISON = frozenset({"sum", "mean", "max", "median", "quantile", "percentile"})
# value functions whose default fill is NaT; the zero-filling ones fill with
# 0 (the epoch, or no duration)
TEMPORAL_NAT_FILL = frozenset(
    {"min", "max", "mean", "median", "quantile", "percentile", "first", "last", "topk", "mode"}
)


def _temporal_fill_value(fill_value: Any, a_dtype: np.dtype, base: str) -> Any:
    """The fill value of a temporal result, as the value of its dtype.

    A nan and the missing default answer NaT (for the functions that fill
    with it) or the zero of the unit; a plain number is a tick; a temporal
    value passes through in the unit of the input.
    """
    if isinstance(fill_value, (np.datetime64, np.timedelta64)):
        return np.asarray(fill_value).astype(a_dtype)[()]
    if isinstance(fill_value, (float, np.floating)) and fill_value == fill_value:
        # a caller's own real number - no tick, no missing value
        return fill_value
    if (
        fill_value is None
        or fill_value is DEFAULT_FILL_VALUE
        or isinstance(fill_value, (bool, np.bool_, int, np.integer))
    ):
        tick = NAT_TICK if base in TEMPORAL_NAT_FILL else 0
        return np.asarray(tick, dtype=np.int64).view(a_dtype)[()]
    return fill_value


def _temporal_fill_tick(fill_value: Any, a_dtype: np.dtype, default_is_nat: bool) -> int:
    """The tick of the fill value: NaT where the default is nan, zero where
    it fills with zero, unless the caller asks for a date."""
    if fill_value is None or fill_value is DEFAULT_FILL_VALUE:
        return NAT_TICK if default_is_nat else 0
    if isinstance(fill_value, float) and fill_value != fill_value:
        return NAT_TICK if default_is_nat else 0
    if isinstance(fill_value, (int, np.integer)):
        return int(fill_value)
    # a datetime64/timedelta64, or anything numpy reads as one
    return int(np.asarray(fill_value, dtype=a_dtype).view(np.int64))


def _groups_holding(
    group_idx: npt.NDArray[np.integer], mask: npt.NDArray[np.bool_], size: int
) -> npt.NDArray[np.bool_]:
    """Which groups hold at least one true value of ``mask``, as a mask over
    the groups.

    Groups are counted rather than picked out: picking them out copies the
    labels (~2ms per million values vs ~0.2ms), and ``np.unique`` would sort
    on top of that.
    """
    positions = np.flatnonzero(mask)
    return np.bincount(group_idx[positions], minlength=size) > 0


def _first_true_position_per_group(
    group_idx: npt.NDArray[np.integer], mask: npt.NDArray[np.bool_], size: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.integer]]:
    """Which groups hold a true value of ``mask``, and the first masked
    position of each (-1 for none).

    numpy answers the position of the first NaT for both argmin and argmax
    of a group holding one; argmin on the ticks delivers it on its own
    (NaT is the smallest tick), argmax is corrected with this.
    """
    positions = np.flatnonzero(mask)
    labels = group_idx[positions]
    # a stable sort keeps the positions of each group ascending, so the
    # first of each run is the first NaT of the group
    order = np.argsort(labels, kind="stable")
    labels, positions = labels[order], positions[order]
    starts = np.concatenate(([True], labels[1:] != labels[:-1]))
    first = np.full(size, -1, np.intp)
    first[labels[starts]] = positions[starts]
    return first >= 0, first


def _per_group_running_mask(group_idx: npt.NDArray[np.integer], nat: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """The positions from the first NaT of their group on.

    Groups are interleaved in the input, so a mask over the whole input
    would poison following groups too; positions are ranked inside their
    group and masked from the rank of the first NaT on.
    """
    n = nat.size
    counts = np.bincount(group_idx)
    starts = np.zeros(counts.size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    # the rank of every position inside its group
    order = np.argsort(group_idx, kind="stable")
    ranks = np.empty(n, dtype=np.intp)
    ranks[order] = np.arange(n, dtype=np.intp) - starts[group_idx[order]]
    # the rank of the first NaT of each group
    unranked = np.iinfo(np.intp).max
    first_nat = np.full(counts.size, unranked, dtype=np.intp)
    np.minimum.at(first_nat, group_idx, np.where(nat, ranks, unranked))
    return cast("npt.NDArray[np.bool_]", ranks >= first_nat[group_idx])


# ---------------------------------------------------------------------------
# the way a backend calls its operations
#
# A backend states how its operations are called:
#
#     call(op, group_idx, values, size, fill_value, dtype, order, axis, **kwargs)
#
# There is no default: a backend that does not say how it calls its
# operations does not get any.


# ---------------------------------------------------------------------------
# the policy of one function
#
# The policy decisions are the same for every call of a function; they are
# taken when the operation is built, so the call itself holds no questions
# about the function it answers.


@dataclass(frozen=True)
class _Policy:
    """What the policy of one function says about temporal input."""

    funcname: str
    base: str  # the name of the plain form
    op: Callable[..., AResult]
    tick_op: Callable[..., AResult] | None  # the operation on the int64 ticks
    flag_op: Callable[..., AResult]
    call: Callable[..., AResult]  # how a backend operation is called
    rejects_datetime: bool
    rejects_timedelta: bool
    is_native: bool  # the kernel takes temporal values as they are
    drops_nat: bool
    swaps_nat: bool  # nan-argreductions stand NaT rows in as a sentinel
    tick_handles_nat: bool  # the tick op carries the NaT policy itself
    flag: str | None  # the question the function asks, if any
    patch: str | None  # what the NaT does to the answer of the kernel
    value_result: bool
    fill_is_nat: bool
    is_array: bool  # the function collects the values of its groups


def _missing_operation(funcname: str) -> Callable[..., AResult]:
    """The operation of a function the backend does not implement.

    The temporal policy refuses some functions before they are answered at
    all, so the refusal comes first and only then the missing-operation
    complaint.
    """

    def missing(*args: Any, **kwargs: Any) -> AResult:
        raise missing_impl_error(funcname)

    return missing


def build_temporal_op(
    funcname: str,
    op: Callable[..., AResult] | None,
    call: Callable[..., AResult],
    tick_op: Callable[..., AResult] | None = None,
    native: bool = False,
    resolve: Callable[[str], Callable[..., AResult] | None] | None = None,
    tick_handles_nat: bool = False,
) -> Callable[..., tuple[AResult, AFillValue]]:
    """The operation of one function for temporal input.

    ``op`` answers the numeric values; ``tick_op`` is the backend operation
    on the int64 ticks, for the functions whose numeric kernel is not exact
    on the ticks (mean, median, quantile).  ``native`` says the kernel takes
    temporal values as they are (the numba compiled kernels order them and
    find their NaT on their own); a backend without such a kernel says no
    and everything is answered on the ticks.

    ``call`` is how the backend operations are called - each backend states
    its own convention.  ``resolve`` is how the backend resolves an operation
    name: the missing-value flags are the plain any and all of the NaT mask,
    resolved here.  ``tick_handles_nat`` says the tick op carries the NaT
    policy of the function itself (plain poisons the group, nan skips it),
    so no NaT row is dropped and the kernel answer is not patched.
    """
    base = funcname.removeprefix("nan")
    is_nan = funcname.startswith("nan")
    if op is None:
        # the policy still refuses what numpy refuses before any missing-op
        # complaint
        op = _missing_operation(funcname)
    # only the functions of TEMPORAL_NATIVE_FUNCS can take values as they are
    is_native = native and base in TEMPORAL_NATIVE_FUNCS
    # nan-argreductions keep every row - the positions they name are the ones
    # of the input - so NaT rows stand in as the extreme tick the reduction
    # skips
    swaps_nat = is_nan and base in ("argmin", "argmax")
    # nan variants skip NaT like nan; plain ones propagate it, and the
    # cumulative ones carry it through their group like numpy.  The nan mean
    # does not filter: the tick mean leaves the NaT of a group out on its
    # own, and filtering would cost a copy of the data.  The nan median
    # filters in its own backend (numba kernel leaves the NaT tick out, the
    # exact numpy one drops rows).  Topk and mode have no nan variant and
    # leave NaT out as well: a NaT is in no selection of theirs.  When the
    # tick op carries the NaT policy itself (the numba kernel of mean,
    # median, quantile), the engine neither drops rows nor patches.
    handles_nat = tick_handles_nat and tick_op is not None
    drops_nat = (
        False
        if handles_nat
        else (
            (is_nan and base not in ("cumsum", "mean") and not (base == "median" and tick_op is not None))
            # the topk and the mode have no nan variant, and the missing value
            # is in no selection of theirs - the rows of the NaT are dropped
            # where the engine answers on the ticks
            or base in ("topk", "mode")
        )
    )
    # what the NaT does to the answer: the plain argmin finds the NaT on its
    # own (its tick is the smallest), the plain argmax never picks it and
    # needs the position of the first NaT of the group.  A single NaT
    # poisons the sum, the maximum and the middles of its group.  The sort
    # puts NaT to the end although it compares as the smallest value
    # (numpy/numpy#12629), and the accumulation carries it through to the
    # end of its group, nancumsum included.
    if base in ("argmin", "argmax"):
        patch = "argmax" if base == "argmax" else None
    elif base in TEMPORAL_NAT_POISON and not is_nan and not handles_nat:
        patch = "poison"
    elif base == "sort":
        patch = "sort"
    elif base in ("cumsum", "cummax", "cummin") and not is_native:
        patch = "running"  # the native backend carries NaT on its own
    else:
        patch = None
    # the missing-value flags are the plain any and all of the NaT mask
    flag = base if base in ("all", "any", "allnan", "anynan") else None
    flag_op = _missing_operation(funcname)
    if flag in ("allnan", "anynan"):
        if resolve is None:
            raise missing_impl_error(funcname)
        resolved_flag_op = resolve("any" if flag == "anynan" else "all")
        if resolved_flag_op is None:
            raise missing_impl_error(funcname)
        flag_op = resolved_flag_op
    policy = _Policy(
        funcname=funcname,
        base=base,
        op=op,
        tick_op=tick_op,
        flag_op=flag_op,
        call=call,
        rejects_datetime=base in TEMPORAL_DATETIME_REJECTED,
        rejects_timedelta=base in TEMPORAL_TIMEDELTA_REJECTED,
        is_native=is_native,
        drops_nat=drops_nat,
        swaps_nat=swaps_nat,
        tick_handles_nat=handles_nat,
        flag=flag,
        patch=patch,
        value_result=base in TEMPORAL_VALUE_FUNCS,
        fill_is_nat=base in TEMPORAL_NAT_FILL,
        is_array=base == "array",
    )

    def temporal_op(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        dtype: npt.DTypeLike | None = None,
        order: RavelOrder = "C",
        axis: int | None = None,
        **kwargs: Any,
    ) -> tuple[AResult, AFillValue]:
        return _temporal_aggregation(policy, group_idx, a, size, fill_value, dtype, kwargs, order, axis)

    temporal_op.__name__ = f"temporal_{funcname}"
    temporal_op.__qualname__ = f"temporal_{funcname}"
    temporal_op.__doc__ = (
        f"The {funcname} of every group of temporal values - the answer of the ticks, in the unit of the input."
    )
    return temporal_op


# ---------------------------------------------------------------------------
# the engine of a temporal aggregation
#
# One stage per concern: the refusal, the scalar, the native kernel, the
# NaT rows, the questions, the patches, the tick answer, the kernel answer,
# and the view back into the unit of the input.


@dataclass
class _Patches:
    """The patches the NaT of the values puts on the kernel answers, gathered
    so that answering and repairing stay apart."""

    nat_groups: npt.NDArray[np.bool_] | None = None  # groups a NaT poisons
    nat_positions: npt.NDArray[np.integer] | None = None  # first NaT per group
    running_nat: npt.NDArray[np.bool_] | None = None  # from first NaT on
    sort_nat_restore: bool = False  # largest tick stood in for NaT while sorting


@dataclass
class _Context:
    """The state of one temporal call."""

    policy: _Policy
    group_idx: npt.NDArray[np.integer]
    values: npt.NDArray[Any]
    a_dtype: np.dtype
    size: int
    fill_value: AFillValue
    user_dtype: np.dtype | None
    kwargs: dict[str, Any]
    order: RavelOrder
    axis: int | None
    ticks: npt.NDArray[np.int64] = field(init=False)  # the int64 the kernels work on
    empty_groups: npt.NDArray[np.bool_] | None = None  # groups holding only NaT
    patches: _Patches = field(default_factory=_Patches)

    def __post_init__(self) -> None:
        self.ticks = self.values.view(np.int64)

    def call(
        self,
        op: Callable[..., AResult],
        values: AValue,
        fill_value: AFillValue,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Any,
    ) -> AResult:
        """The call of an operation of the backend."""
        return self.policy.call(
            op,
            self.group_idx,
            values,
            self.size,
            fill_value,
            dtype,
            self.order,
            self.axis,
            **kwargs,
        )


def _reject_unsupported(ctx: _Context) -> None:
    """What numpy refuses for temporal input, we refuse as well."""
    policy = ctx.policy
    is_datetime = ctx.a_dtype.kind == "M"
    if (is_datetime and policy.rejects_datetime) or (not is_datetime and policy.rejects_timedelta):
        raise TypeError(
            f"aggregate {policy.funcname!r} is not supported for {ctx.a_dtype.name} input - "
            f"numpy refuses {policy.base} of {'datetime64' if is_datetime else 'timedelta64'} values as well"
        )


def _native_fill_value(ctx: _Context) -> Any:
    """The fill value of a native kernel: a number/bool/object for positions,
    counts, flags and collected values, NaT-or-zero of the unit otherwise."""
    policy = ctx.policy
    base = policy.base
    if base in ("len", "nanlen", "nunique", "nannunique", "argmin", "argmax", "nanargmin", "nanargmax"):
        return resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(np.int64))
    if base in ("allnan", "anynan"):
        return resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(np.bool_))
    if base == "array":
        return resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(object))
    # a value of the input: fill with the NaT or zero of the unit
    return _temporal_fill_value(ctx.fill_value, ctx.a_dtype, base)


def _swap_nat_rows(ctx: _Context) -> None:
    """Swap NaT rows of the nan-argreductions for the extreme tick the
    reduction skips (they keep every row).  Groups holding nothing but NaT
    keep the fill value like their float counterpart."""
    nat = np.isnat(ctx.values)
    skip = np.iinfo(np.int64).max if ctx.policy.base == "argmin" else NAT_TICK + 1
    ctx.values = np.where(nat, np.int64(skip), ctx.values.view(np.int64)).view(ctx.a_dtype)
    # groups holding nothing but NaT keep the fill value, like a float group
    # holding nothing but nans
    only_nat = _groups_holding(ctx.group_idx, nat, ctx.size)
    counts = np.bincount(ctx.group_idx, minlength=ctx.size)
    only_nat &= counts == np.bincount(ctx.group_idx, weights=nat, minlength=ctx.size)
    ctx.empty_groups = only_nat


def _drop_nat_rows(ctx: _Context) -> None:
    """Drop the NaT rows the nan-variants and topk leave out.  Done here so
    the kernels - public entry points handed a NaT of their own - need no
    wrapper preparation."""
    nat = np.isnat(ctx.values)
    if not nat.any():
        return
    good = ~nat
    ctx.values = ctx.values[good]
    ctx.group_idx = ctx.group_idx[good]


def _flag_result(ctx: _Context) -> tuple[AResult, AFillValue]:
    """The answer of the question functions: the truth of a temporal value is
    its tick being non-zero and the value not NaT; the missing-value flags
    are the plain any and all of the NaT mask."""
    policy = ctx.policy
    if policy.flag in ("all", "any"):
        truth: npt.NDArray[np.bool_] = ctx.values.view(np.int64) != 0
        truth &= ~np.isnat(ctx.values)
        ret = ctx.call(
            policy.op,
            truth,
            resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(np.bool_)),
            **ctx.kwargs,
        )
        return ret, ctx.fill_value
    # the missing-value flags are the plain any and all of the NaT mask
    ret = ctx.call(
        policy.flag_op,
        np.isnat(ctx.values),
        resolve_fill_value("all", ctx.fill_value, np.dtype(np.bool_)),
        **ctx.kwargs,
    )
    return ret, ctx.fill_value


def _nan_patches(ctx: _Context) -> None:
    """Gather the patches the NaT of the values puts on the kernel answers."""
    policy = ctx.policy
    patches = ctx.patches
    if policy.patch not in ("argmax", "poison", "sort", "running"):
        return
    nat = np.isnat(ctx.values)
    if not nat.any():
        return
    if policy.patch == "argmax":
        patches.nat_groups, patches.nat_positions = _first_true_position_per_group(ctx.group_idx, nat, ctx.size)
        return
    if policy.patch == "poison":
        patches.nat_groups = _groups_holding(ctx.group_idx, nat, ctx.size)
        return
    if policy.patch == "sort":
        # the largest tick stands in for the NaT through the sort; the
        # answer turns it back into NaT
        ctx.ticks = np.where(nat, np.iinfo(np.int64).max, ctx.ticks)
        patches.sort_nat_restore = True
        return
    patches.running_nat = _per_group_running_mask(ctx.group_idx, nat)


def _mean_result(ctx: _Context) -> tuple[npt.NDArray[Any] | None, bool, bool]:
    """The mean - as a float, or as a tick.

    The tick mean of the backend measures the deviations from the first
    value of each group, so the sum of ticks stays exact and the answer is
    a tick again; the plain kernel poisons the group of a NaT on its own,
    the nan one leaves NaT out.  A backend without a kernel gets the exact
    mean of this module (python-integer sums).  The float answer is final:
    it is not viewed back into the dtype of the input.
    """
    policy = ctx.policy
    fill_tick = _temporal_fill_tick(ctx.fill_value, ctx.a_dtype, True)
    if ctx.user_dtype is not None and ctx.user_dtype.kind == "f":
        mean_idx, mean_ticks = ctx.group_idx, ctx.ticks
        nat_groups_for_float = None
        if policy.funcname.startswith("nan"):
            # the NaT rows have no tick to average
            good = ctx.ticks != NAT_TICK
            mean_idx, mean_ticks = ctx.group_idx[good], ctx.ticks[good]
        else:
            # a single NaT poisons its group
            nat = np.isnat(ctx.values)
            if nat.any():
                nat_groups_for_float = _groups_holding(ctx.group_idx, nat, ctx.size)
        computed = cast(
            "npt.NDArray[Any]",
            exact_integer_group_mean(mean_idx, mean_ticks, ctx.size, fill_tick, ctx.user_dtype),
        )
        if nat_groups_for_float is not None:
            computed = cast("npt.NDArray[Any]", np.asarray(computed))
            computed[nat_groups_for_float] = np.nan
        return computed, True, False

    if policy.tick_op is None:
        raise missing_impl_error(policy.funcname)
    try:
        computed = cast(
            "npt.NDArray[Any]",
            ctx.call(policy.tick_op, ctx.ticks, fill_tick, dtype=np.dtype(np.int64), **ctx.kwargs),
        )
    except OverflowError:
        # The values of a group span one end of the int64 range to the other:
        # the sum of deviations leaves the range, and the kernel refuses
        # rather than wrapping around silently; the limb sums hold it exact.
        if policy.funcname.startswith("nan"):
            # the nan kernel never sums the NaT ticks, so the limb sums take
            # the values it would have averaged
            good = ctx.ticks != NAT_TICK
            computed = cast(
                "npt.NDArray[Any]",
                exact_integer_group_mean(ctx.group_idx[good], ctx.ticks[good], ctx.size, fill_tick, np.dtype(np.int64)),
            )
        else:
            # a NaT may be what broke the span of the deviations - the
            # poison comes back through the patch
            nat = np.isnat(ctx.values)
            if nat.any():
                ctx.patches.nat_groups = _groups_holding(ctx.group_idx, nat, ctx.size)
            computed = cast(
                "npt.NDArray[Any]",
                exact_integer_group_mean(ctx.group_idx, ctx.ticks, ctx.size, fill_tick, np.dtype(np.int64)),
            )
    return computed, False, True


def _median_result(ctx: _Context) -> tuple[npt.NDArray[Any] | None, bool, bool]:
    """The middle of the ticks of every group: the backend kernel in compiled
    code, or the exact median of this module (plain answers NaT-poisoned
    groups through the patch, nan leaves the NaT tick out on its own)."""
    policy = ctx.policy
    if policy.tick_op is None:
        raise missing_impl_error(policy.funcname)
    fill_tick = _temporal_fill_tick(ctx.fill_value, ctx.a_dtype, True)
    computed = cast(
        "npt.NDArray[Any]",
        ctx.call(policy.tick_op, ctx.ticks, fill_tick, dtype=np.dtype(np.int64), **ctx.kwargs),
    )
    return computed, False, True


def _quantile_result(ctx: _Context) -> tuple[npt.NDArray[Any] | None, bool, bool]:
    """The quantile of the ticks of every group, interpolating with the
    method the caller names (as in ``np.quantile``)."""
    policy = ctx.policy
    if policy.tick_op is None:
        raise missing_impl_error(policy.funcname)
    fill_tick = _temporal_fill_tick(ctx.fill_value, ctx.a_dtype, True)
    fraction = cast("float", ctx.kwargs.get("q", 50.0 if policy.base == "percentile" else 0.5))
    if policy.base == "percentile":
        fraction = fraction / 100.0
    method = cast("str", ctx.kwargs.get("method", "linear"))
    computed = ctx.call(
        policy.tick_op,
        ctx.ticks,
        fill_tick,
        dtype=np.dtype(np.int64),
        q=fraction,
        method=method,
    )
    return cast("npt.NDArray[Any]", computed), False, True


def _ticks_result(ctx: _Context) -> tuple[npt.NDArray[Any] | None, bool, bool]:
    """The wrapper-computed answer - mean, median, quantiles - where the
    ticks make the arithmetic exact; ``None`` for every other function.
    Second value: whether the answer is final (the float mean is).  Third:
    whether it is a temporal value whose ticks have to be viewed back.
    """
    policy = ctx.policy
    if policy.base == "mean":
        return _mean_result(ctx)
    if policy.base == "median":
        return _median_result(ctx)
    if policy.base in ("quantile", "percentile"):
        return _quantile_result(ctx)
    # every other function is answered by the kernel of its operation
    return None, False, policy.value_result


def _array_result(ctx: _Context) -> tuple[AResult, AFillValue]:
    """The values of every group, back in the dtype they came in."""
    policy = ctx.policy
    empty = resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(object))
    ret = ctx.call(policy.op, ctx.ticks, empty, **ctx.kwargs)
    groups = ret if isinstance(ret, np.ndarray) else np.asarray(ret, dtype=object)
    return (
        cast("AResult", np.array([np.asarray(one).view(ctx.a_dtype) for one in groups], dtype=object)),
        ctx.fill_value,
    )


def _kernel_result(ctx: _Context, computed: npt.NDArray[Any] | None, value_result: bool) -> AResult:
    """The kernel answer.  Value results are answered in ticks with a tick
    fill; positions, counts and flags keep their own dtypes and fill values."""
    policy = ctx.policy
    if computed is not None:
        return computed
    if value_result:
        fill_tick = _temporal_fill_tick(ctx.fill_value, ctx.a_dtype, policy.fill_is_nat)
        return ctx.call(policy.op, ctx.ticks, fill_tick, dtype=np.dtype(np.int64), **ctx.kwargs)
    out_dtype = ctx.user_dtype if ctx.user_dtype is not None else np.dtype(np.intp)
    if policy.flag in ("allnan", "anynan"):
        out_dtype = np.dtype(np.bool_)
    return ctx.call(
        policy.op,
        ctx.ticks,
        resolve_fill_value(policy.funcname, ctx.fill_value, out_dtype),
        dtype=ctx.user_dtype,
        **ctx.kwargs,
    )


def _patch_result(ctx: _Context, ret: AResult) -> AResult:
    """Repair the kernel answer by the NaT patches: running NaT, the sort's
    sentinel back to NaT, poisoned groups to NaT, argmax to the first NaT
    position, empty groups to their fill value."""
    patches = ctx.patches
    if patches.running_nat is not None:
        ret = np.asarray(ret)
        ret[patches.running_nat] = NAT_TICK
    if patches.sort_nat_restore:
        # the largest tick stood in for NaT while sorting
        ret = np.asarray(ret)
        ret[ret == np.iinfo(np.int64).max] = NAT_TICK
    if patches.nat_groups is not None:
        ret = np.asarray(ret)
        if patches.nat_positions is not None:
            ret[patches.nat_groups] = patches.nat_positions[patches.nat_groups]
        else:
            ret[patches.nat_groups] = NAT_TICK
    if ctx.empty_groups is not None:
        # groups holding nothing but the missing value keep their fill value
        ret = np.asarray(ret)
        empty_fill = resolve_fill_value(ctx.policy.funcname, ctx.fill_value, np.dtype(np.int64))
        ret[ctx.empty_groups] = cast("int", empty_fill if empty_fill is not None else -1)
    return ret


def _view_result(ret: AResult, a_dtype: np.dtype, user_dtype: np.dtype | None) -> AResult:
    """Turn the kernel answer back into a temporal one: the unit of the
    input is the unit of the answer - unless the caller asks for another
    unit (then the ticks are rescaled) or for the ticks themselves."""
    ret = np.asarray(ret)
    if user_dtype is not None:
        if user_dtype == np.dtype(np.int64):
            return ret  # the ticks the caller asked for
        if user_dtype.kind in TEMPORAL_KINDS:
            # rescale from the unit of the input to the one asked for
            return ret.view(a_dtype).astype(user_dtype)
        return ret.view(np.int64).astype(user_dtype)
    return ret.view(a_dtype)


def _temporal_aggregation(
    policy: _Policy,
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    size: int,
    fill_value: AFillValue,
    dtype: npt.DTypeLike | None,
    kwargs: dict[str, Any],
    order: RavelOrder = "C",
    axis: int | None = None,
) -> tuple[AResult, AFillValue]:
    """The answer of a temporal aggregation, and the fill it used.

    What numpy refuses, we refuse; plain reductions hand NaT on to their
    group and nan variants leave it out; mean, median and quantiles compute
    on the int64 ticks, where the arithmetic is exact; the answer is
    temporal, in the unit of the input.  The policy is resolved when the
    operation is built.
    """
    values = np.asarray(a)
    ctx = _Context(
        policy=policy,
        group_idx=group_idx,
        values=values,
        a_dtype=values.dtype,
        size=size,
        fill_value=fill_value,
        user_dtype=np.dtype(dtype) if dtype is not None else None,
        kwargs=kwargs,
        order=order,
        axis=axis,
    )
    _reject_unsupported(ctx)

    if not isinstance(a, np.ndarray):
        # a temporal scalar comes through as its tick - sum and len answer
        # in int64, like a plain integer scalar
        ret = ctx.call(policy.op, cast("AValue", np.int64(values.view(np.int64))), fill_value, **kwargs)
        return ret, fill_value

    plain_fill = isinstance(fill_value, (bool, np.bool_, int, np.integer))
    if policy.is_native and ctx.user_dtype is None and not plain_fill:
        # the kernel takes temporal values as they are: it finds NaT, its
        # position and the empty groups on its own, in one pass
        fill = _native_fill_value(ctx)
        ret = ctx.call(policy.op, ctx.values, fill, **kwargs)
        return ret, fill_value

    if policy.swaps_nat:
        _swap_nat_rows(ctx)
    elif policy.drops_nat:
        _drop_nat_rows(ctx)
    # the ticks as they are now - the nan variants dropped their NaT rows
    ctx.ticks = ctx.values.view(np.int64)

    if policy.flag is not None:
        return _flag_result(ctx)

    _nan_patches(ctx)

    computed, computed_final, value_result = _ticks_result(ctx)

    if policy.is_array:
        return _array_result(ctx)

    ret = _kernel_result(ctx, computed, value_result)

    if computed is not None and computed_final:
        # the float mean is a float and not a tick - the final answer
        return ret, fill_value

    ret = _patch_result(ctx, ret)

    if value_result:
        return _view_result(ret, ctx.a_dtype, ctx.user_dtype), fill_value
    return ret, fill_value
