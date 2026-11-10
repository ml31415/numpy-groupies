"""The temporal policy of the numpy based backends.

Everything both numpy based backends need to know about datetime64 and
timedelta64 input lives here: the functions a dtype refuses (the ones numpy
refuses itself), the functions whose compiled kernels take the values as
they are, what a plain reduction does with the NaT of a group and what the
nan variants leave out, and the fills of the answers.

The temporals are built like the other operations of a backend: one
operation per function name, composed of the policy of this module and of
the kernels of the backend, and put into the dispatch of the backend beside
the numeric ones.  ``build_temporal_op`` is the builder - it takes the
operation of the backend, the operation of the backend on the int64 ticks
and the kernel of the backend that takes the temporal values as they are,
resolves the policy of the function name once, and hands out the operation
that answers the function for temporal input.

The dtype facts stay in ``numpy_groupies``: the kinds of the temporal dtypes.
The order statistics of the numpy backend live in
``aggregate_numpy.specials`` - the exact median of the ticks - and the
numba kernels in ``aggregate_numba``: nothing in this module reaches into
a backend, so both of them reach into this module.
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
# The tables name the functions of the package and what their temporal
# answer is made of.  NaT is the int64 tick ``-2**63``, so every value a
# kernel computes on the ticks of the values is a tick again - the
# wrapper below turns the tick results back into temporal ones.
#
# The policy follows numpy:
# - plain reductions propagate NaT (a group holding one answers NaT), the
#   nan-variants skip it,
# - what numpy refuses, we refuse: datetime64 has no sum, prod, cumsum,
#   cumprod, median, var, std, skew, kurt or trapezoid, and
#   timedelta64 has no prod, cumprod, var, std, skew, kurt or trapezoid - in
#   both cases numpy itself raises.  The quantile and the percentile of a
#   datetime64 numpy does compute (numpy/numpy#11620, closed with
#   numpy/numpy#11627), and so do we - the median it cannot, because it
#   adds two middles,
# - the unit of the input is the unit of the answer, and NaT is the fill
#   value.

# the functions whose result holds ticks of the input again - their answer is
# viewed back into the dtype of the input.  datetime64 has no sum and no
# median, but both raise before anything is viewed back.
# the tick of the NaT of a temporal value - the value every tick kernel
# finds as the smallest tick of its group, and the one the nan forms leave
# out and the plain ones take their group over with

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
# the functions numpy refuses for datetime64 - the plain and the nan variant
# share their fate.  The quantile and the percentile are not among them:
# numpy computes them of a datetime64 (numpy/numpy#11620, closed with
# numpy/numpy#11627), while the median it cannot - it adds two middles,
# and numpy has no addition of datetime64 values.
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
# and the ones numpy refuses for timedelta64 - sum, cumsum, mean, median and
# the quantiles work there like they do on integers
TEMPORAL_TIMEDELTA_REJECTED = frozenset({"prod", "cumprod", "var", "std", "skew", "kurt", "trapezoid", "sumofsquares"})
# the functions whose kernel takes the temporal values as they are: the
# comparisons of compiled code order datetime64 and timedelta64 values, and
# ``val != val`` detects NaT like a nan - so the kernel finds the NaT of a
# group, its position and the empty groups on its own, in one pass.  The sum
# and the cumulative sums of the durations belong to them: the addition of a
# duration is compiled, and it propagates the NaT like numpy does.  The mean
# is the one value function that stays behind: numba's division of durations
# would truncate it, where the package rounds half to even.
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
# plain variants of these poison their whole group with a single NaT - the
# group answers NaT rather than the value of its ticks.  The sum of the
# durations joins them: numpy adds NaT into a temporal sum and the answer is
# NaT, while the ticks would wrap around in the int64 accumulation.
TEMPORAL_NAT_POISON = frozenset({"sum", "mean", "max", "median", "quantile", "percentile"})
# the value functions whose default fill is nan - the temporal answer of an
# empty group is NaT for them, and 0 (the epoch, or no duration) for the
# functions that fill with zero, like the sum of the durations
TEMPORAL_NAT_FILL = frozenset(
    {"min", "max", "mean", "median", "quantile", "percentile", "first", "last", "topk", "mode"}
)


def _temporal_fill_value(fill_value: Any, a_dtype: np.dtype, base: str) -> Any:
    """The fill value of a temporal result, as the value of its dtype.

    A nan (the default of the averaging functions, or the caller's own) and
    the missing default answer NaT for the functions that fill with it and
    the zero of the unit for the ones that fill with zero; a plain number is
    a tick; a temporal value passes through in the unit of the input.
    """
    if isinstance(fill_value, (np.datetime64, np.timedelta64)):
        return np.asarray(fill_value).astype(a_dtype)[()]
    if isinstance(fill_value, (float, np.floating)) and fill_value == fill_value:
        # a real number of the caller's own - no tick, and no missing value
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
    """The tick of the fill value - NaT where the default of the function is
    nan, zero where it fills with zero, unless the caller asks for a date."""
    if fill_value is None or fill_value is DEFAULT_FILL_VALUE:
        return NAT_TICK if default_is_nat else 0
    if isinstance(fill_value, float) and fill_value != fill_value:
        return NAT_TICK if default_is_nat else 0
    if isinstance(fill_value, (int, np.integer)):
        return int(fill_value)
    # a datetime64/timedelta64, or anything numpy reads as one: its tick in
    # the unit of the input
    return int(np.asarray(fill_value, dtype=a_dtype).view(np.int64))


def _groups_holding(
    group_idx: npt.NDArray[np.integer], mask: npt.NDArray[np.bool_], size: int
) -> npt.NDArray[np.bool_]:
    """Which groups hold at least one true value of ``mask``, as a mask over
    the groups - it indexes the answer of an aggregation just like the
    labels of the groups would.

    The groups are counted rather than picked out: picking them out indexes
    the labels with the mask, and that is a copy of the labels - measured
    2ms of a million values, where the counting over the masked labels takes
    0.2ms.  ``np.unique`` would sort them on top of that.
    """
    positions = np.flatnonzero(mask)
    return np.bincount(group_idx[positions], minlength=size) > 0


def _first_true_position_per_group(
    group_idx: npt.NDArray[np.integer], mask: npt.NDArray[np.bool_], size: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.integer]]:
    """Which groups hold a true value of ``mask``, and the first masked
    position of every one of them (-1 for the groups that hold none).

    numpy answers the position of the first NaT for both argmin and argmax of
    a group that holds one - the argreductions on the ticks deliver the first
    NaT for argmin on their own (NaT is the smallest tick), argmax is
    corrected with this.
    """
    positions = np.flatnonzero(mask)
    labels = group_idx[positions]
    # a stable sort keeps the positions of every group in ascending order, so
    # the first of each run is the first NaT of the group
    order = np.argsort(labels, kind="stable")
    labels, positions = labels[order], positions[order]
    starts = np.concatenate(([True], labels[1:] != labels[:-1]))
    first = np.full(size, -1, np.intp)
    first[labels[starts]] = positions[starts]
    return first >= 0, first


def _per_group_running_mask(group_idx: npt.NDArray[np.integer], nat: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """The positions from the first NaT of their group on.

    The accumulation of a group carries a NaT through to its end (numpy does
    the same), but the groups are interleaved in the input - a mask that
    accumulates over the whole input would poison the groups that follow a
    group with a NaT as well.  The positions are therefore ranked inside
    their group, and every group is masked from the rank of its first NaT on.
    """
    n = nat.size
    counts = np.bincount(group_idx)
    starts = np.zeros(counts.size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    # the rank of every position inside its group, in the order of the input
    order = np.argsort(group_idx, kind="stable")
    ranks = np.empty(n, dtype=np.intp)
    ranks[order] = np.arange(n, dtype=np.intp) - starts[group_idx[order]]
    # the rank of the first NaT of every group
    unranked = np.iinfo(np.intp).max
    first_nat = np.full(counts.size, unranked, dtype=np.intp)
    np.minimum.at(first_nat, group_idx, np.where(nat, ranks, unranked))
    return cast("npt.NDArray[np.bool_]", ranks >= first_nat[group_idx])


# ---------------------------------------------------------------------------
# the way a backend calls its operations
#
# The operations of a backend have the signature of that backend, and the
# builder of a temporal operation is told how to call them: the two
# backends have two conventions, and each of them knows its own.  The
# convention is a function of the signature of the operation of the backend
#
#     call(op, group_idx, values, size, fill_value, dtype, order, axis, **kwargs)
#
# and the builder hands the arguments of the call to it.  There is no
# default: a backend that does not say how it calls its operations does not
# get any - the conventions live in the backends.


# ---------------------------------------------------------------------------
# the policy of one function
#
# The decisions the policy makes are the same for every call of a function -
# what numpy refuses for it, whether its kernel takes the temporal values as
# they are, what the NaT does to its answer and whether the answer is a
# temporal value at all.  They are taken when the operation is built, so
# that the call itself holds no questions about the function it answers.


@dataclass(frozen=True)
class _Policy:
    """What the policy of one function says about temporal input."""

    # the name of the function and the name of its plain form
    funcname: str
    base: str
    # the operation of the backend, the one on the int64 ticks of the values
    # and the one of the flags - the plain any and all the missing-value
    # flags are
    op: Callable[..., AResult]
    tick_op: Callable[..., AResult] | None
    flag_op: Callable[..., AResult]
    # the way the operation of the backend is called
    call: Callable[..., AResult]
    # what numpy refuses for a datetime64 and for a timedelta64
    rejects_datetime: bool
    rejects_timedelta: bool
    # whether the kernel of the backend takes the temporal values as they are
    is_native: bool
    # whether the rows of the NaT are left out, and whether the rows of the
    # nan-variants of the argreductions stand in for a sentinel
    drops_nat: bool
    swaps_nat: bool
    # whether the operation of the backend on the ticks carries the NaT
    # policy of the function itself
    tick_handles_nat: bool
    # the question the function asks, if it asks one
    flag: str | None
    # what the NaT does to the answer of the kernel
    patch: str | None
    # whether the answer is a temporal value, and whether the function fills
    # with the NaT of its unit
    value_result: bool
    fill_is_nat: bool
    # whether the function collects the values of its groups
    is_array: bool


def _missing_operation(funcname: str) -> Callable[..., AResult]:
    """The operation of a function the backend does not implement.

    The policy of the temporal input refuses some of the functions before
    they are answered at all - numpy refuses the sum of a datetime64 as
    well - so the refusal comes first, and only then the complaint that the
    backend has no operation for the function.
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

    ``op`` is the operation of the backend for the function - the operation
    that answers the numeric values.  ``tick_op`` is the operation of the
    backend on the int64 ticks of the values: the functions whose numeric
    kernel is not exact on the ticks (the mean, the median and the
    quantile) have one of their own, and the others do not - their numeric
    kernel answers the ticks as they are.

    ``native`` says that the kernel of the backend takes the temporal values
    as they are - the compiled kernels of the numba backend order them, find
    their NaT and add the durations on their own, in one pass over the data.
    A backend without such a kernel (the numpy one works on the ticks) says
    no, and every function of it is answered on the ticks.

    ``call`` is the way the operations of the backend are called - the two
    backends have two conventions, and each of them knows its own: the
    policy of this module does not have to know which one it is serving.
    Every backend states its own.

    ``resolve`` is the way the backend resolves the name of an operation -
    the missing-value flags are the plain any and all of the NaT mask, and
    their operation is resolved here, when the operation is built.

    ``tick_handles_nat`` says that the operation of the backend on the
    ticks carries the NaT policy of the function itself: the plain form
    poisons the group of a NaT and the nan form leaves the NaT out.  The
    engine then hands the rows over as they are - no row of the NaT is
    dropped, and the answer of the kernel is not patched afterwards.

    The operation that comes out of this builder has the signature of the
    other operations of the backend: the values, the size, the fill value
    and the dtype - and the arguments of the function itself behind them.
    """
    base = funcname.removeprefix("nan")
    is_nan = funcname.startswith("nan")
    if op is None:
        # the backend does not implement the function - the policy of the
        # temporal input still refuses what numpy refuses for it
        op = _missing_operation(funcname)
    # the kernel of the backend that takes the values as they are - only the
    # functions of TEMPORAL_NATIVE_FUNCS have one that can
    is_native = native and base in TEMPORAL_NATIVE_FUNCS
    # the nan-variants of the argreductions keep every row - the positions
    # they name are the ones of the input - so the NaT rows stand in as the
    # extreme tick the reduction skips
    swaps_nat = is_nan and base in ("argmin", "argmax")
    # the nan-variants skip NaT like they skip nan - the plain ones
    # propagate it, and the cumulative ones carry it through their group
    # like numpy does.  The nan mean does not filter them: the tick mean
    # leaves the NaT of a group out on its own, and filtering the rows would
    # cost a copy of the data.  The nan median filters them in its own
    # backend: the kernel of the numba one leaves the NaT tick out, the
    # exact one of the numpy one drops its rows.  The topk and the mode have
    # no nan variant, and they leave the NaT out as well: a NaT is in no
    # selection of theirs - neither the largest nor the smallest value of a
    # group, and never the most frequent one.
    #
    # the operation of the backend on the ticks can carry the NaT policy of
    # the function itself - the plain form poisons the group of a NaT and
    # the nan form leaves the NaT out - and then the engine does neither:
    # no row of the NaT is dropped, and the answer of the kernel is not
    # patched afterwards.  The kernel of the numba backend does that for
    # the mean, the median and the quantile.
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
    # own - its tick is the smallest one - while the plain argmax never
    # picks it, and numpy answers the position of the first NaT of the
    # group.  A single NaT poisons the sum, the maximum and - for the
    # durations - the middles of its whole group.  The sort puts the NaT to
    # the end of its group although it compares as the smallest value
    # everywhere else (numpy/numpy#12629, fixed by PR numpy/numpy#12658),
    # and the accumulation carries the NaT through to the end of its group,
    # nancumsum included.
    if base in ("argmin", "argmax"):
        patch = "argmax" if base == "argmax" else None
    elif base in TEMPORAL_NAT_POISON and not is_nan and not handles_nat:
        # the plain variants of these poison their group with a NaT - unless
        # the operation of the backend on the ticks poisons it on its own,
        # which the kernel of the numba backend does for the mean, the
        # median and the quantile
        patch = "poison"
    elif base == "sort":
        patch = "sort"
    elif base in ("cumsum", "cummax", "cummin") and not is_native:
        # the native backend keeps its rows: its kernels carry the NaT on
        # their own
        patch = "running"
    else:
        patch = None
    # the questions - the missing-value flags are the plain any and all of
    # the NaT mask, and their operation is resolved here
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
# The stages of the call, one per concern: the refusal of what numpy
# refuses, the scalar, the kernel that takes the values as they are, the
# rows of the NaT, the questions, the patches of the NaT, the answer of the
# operation on the ticks, the answer of the kernel, and the view back into
# the unit of the input.


@dataclass
class _Patches:
    """The patches the NaT of the values puts on the answer of the kernels.

    A NaT poisons the group it is in, the argreductions name its position,
    the accumulations carry it through to the end of their group and the
    sort stands a sentinel in for it.  The patches are gathered here, so
    that the stage that answers and the stage that repairs the answer stay
    apart.
    """

    # the groups a NaT poisons
    nat_groups: npt.NDArray[np.bool_] | None = None
    # the position of the first NaT of every group - the argmax names it
    nat_positions: npt.NDArray[np.integer] | None = None
    # the positions from the first NaT of their group on
    running_nat: npt.NDArray[np.bool_] | None = None
    # whether the largest tick stood in for a NaT while sorting
    sort_nat_restore: bool = False


@dataclass
class _Context:
    """The state of one temporal call: the values as they are now, and the
    answers of the caller."""

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
    # the ticks of the values - the int64 the kernels work on
    ticks: npt.NDArray[np.int64] = field(init=False)
    # the groups that hold nothing but the missing value - the argreductions
    # of the nan variants have no position to name for them
    empty_groups: npt.NDArray[np.bool_] | None = None
    # what the NaT does to the answer of the kernel
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
    """The fill value of a kernel that takes the temporal values as they are.

    A position or a count is filled with the number of the function, a flag
    with a bool, the collected values with an object - and every value of
    the input with the NaT or the zero of the unit.
    """
    policy = ctx.policy
    base = policy.base
    if base in ("len", "nanlen", "nunique", "nannunique", "argmin", "argmax", "nanargmin", "nanargmax"):
        return resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(np.int64))
    if base in ("allnan", "anynan"):
        return resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(np.bool_))
    if base == "array":
        return resolve_fill_value(policy.funcname, ctx.fill_value, np.dtype(object))
    # a value of the input: the fill is the NaT or the zero of the unit
    return _temporal_fill_value(ctx.fill_value, ctx.a_dtype, base)


def _swap_nat_rows(ctx: _Context) -> None:
    """The swap of the NaT rows of the nan-variants of the argreductions.

    They keep every row - the positions they name are the ones of the
    input - so the NaT rows stand in as the extreme tick the reduction
    skips.  A group of nothing but NaT holds no tick of its own after the
    swap, and is left to the fill value like its float counterpart.
    """
    nat = np.isnat(ctx.values)
    skip = np.iinfo(np.int64).max if ctx.policy.base == "argmin" else NAT_TICK + 1
    ctx.values = np.where(nat, np.int64(skip), ctx.values.view(np.int64)).view(ctx.a_dtype)
    # the groups that hold nothing but NaT have no tick to name - the
    # sentinel above stands in for the NaT of every one of them, and the
    # reduction would name its position.  They keep the fill value, like
    # the groups of a nan group of floats that hold nothing but nans.
    only_nat = _groups_holding(ctx.group_idx, nat, ctx.size)
    counts = np.bincount(ctx.group_idx, minlength=ctx.size)
    only_nat &= counts == np.bincount(ctx.group_idx, weights=nat, minlength=ctx.size)
    ctx.empty_groups = only_nat


def _drop_nat_rows(ctx: _Context) -> None:
    """The NaT rows the nan-variants and the topk leave out.

    The rows are dropped here and the kernel drops the NaT again: the
    kernels are public entry points of their own, and one that is handed a
    NaT has to leave it out without the wrapper having prepared anything.
    """
    nat = np.isnat(ctx.values)
    if not nat.any():
        return
    good = ~nat
    ctx.values = ctx.values[good]
    ctx.group_idx = ctx.group_idx[good]


def _flag_result(ctx: _Context) -> tuple[AResult, AFillValue]:
    """The answer of the functions that ask a question instead of a value.

    The truth of a temporal value is its tick being non-zero and the value
    not being NaT - a NaT is no value at all - and the missing-value flags
    are the plain any and all of the NaT mask.
    """
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
    """The patches the NaT of the values puts on the answer of the kernels."""
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
        # the largest tick stands in for the NaT on the way through the sort,
        # and the answer turns it back into NaT
        ctx.ticks = np.where(nat, np.iinfo(np.int64).max, ctx.ticks)
        patches.sort_nat_restore = True
        return
    patches.running_nat = _per_group_running_mask(ctx.group_idx, nat)


def _mean_result(ctx: _Context) -> tuple[npt.NDArray[Any] | None, bool, bool]:
    """The mean of the values - as a float, or as a tick.

    The float answer is the exact quotient itself, not the rounded tick:
    the tick mean of the backend measures the deviations from the first
    value of the group, so the sum of the ticks stays exact and the answer
    is a tick again - the plain kernel poisons the group of a NaT on its
    own, the nan one leaves the NaT out.  A backend without a kernel gets
    the exact mean of this module, which holds the sum of the ticks in
    python integers.

    The answer of the float is final: it is a float and not a tick, so it
    is not viewed back into the dtype of the input.
    """
    policy = ctx.policy
    fill_tick = _temporal_fill_tick(ctx.fill_value, ctx.a_dtype, True)
    if ctx.user_dtype is not None and ctx.user_dtype.kind == "f":
        mean_idx, mean_ticks = ctx.group_idx, ctx.ticks
        nat_groups_for_float = None
        if policy.funcname.startswith("nan"):
            # the NaT rows have no tick to average - the nan variant leaves
            # them out, like its kernel does
            good = ctx.ticks != NAT_TICK
            mean_idx, mean_ticks = ctx.group_idx[good], ctx.ticks[good]
        else:
            # a single NaT poisons its group - the float answer of a group
            # without a quotient is nan
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
        # The values of a group reach from one end of the int64 range to
        # the other: their sum of deviations leaves the range, and the
        # kernel refuses the mean rather than wrapping it around silently.
        # The limb sums of the ticks hold it exactly.
        if policy.funcname.startswith("nan"):
            # the nan kernel never sums the NaT ticks - their rows are no
            # part of its deviations - so the limb sums take the values it
            # would have averaged
            good = ctx.ticks != NAT_TICK
            computed = cast(
                "npt.NDArray[Any]",
                exact_integer_group_mean(ctx.group_idx[good], ctx.ticks[good], ctx.size, fill_tick, np.dtype(np.int64)),
            )
        else:
            # a NaT of the group is what may have broken the span of its
            # deviations - the poison comes back through the patch
            nat = np.isnat(ctx.values)
            if nat.any():
                ctx.patches.nat_groups = _groups_holding(ctx.group_idx, nat, ctx.size)
            computed = cast(
                "npt.NDArray[Any]",
                exact_integer_group_mean(ctx.group_idx, ctx.ticks, ctx.size, fill_tick, np.dtype(np.int64)),
            )
    return computed, False, True


def _median_result(ctx: _Context) -> tuple[npt.NDArray[Any] | None, bool, bool]:
    """The middle of the ticks of every group.

    The kernel of the backend selects it in compiled code, a backend
    without one gets the exact median of this module - the plain variant
    answers the groups that hold a NaT through the poison of the patch, the
    nan one leaves the NaT tick out on its own.
    """
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
    """The quantile of the ticks of every group.

    The method of the interpolation is the one the caller names - numpy
    knows five of them, and the temporal answer follows the one it is
    given.
    """
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
    """The answer the wrapper computes itself - the mean, the median and the
    quantiles, where the ticks make the arithmetic exact.

    The answer is ``None`` for every other function: those are answered by
    the kernel of their operation.  The second value says whether the
    answer is the final one (the float mean is), the third whether it is a
    temporal value - its ticks have to be viewed back.
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
    """The answer of the kernel of the operation.

    The kernels answer in ticks, and the fill value is a tick, for the
    functions whose answer is a value of the input; the positions, the
    counts and the flags are answered in their own dtypes, with the fill
    value of their function.
    """
    policy = ctx.policy
    if computed is not None:
        return computed
    if value_result:
        fill_tick = _temporal_fill_tick(ctx.fill_value, ctx.a_dtype, policy.fill_is_nat)
        return ctx.call(policy.op, ctx.ticks, fill_tick, dtype=np.dtype(np.int64), **ctx.kwargs)
    out_dtype = ctx.user_dtype if ctx.user_dtype is not None else np.dtype(np.int64)
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
    """The answer of the kernels, repaired by the patches of the NaT.

    The accumulation of a group carries a NaT through to its end, the sort
    turns its sentinel back into a NaT, the poisoned groups answer NaT, the
    argmax names the position of the first NaT of its group, and the groups
    that hold nothing but the missing value keep the fill value of their
    function - they have no position to name.
    """
    patches = ctx.patches
    if patches.running_nat is not None:
        ret = np.asarray(ret)
        ret[patches.running_nat] = NAT_TICK
    if patches.sort_nat_restore:
        # the largest tick stood in for NaT while sorting - hand it back
        ret = np.asarray(ret)
        ret[ret == np.iinfo(np.int64).max] = NAT_TICK
    if patches.nat_groups is not None:
        ret = np.asarray(ret)
        if patches.nat_positions is not None:
            ret[patches.nat_groups] = patches.nat_positions[patches.nat_groups]
        else:
            ret[patches.nat_groups] = NAT_TICK
    if ctx.empty_groups is not None:
        # the groups that hold nothing but the missing value have no
        # position to name - they keep the fill value of their function
        ret = np.asarray(ret)
        empty_fill = resolve_fill_value(ctx.policy.funcname, ctx.fill_value, np.dtype(np.int64))
        ret[ctx.empty_groups] = cast("int", empty_fill if empty_fill is not None else -1)
    return ret


def _view_result(ret: AResult, a_dtype: np.dtype, user_dtype: np.dtype | None) -> AResult:
    """The answer of the kernels, turned back into a temporal one.

    The unit of the input is the unit of the answer - unless the caller
    asks for another one, in which case the ticks are rescaled into it, or
    for the ticks themselves.
    """
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
    """The answer of an aggregation of temporal values, and the fill it used.

    Everything temporal lives here: what numpy refuses for temporal input, we
    refuse as well; the plain reductions hand a NaT on to their group and the
    nan variants leave it out; the mean, the median and the quantiles compute
    on the int64 ticks of the values, where the arithmetic is exact; and the
    answer is a temporal one, in the unit of the input.

    The stages of the call are the functions of this module, and the policy
    of the function - what it refuses, what its kernel takes, what the NaT
    does to its answer - is resolved when the operation is built.
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
        # a temporal scalar comes through as its tick - sum and len of a
        # scalar timedelta64 answer in int64, like a plain integer scalar
        ret = ctx.call(policy.op, cast("AValue", np.int64(values.view(np.int64))), fill_value, **kwargs)
        return ret, fill_value

    plain_fill = isinstance(fill_value, (bool, np.bool_, int, np.integer))
    if policy.is_native and ctx.user_dtype is None and not plain_fill:
        # the kernel takes the temporal values as they are - it finds the NaT
        # of every group, its position and the empty groups on its own, in
        # one pass, and the wrapper has nothing to prepare for it
        fill = _native_fill_value(ctx)
        ret = ctx.call(policy.op, ctx.values, fill, **kwargs)
        return ret, fill_value

    if policy.swaps_nat:
        _swap_nat_rows(ctx)
    elif policy.drops_nat:
        _drop_nat_rows(ctx)
    # the ticks of the values as they are now - the nan-variants dropped
    # their NaT rows
    ctx.ticks = ctx.values.view(np.int64)

    if policy.flag is not None:
        return _flag_result(ctx)

    _nan_patches(ctx)

    computed, computed_final, value_result = _ticks_result(ctx)

    if policy.is_array:
        return _array_result(ctx)

    ret = _kernel_result(ctx, computed, value_result)

    if computed is not None and computed_final:
        # the float mean is a float and not a tick - it is the final answer
        return ret, fill_value

    ret = _patch_result(ctx, ret)

    if value_result:
        return _view_result(ret, ctx.a_dtype, ctx.user_dtype), fill_value
    return ret, fill_value
