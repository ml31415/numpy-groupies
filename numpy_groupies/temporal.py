"""The temporal policy of the numpy based backends.

Everything both numpy based backends need to know about datetime64 and
timedelta64 input lives here: the functions a dtype refuses (the ones numpy
refuses itself), the functions whose compiled kernels take the values as
they are, what a plain reduction does with the NaT of a group and what the
nan variants leave out, and the fills of the answers.  The backends hand
their kernels to ``_temporal_aggregation`` - the numpy one its tick mean
and its exact tick median, the numba one the compiled forms of both - and
the wrapper answers for them.

The dtype facts stay in ``nptools``: the kinds of the temporal dtypes.
The order statistics of the numpy backend live in
``aggregate_numpy.specials`` - the exact median of the ticks - and the
numba kernels in ``aggregate_numba``: nothing in this module reaches into
a backend, so both of them reach into this module.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.common import resolve_fill_value
from numpy_groupies.nptools import exact_integer_group_mean
from numpy_groupies.registry import DEFAULT_FILL_VALUE, missing_impl_error
from numpy_groupies.typing import AFillValue, AResult, AValue

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
# the kinds of the temporal dtypes - datetime64 and timedelta64, the "M"
# and the "m" of numpy
TEMPORAL_KINDS = "Mm"

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
TEMPORAL_NAT_FILL = frozenset({"min", "max", "mean", "median", "quantile", "percentile", "first", "last"})


def is_temporal_input(a: Any) -> bool:
    """Whether the input carries datetime64 or timedelta64 values - an array
    of them, or the one scalar standing for a value of every group.

    The kind of the dtype is asked, not the dtype itself: the check runs on
    every call, and building a dtype out of a dtype is not free.
    """
    kind = getattr(getattr(a, "dtype", None), "kind", None)
    return kind is not None and kind in TEMPORAL_KINDS


def is_own_function(funcname: str | Callable[..., Any]) -> bool:
    """Whether the func names one of the functions of the package.

    Every alias of the package resolves to the name of its function - the
    names themselves, and the numpy functions that mean the same thing
    (``np.mean`` is the mean of the package, ``np.max`` its maximum, and so
    on).  A callable of the caller's own does not resolve to a name, and it
    defines its own semantics: it sees the values as they are, the temporal
    dtypes included.
    """
    return isinstance(funcname, str)


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


def _temporal_aggregation(
    resolve: Callable[[str], Callable[..., AResult]],
    tick_mean: Callable[..., AResult],
    tick_median: Callable[..., AResult],
    tick_quantile: Callable[..., AResult],
    funcname: str,
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    size: int,
    fill_value: AFillValue,
    dtype: npt.DTypeLike | None,
    kwargs: dict[str, Any],
    native: bool = False,
) -> tuple[AResult, AFillValue]:
    """The answer of an aggregation of temporal values, and the fill it used.

    Everything temporal lives here: what numpy refuses for temporal input, we
    refuse as well; the plain reductions hand a NaT on to their group and the
    nan variants leave it out; the mean, the median and the quantiles compute
    on the int64 ticks of the values, where the arithmetic is exact; and the
    answer is a temporal one, in the unit of the input.

    The operation of the function is resolved through ``resolve``, so that
    the caller does not have to know whether the wrapper computed the answer
    itself or the kernel of the operation did.

    ``native`` says that the kernel of the backend takes the temporal values
    as they are: the compiled kernels of the numba backend order them, find
    their NaT, and add and accumulate the durations on their own, in one
    pass over the data, and the wrapper has nothing to prepare for them.
    The numpy backend works on the ticks, so it leaves the flag alone.

    The mean, the median and the quantile of the ticks are the three answers
    that no kernel of the package computes on its own - a backend hands them
    in as ``tick_mean``, ``tick_median`` and ``tick_quantile``: the numba
    backend has compiled kernels for the first two, whose deviations from
    the first value of a group and selection of the middle of a block run in
    one pass, and its quantile goes to the exact path of the numpy backend -
    whose python integers hold the sum of the ticks of a group and the
    interpolation of the quantile exactly, where the sum leaves the window
    float64 represents and no int64 arithmetic holds the product of a tick
    difference and a fraction.  The kernels take the ``nan`` flag, because
    the nan variant leaves the NaT of a group out on its own instead of
    having the rows filtered out before it.
    """

    def operation(name: str) -> Callable[..., AResult]:
        """The operation of a name, and the complaint if there is none."""
        op = resolve(name)
        if op is None:
            raise missing_impl_error(name)
        return op

    base = funcname.removeprefix("nan")
    values = np.asarray(a)
    a_dtype = values.dtype
    is_datetime = a_dtype.kind == "M"
    rejected = TEMPORAL_DATETIME_REJECTED if is_datetime else TEMPORAL_TIMEDELTA_REJECTED
    if base in rejected:
        raise TypeError(
            f"aggregate {funcname!r} is not supported for {a_dtype.name} input - "
            f"numpy refuses {base} of {'datetime64' if is_datetime else 'timedelta64'} values as well"
        )
    user_dtype = np.dtype(dtype) if dtype is not None else None

    # what the NaT of the values does to the answer: the groups it poisons,
    # the positions it holds, and the running mask of the accumulations
    nat: npt.NDArray[np.bool_] | None = None
    nat_groups: npt.NDArray[np.bool_] | None = None
    nat_positions: npt.NDArray[np.integer] | None = None
    running_nat: npt.NDArray[np.bool_] | None = None
    sort_nat_restore = False
    # the answer the wrapper computes itself (the mean, the median and the
    # quantiles), and whether it is the final one
    computed: npt.NDArray[Any] | None = None
    computed_final = False
    # whether the answer is a temporal value - the ticks of it have to be
    # viewed back
    value_result = False

    if not isinstance(a, np.ndarray):
        # a temporal scalar comes through as its tick - sum and len of a
        # scalar timedelta64 answer in int64, like a plain integer scalar
        op = operation(funcname)
        ret = op(
            group_idx,
            cast("AValue", np.int64(values.view(np.int64))),
            size,
            fill_value=fill_value,
            dtype=None,
            **kwargs,
        )
        return cast("AResult", ret), fill_value

    # the groups that hold nothing but the missing value - the argreductions
    # of the nan variants have no position to name for them
    empty_groups: npt.NDArray[np.bool_] | None = None
    # a fill value that is a plain number is a tick, and a dtype the caller
    # asks for may be the ticks or another unit - both are the business of
    # the tick path below
    plain_fill = isinstance(fill_value, (bool, np.bool_, int, np.integer))
    if native and base in TEMPORAL_NATIVE_FUNCS and dtype is None and not plain_fill:
        # the kernel takes the temporal values as they are - it finds the NaT
        # of every group, its position and the empty groups on its own, in
        # one pass, and the wrapper has nothing to prepare for it
        op = operation(funcname)
        if base in ("len", "nanlen", "nunique", "nannunique", "argmin", "argmax", "nanargmin", "nanargmax"):
            # a position or a count: the fill is the number the function uses
            fill = resolve_fill_value(funcname, fill_value, np.dtype(np.int64))
        elif base in ("allnan", "anynan"):
            fill = resolve_fill_value(funcname, fill_value, np.dtype(np.bool_))
        elif base == "array":
            fill = resolve_fill_value(funcname, fill_value, np.dtype(object))
        else:
            # a value of the input: the fill is the NaT or the zero of the unit
            fill = _temporal_fill_value(fill_value, a_dtype, base)
        ret = op(group_idx, a, size, fill_value=fill, dtype=None, **kwargs)
        return cast("AResult", ret), fill_value

    if funcname.startswith("nan") and base in ("argmin", "argmax"):
        # the nan-variants of the argreductions keep every row - the
        # positions they name are the ones of the input - so the NaT rows
        # stand in as the extreme tick the reduction skips.  A group of
        # nothing but NaT holds no tick of its own after the swap, and is
        # left to the fill value like its float counterpart
        nat = np.isnat(a)
        skip = np.iinfo(np.int64).max if base == "argmin" else NAT_TICK + 1
        a = np.where(nat, np.int64(skip), values.view(np.int64)).view(a.dtype)
        # the groups that hold nothing but NaT have no tick to name - the
        # sentinel above stands in for the NaT of every one of them, and the
        # reduction would name its position.  They keep the fill value, like
        # the groups of a nan group of floats that hold nothing but nans.
        only_nat = _groups_holding(group_idx, nat, size)
        counts = np.bincount(group_idx, minlength=size)
        only_nat &= counts == np.bincount(group_idx, weights=nat, minlength=size)
        nat = None
        # the groups that hold nothing but NaT are answered the fill value
        # below - the mask of the groups that hold a NaT is not theirs, for
        # the plain forms the NaT is the value they name
        empty_groups = only_nat
    elif (
        funcname.startswith("nan")
        and base not in ("cumsum", "mean")
        and not (base == "median" and tick_median is not None)
    ):
        # the nan-variants skip NaT like they skip nan - the plain ones
        # propagate it, and the cumulative ones carry it through their group
        # like numpy does.  Deliberately unlike numpy for the argreductions:
        # np.nanargmin and np.nanargmax do not skip a NaT of datetime64,
        # they answer its position like the plain forms, while np.nanmin
        # and np.nanmax do skip it (numpy/numpy#5222 - closed as fixed in
        # 2020, while its own closing output still shows the nanarg forms of
        # datetime64 answering the NaT position where the float nanarg
        # skips, unaddressed; reported upstream as numpy/numpy#33207) - the
        # contract of the nan forms says skip, and we keep it for every
        # dtype.  The nan mean does not filter them
        # either: the tick mean leaves the NaT of a group out on its own,
        # and filtering the rows would cost a copy of the data.  The nan
        # median filters them in its own backend: the kernel of the numba
        # one leaves the NaT tick out, the exact one of the numpy one drops
        # its rows.  The nan count of the distinct values does not filter
        # them either: the hash table of the kernel skips a NaT like it
        # skips a nan, and it takes the values as they are.
        nat = np.isnat(a)
        if nat.any():
            good = ~nat
            a = a[good]
            group_idx = group_idx[good]
            nat = None
    # the values as they are now - the nan-variants dropped their NaT rows
    values = np.asarray(a)

    if base in ("all", "any"):
        # the truth of a temporal value is its tick being non-zero and the
        # value not being NaT - a NaT is no value at all
        truth: npt.NDArray[np.bool_] = values.view(np.int64) != 0
        truth &= ~np.isnat(values)
        op = operation(funcname)
        ret = op(
            group_idx,
            truth,
            size,
            fill_value=resolve_fill_value(funcname, fill_value, np.dtype(np.bool_)),
            dtype=None,
            **kwargs,
        )
        return cast("AResult", ret), fill_value
    if base in ("allnan", "anynan"):
        # the missing-value flags are the plain any/all of the NaT mask
        op = operation("any" if base == "anynan" else "all")
        ret = op(
            group_idx,
            np.isnat(a),
            size,
            fill_value=resolve_fill_value("all", fill_value, np.dtype(np.bool_)),
            dtype=None,
            **kwargs,
        )
        return cast("AResult", ret), fill_value

    ticks = values.view(np.int64)
    # the NaT of the values, where the ticks of the kernels alone do not
    # know it - every branch of the chain below is the one of its operation,
    # so no call computes the mask more than once
    if base in ("argmin", "argmax"):
        # the plain argmin finds the NaT on its own - its tick is the
        # smallest one - while the plain argmax never picks it, and numpy
        # answers the position of the first NaT of the group
        if base == "argmax":
            nat = np.isnat(values)
            if nat.any():
                nat_groups, nat_positions = _first_true_position_per_group(group_idx, nat, size)
    elif base in TEMPORAL_NAT_POISON and not funcname.startswith("nan") and not (native and base == "mean"):
        # a single NaT poisons the sum, the maximum and - for the durations
        # - the middles of its whole group.  The mean of the native backend
        # is not among them: its kernel poisons the group on its own, and
        # the numpy backend poisons through the patch further down.
        nat = np.isnat(values)
        if nat.any():
            nat_groups = _groups_holding(group_idx, nat, size)
    elif base == "sort":
        # numpy sorts NaT to the end of its group although it compares as the
        # smallest value everywhere else (numpy/numpy#12629, fixed by PR
        # numpy/numpy#12658) - the largest tick stands in for it on the way
        # through the sort, and the answer turns it back into NaT
        nat = np.isnat(values)
        if nat.any():
            ticks = np.where(nat, np.iinfo(np.int64).max, ticks)
            sort_nat_restore = True
    elif base in ("cumsum", "cummax", "cummin") and not native:
        # the accumulation carries the NaT through to the end of its group,
        # nancumsum included - numpy propagates it for temporal input, and
        # the groups are poisoned one by one.  The native backend keeps its
        # rows: its kernels carry the NaT on their own.
        nat = np.isnat(values)
        if nat.any():
            running_nat = _per_group_running_mask(group_idx, nat)

    if base == "mean":
        fill_tick = _temporal_fill_tick(fill_value, a_dtype, True)
        if user_dtype is not None and user_dtype.kind == "f":
            # a float answer is the exact quotient itself, not the rounded tick
            mean_idx, mean_ticks = group_idx, ticks
            nat_groups_for_float = None
            if funcname.startswith("nan"):
                # the NaT rows have no tick to average - the nan variant
                # leaves them out, like its kernel does
                good = ticks != NAT_TICK
                mean_idx, mean_ticks = group_idx[good], ticks[good]
            else:
                # a single NaT poisons its group - the float answer of a
                # group without a quotient is nan
                nat = np.isnat(values)
                if nat.any():
                    nat_groups_for_float = _groups_holding(group_idx, nat, size)
            computed = cast(
                "npt.NDArray[Any]",
                exact_integer_group_mean(mean_idx, mean_ticks, size, fill_tick, user_dtype),
            )
            if nat_groups_for_float is not None:
                computed = cast("npt.NDArray[Any]", np.asarray(computed))
                computed[nat_groups_for_float] = np.nan
            computed_final = True
        else:
            # the tick mean: the kernel of the backend measures the
            # deviations from the first value of the group, so the sum of
            # the ticks stays exact and the answer is a tick again - the
            # plain kernel poisons the group of a NaT on its own, the nan one
            # leaves the NaT out.  A backend without a kernel gets the exact
            # mean of this module, which holds the sum of the ticks in
            # python integers.
            try:
                computed = cast(
                    "npt.NDArray[Any]",
                    tick_mean(
                        group_idx,
                        ticks,
                        size,
                        fill_value=fill_tick,
                        dtype=np.dtype(np.int64),
                        nan=funcname.startswith("nan"),
                        **kwargs,
                    ),
                )
            except OverflowError:
                # The values of a group reach from one end of the int64 range
                # to the other: their sum of deviations leaves the range, and
                # the kernel refuses the mean rather than wrapping it around
                # silently.  The limb sums of the ticks hold it exactly.
                if funcname.startswith("nan"):
                    # the nan kernel never sums the NaT ticks - their rows
                    # are no part of its deviations - so the limb sums take
                    # the values it would have averaged
                    good = ticks != NAT_TICK
                    computed = cast(
                        "npt.NDArray[Any]",
                        exact_integer_group_mean(group_idx[good], ticks[good], size, fill_tick, np.dtype(np.int64)),
                    )
                else:
                    # a NaT of the group is what may have broken the span of
                    # its deviations - the poison comes back through the
                    # patch further down
                    nat = np.isnat(values)
                    if nat.any():
                        nat_groups = _groups_holding(group_idx, nat, size)
                    computed = cast(
                        "npt.NDArray[Any]",
                        exact_integer_group_mean(group_idx, ticks, size, fill_tick, np.dtype(np.int64)),
                    )
            value_result = True
    elif base in ("median", "quantile", "percentile"):
        # only timedelta64 reaches this point - datetime64 has no middles
        fill_tick = _temporal_fill_tick(fill_value, a_dtype, True)
        if base == "median":
            # the middle of the ticks of every group: the kernel of the
            # backend selects it in compiled code, a backend without one
            # gets the exact median of this module - the plain variant
            # answers the groups that hold a NaT through the poison below,
            # the nan one leaves the NaT tick out on its own
            computed = cast(
                "npt.NDArray[Any]",
                tick_median(
                    group_idx,
                    ticks,
                    size,
                    fill_value=fill_tick,
                    dtype=np.dtype(np.int64),
                    nan=funcname.startswith("nan"),
                    **kwargs,
                ),
            )
        else:
            fraction = cast("float", kwargs.get("q", 50.0 if base == "percentile" else 0.5))
            if base == "percentile":
                fraction = fraction / 100.0
            # the method of the interpolation - numpy knows five of them, and
            # the temporal answer follows the one the caller names
            method = cast("str", kwargs.get("method", "linear"))
            computed = tick_quantile(group_idx, ticks, size, fill_tick, fraction, method)
        value_result = True
    elif base in TEMPORAL_VALUE_FUNCS:
        value_result = True

    if base == "array":
        # the values of every group, back in the dtype they came in
        op = operation(funcname)
        empty = resolve_fill_value(funcname, fill_value, np.dtype(object))
        ret = op(group_idx, ticks, size, fill_value=empty, dtype=None, **kwargs)
        groups = ret if isinstance(ret, np.ndarray) else np.asarray(ret, dtype=object)
        return (
            cast("AResult", np.array([np.asarray(one).view(a_dtype) for one in groups], dtype=object)),
            fill_value,
        )

    if computed is not None:
        # the wrapper has computed the answer itself - the mean, the median
        # and the quantiles, where the ticks make the arithmetic exact
        ret = computed
    elif value_result:
        # the kernels answer in ticks, the fill is a tick, and the dtype they
        # are asked for is int64
        fill_tick = _temporal_fill_tick(fill_value, a_dtype, base in TEMPORAL_NAT_FILL)
        op = operation(funcname)
        ret = op(group_idx, ticks, size, fill_value=fill_tick, dtype=np.dtype(np.int64), **kwargs)
    else:
        # positions, counts and flags: the kernels answer in their own dtypes
        # and their fill value is the one of their function
        out_dtype = user_dtype if user_dtype is not None else np.dtype(np.int64)
        if base in ("allnan", "anynan"):
            out_dtype = np.dtype(np.bool_)
        op = operation(funcname)
        ret = op(
            group_idx,
            ticks,
            size,
            fill_value=resolve_fill_value(funcname, fill_value, out_dtype),
            dtype=user_dtype,
            **kwargs,
        )

    # the answer of the kernels, turned back into a temporal one
    if computed is not None and computed_final:
        return cast("AResult", ret), fill_value
    if running_nat is not None:
        ret = np.asarray(ret)
        ret[running_nat] = NAT_TICK
    if sort_nat_restore:
        # the largest tick stood in for NaT while sorting - hand it back
        ret = np.asarray(ret)
        ret[ret == np.iinfo(np.int64).max] = NAT_TICK
    if nat_groups is not None:
        ret = np.asarray(ret)
        if nat_positions is not None:
            ret[nat_groups] = nat_positions[nat_groups]
        else:
            ret[nat_groups] = NAT_TICK
    if empty_groups is not None:
        # the groups that hold nothing but the missing value have no
        # position to name - they keep the fill value of their function
        ret = np.asarray(ret)
        empty_fill = resolve_fill_value(funcname, fill_value, np.dtype(np.int64))
        ret[empty_groups] = cast("int", empty_fill if empty_fill is not None else -1)
    if value_result:
        ret = np.asarray(ret)
        if user_dtype is not None:
            if user_dtype == np.dtype(np.int64):
                return cast("AResult", ret), fill_value  # the ticks the caller asked for
            if user_dtype.kind in TEMPORAL_KINDS:
                # rescale from the unit of the input to the one asked for
                return cast("AResult", ret.view(a_dtype).astype(user_dtype)), fill_value
            return cast("AResult", ret.view(np.int64).astype(user_dtype)), fill_value
        return cast("AResult", ret.view(a_dtype)), fill_value
    return cast("AResult", ret), fill_value
