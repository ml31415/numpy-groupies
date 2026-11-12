"""The composition of the pure python implementation: the groups of the values
and the answer of the operation of every group."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from typing import Any, cast

from numpy_groupies.npg_purepy.ops import (
    _ONE_PER_ITEM,
    _TAKES_INDEX_FIRST,
    _sort,
    _topk,
)


def answer_groups(
    pos: Sequence[int],
    idx: Sequence[Any],
    vals: Sequence[Any],
    op: Any,
    size: int,
    fill_value: Any,
    kwargs: dict[str, Any],
    k: int,
    largest: bool,
) -> list[Any]:
    """The answer of every group, the operation of the group applied to it."""
    if op is _sort:
        return cast("list[Any]", _sort(idx, vals, reverse=kwargs.get("reverse", False)))
    if op in _ONE_PER_ITEM:
        # the running values answer one value per input item, in the order of
        # the input - the places of a group hold its running values
        ret = [fill_value] * len(pos)
        running_data = sorted(zip(pos, idx, vals), key=operator.itemgetter(1))
        for _ix, entries_of_group in itertools.groupby(running_data, key=operator.itemgetter(1)):
            group_entries: list[tuple[Any, Any, Any]] = list(entries_of_group)  # the groupby iterator is spent
            running = op([val for _, _, val in group_entries], fill_value=fill_value, **kwargs)
            for (p, _ix2, _val), value in zip(group_entries, running):
                ret[p] = value
        return ret
    if op is _topk:
        # k values per group, a list per group - there are no arrays here
        topk_ret: list[Any] = [[fill_value] * k for _ in range(size)]
        topk_data = sorted(zip(idx, vals), key=operator.itemgetter(0))
        for ix, group in itertools.groupby(topk_data, key=operator.itemgetter(0)):
            topk_ret[ix] = _topk([val for _, val in group], k, largest, fill_value)
        return topk_ret

    # sort data and evaluate function on groups
    ret = [fill_value] * size
    if op not in _TAKES_INDEX_FIRST:
        data = sorted(zip(idx, vals), key=operator.itemgetter(0))
        for ix, values_of_group in itertools.groupby(data, key=operator.itemgetter(0)):
            value = op([val for _, val in values_of_group], **kwargs)
            if value is not None:  # no answer: the group stays at the fill value
                ret[ix] = value
    else:
        # a group holding a nan has no extreme to point at, and so no position
        poisoned = {ix for ix, val in zip(idx, vals) if val != val}
        data_with_pos = sorted(zip(pos, idx, vals), key=operator.itemgetter(1))
        for ix, entries in itertools.groupby(data_with_pos, key=operator.itemgetter(1)):
            if ix in poisoned:
                continue  # stays at the fill value
            ret[ix] = op([(p, val) for p, _, val in entries], **kwargs)

    return ret
