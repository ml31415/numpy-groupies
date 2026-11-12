"""The tools of the groups: the ways to lay them out, to number them and to
carry their values.
"""

from collections.abc import Iterable, Sequence
from typing import Any, SupportsFloat, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.fills import DefaultFillMarker
from numpy_groupies.typing import AScalar
from numpy_groupies.validation import EXACT_FLOAT64_INTEGER_LIMIT


def multi_arange(n: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """Concatenated ``arange``s, e.g. ``n = [0, 0, 3, 0, 2, 2, 1]`` gives
    ``[0, 1, 2, 0, 1, 0, 1, 0]``. Faster than ``hstack((arange(n_i) for n_i in n))``.
    """
    if n.ndim != 1:
        raise ValueError("n is supposed to be 1d array.")
    if n.size == 0:
        # no group, no value
        return np.zeros(0, dtype=np.intp)

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1] + 1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return cast("npt.NDArray[np.intp]", np.cumsum(ret)[:-1])


def label_contiguous_1d(X: npt.NDArray[np.number]) -> npt.NDArray[np.intp]:
    """Label each contiguous block of truthy (or equal) values, starting at 1.
    E.g. ``[F T T F F T]`` -> ``[0 1 1 0 0 2]``.

    WARNING: the API is liable to change!
    """

    if X.ndim != 1:
        raise ValueError("this is for 1d masks only.")

    is_start = np.empty(len(X), dtype=bool)
    is_start[0] = X[0]  # True if X[0] is True or non-zero

    if X.dtype.kind == "b":
        # the kind test has established a boolean array, the signature says generic
        M = cast("npt.NDArray[np.bool_]", X)
        is_start[1:] = ~M[:-1] & M[1:]
    else:
        M = X.astype(bool)
        is_start[1:] = X[:-1] != X[1:]
        is_start[~M] = False

    L = np.cumsum(is_start)
    L[~M] = 0
    return cast("npt.NDArray[np.intp]", L)


def relabel_groups_unique(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """Relabel groups to fill gaps, keeping order. See ``relabel_groups_masked``.
    E.g. unique input groups ``1,2,3,5`` -> group 5 becomes ``4``.
    """

    keep_group = np.zeros(np.max(group_idx) + 1, dtype=bool)
    keep_group[0] = True
    keep_group[group_idx] = True
    return relabel_groups_masked(group_idx, keep_group)


def relabel_groups_masked(
    group_idx: npt.NDArray[np.integer], keep_group: npt.NDArray[np.bool_]
) -> npt.NDArray[np.intp]:
    """Relabel groups so that groups masked out by ``keep_group`` are removed and the
    gaps are filled, keeping order (``keep_group[0]`` is meaningless, group 0 stays).
    E.g. removing group 2 of ``[0 3 3 0 2 5]`` relabels 3, 5 to 2, 4.
    """

    # astype(bool) is what normalizes an int mask that arrives anyway
    keep_group = keep_group.astype(bool, copy=False)
    if not keep_group[0]:  # ensuring keep_group[0] is True makes life easier
        # the copy belongs here, to the one write of its own, and not to a flag
        # that has to read the element it is about to change
        keep_group = keep_group.copy()
        keep_group[0] = True

    relabel = np.zeros(keep_group.size, dtype=group_idx.dtype)
    relabel[keep_group] = np.arange(np.count_nonzero(keep_group))
    return cast("npt.NDArray[np.intp]", relabel[group_idx])


def exact_integer_group_mean(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.integer],
    size: int,
    fill_value: AScalar,
    dtype: np.dtype,
) -> npt.NDArray[np.floating] | npt.NDArray[np.integer]:
    """Per-group mean of integer values, exact where float64 would round (e.g.
    datetime64 ticks summing beyond +/-2**53).

    Values are split into 32-bit limbs (16-bit for >2**21 values per group),
    each limb is summed exactly via ``bincount`` and the totals are combined
    as python integers; the division is exact, the remainder rounds half to
    even. An integer dtype gets the exact rounded quotient, otherwise the
    exact float64 quotient.
    """
    u = np.ascontiguousarray(a).view(np.uint64)
    # limb width: the limb sum must stay inside the exact-float64 window
    width = 32 if len(group_idx) * (2**32 - 1) <= EXACT_FLOAT64_INTEGER_LIMIT else 16
    mask = np.uint64((1 << width) - 1)
    shifts = tuple(range(0, 64, width))
    limb_sums = [
        np.bincount(
            group_idx,
            # explicit cast: masked limbs (small numbers) cast far quicker
            # than the whole bit patterns
            weights=((u >> np.uint64(shift)) & mask).astype(np.float64),
            minlength=size,
        ).astype(np.int64)
        for shift in shifts
    ]
    counts = np.bincount(group_idx, minlength=size)
    total: npt.NDArray[Any] = limb_sums[-1].astype(object)
    for limb in reversed(limb_sums[:-1]):
        total = (total << width) + limb.astype(object)
    if a.dtype.kind == "i":
        # two's-complement limb sums count one 2**64 per negative value;
        # correct by the count of negatives per group
        negatives = np.bincount(group_idx, weights=u >= np.uint64(2**63), minlength=size)
        total -= negatives.astype(np.int64).astype(object) << 64
    present = counts > 0
    quotients = np.zeros(size, dtype=object)
    remainders = np.zeros(size, dtype=object)
    quotients[present], remainders[present] = (
        np.floor_divide(total[present], counts[present].astype(object)),
        np.remainder(total[present], counts[present].astype(object)),
    )
    # empty groups: the dtype's default unless the caller asked for a
    # number - nan and the sentinel both mean "default"
    fill: int | float | None
    if isinstance(fill_value, (float, np.floating)) and fill_value != fill_value:
        fill = None  # a nan asks for the default
    elif isinstance(fill_value, (bool, np.bool_, int, np.integer)):
        fill = int(fill_value)
    elif fill_value is None or isinstance(fill_value, DefaultFillMarker):
        fill = None  # the marker of the call asks for the default
    else:
        fill = float(cast("SupportsFloat", fill_value))
    if np.issubdtype(dtype, np.integer):
        rounded = quotients.copy()
        bump = (2 * remainders > counts) | ((2 * remainders == counts) & (quotients % 2 == 1))
        rounded[bump] += 1
        ret = rounded.astype(np.int64)
    else:
        # long double holds every int64 quotient exactly, so the assembled
        # float64 stays within half an ulp of the exact quotient
        quotients = quotients.astype(np.longdouble)
        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = np.true_divide(remainders.astype(np.longdouble), counts.astype(np.longdouble))
        ret = (quotients + fractions).astype(np.float64)
        if fill is not None:
            ret[~present] = fill
        return ret
    if fill is not None and fill != 0:
        ret[~present] = fill
    return ret


def fill_untouched(idx: npt.NDArray[np.integer], ret: npt.NDArray[np.generic], fill_value: AScalar) -> None:
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value


def blocks_to_object_array(
    blocks: Iterable[npt.NDArray[np.generic]],
    group_idx: npt.NDArray[np.integer],
    fill_value: AScalar | Sequence[AScalar] | None,
) -> npt.NDArray[np.object_]:
    """The blocks of the groups as an object array, one sequence per group.

    Built element by element - ``np.asanyarray`` would collapse equal-length
    groups into a regular array, losing the sequences. Empty groups take the
    fill value. Lives here so both backends can use it without importing
    each other.
    """
    if fill_value is not None and not (
        np.isscalar(fill_value) or (isinstance(fill_value, Sequence) and len(fill_value) == 0)
    ):
        raise ValueError("fill_value must be None, a scalar or an empty sequence")
    blocks = list(blocks)
    ret = np.empty(len(blocks), dtype=object)
    for g, block in enumerate(blocks):
        ret[g] = block
    if fill_value is None or np.isscalar(fill_value):
        fill_untouched(group_idx, ret, cast("AScalar", fill_value))
    return ret
