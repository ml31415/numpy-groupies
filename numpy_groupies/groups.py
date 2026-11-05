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
    """By example:

        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]

    This is equivalent to something like this :

        hstack((arange(n_i) for n_i in n))

    This version seems quite a bit faster, at least for some possible inputs, and at any rate it
    encapsulates a task in a function.
    """
    if n.ndim != 1:
        raise ValueError("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1] + 1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return cast("npt.NDArray[np.intp]", np.cumsum(ret)[:-1])


def label_contiguous_1d(X: npt.NDArray[np.number]) -> npt.NDArray[np.intp]:
    """
    WARNING: API for this function is liable to change!!!

    By example:

        X =      [F T T F F T F F F T T T]
        result = [0 1 1 0 0 2 0 0 0 3 3 3]

    Or:
        X =      [0 3 3 0 0 5 5 5 1 1 0 2]
        result = [0 1 1 0 0 2 2 2 3 3 0 4]

    The ``0`` or ``False`` elements of ``X`` are labeled as ``0`` in the output. If ``X`` is a boolean
    array, each contiguous block of ``True`` is given an integer label, if ``X`` is not boolean, then
    each contiguous block of identical values is given an integer label. Integer labels are 1, 2, 3,
    ..... (i.e. start a 1 and increase by 1 for each block with no skipped numbers.)
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
    """
    See also ``relabel_groups_masked``.

    keep_group:  [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]
    ret:         [0 3 3 3 0 2 4 2 0 1 1 0 3 4 4]

    Description of above: unique groups in input was ``1,2,3,5``, i.e.
    ``4`` was missing, so group 5 was relabeled to be ``4``.
    Relabeling maintains order, just "compressing" the higher numbers
    to fill gaps.
    """

    keep_group = np.zeros(np.max(group_idx) + 1, dtype=bool)
    keep_group[0] = True
    keep_group[group_idx] = True
    return relabel_groups_masked(group_idx, keep_group)


def relabel_groups_masked(
    group_idx: npt.NDArray[np.integer], keep_group: npt.NDArray[np.bool_]
) -> npt.NDArray[np.intp]:
    """
    group_idx: [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]

                 0 1 2 3 4 5
    keep_group: [0 1 0 1 1 1]

    ret:       [0 2 2 2 0 0 4 0 0 1 1 0 2 4 4]

    Description of above in words: remove group 2, and relabel group 3,4, and 5 to be 2, 3 and 4
    respectively, in order to fill the gap.  Note that group 4 was never used in the input group_idx,
    but the user supplied mask said to keep group 4, so group 5 is only moved up by one place to fill
    the gap created by removing group 2.

    That is, the mask describes which groups to remove, the remaining groups are relabeled to remove the
    gaps created by the falsy elements in ``keep_group``. Note that ``keep_group[0]`` has no particular
    meaning because it refers to the zero group which cannot be "removed".

    ``keep_group`` should be bool and ``group_idx`` int. Values in ``group_idx`` can be any order.
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
    """The mean of integer values per group, exact where float64 would round.

    The ticks of datetime input add up beyond the +/-2**53 window float64
    represents exactly, and so can the sums of plain integers - instead of
    refusing the mean, the values are split into limbs, every limb
    accumulates in an exact sum through ``bincount``, and the totals of the
    groups are combined as python integers.  The division is exact and the
    remainder rounds half to even.

    The limbs are as wide as they can be: a 32-bit limb holds its sum
    inside the window float64 represents exactly as long as there are fewer
    than 2**21 values, and a 16-bit limb serves more values than that.  The
    wider the limb, the fewer of them there are - and every limb is one pass
    over the data.

    An integer mean of small magnitudes keeps coming back as float64 - the
    quotient is exact in float64 there, so its answer does not change - and a
    caller asking for an integer dtype gets the exact rounded quotient.
    """
    u = np.ascontiguousarray(a).view(np.uint64)
    # the width of the limbs: the sum of a limb has to stay inside the
    # window float64 represents exactly
    width = 32 if len(group_idx) * (2**32 - 1) <= EXACT_FLOAT64_INTEGER_LIMIT else 16
    mask = np.uint64((1 << width) - 1)
    shifts = tuple(range(0, 64, width))
    limb_sums = [
        np.bincount(
            group_idx,
            # the cast to float64 is explicit: numpy would cast the weights
            # itself, and the cast of the masked limbs - small numbers - is
            # far quicker than the cast of the whole bit patterns
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
        # signed values sum to the two's-complement limb sum of their bit
        # pattern plus one 2**64 per negative value - the pattern sums are
        # corrected by the count of the negatives of every group
        negatives = np.bincount(group_idx, weights=u >= np.uint64(2**63), minlength=size)
        total -= negatives.astype(np.int64).astype(object) << 64
    present = counts > 0
    quotients = np.zeros(size, dtype=object)
    remainders = np.zeros(size, dtype=object)
    quotients[present], remainders[present] = (
        np.floor_divide(total[present], counts[present].astype(object)),
        np.remainder(total[present], counts[present].astype(object)),
    )
    # the fill of the empty groups: the default of the dtype unless the caller
    # asked for a number - the nan of a float result and the sentinel of the
    # call both mean "default"
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
        # the float64 answer: the exact quotient and remainder of the python
        # integers, so the total itself can never round - the 64-bit mantissa
        # of the long double holds every int64 quotient exactly, and the two
        # roundings of the assembly stay within half an ulp of the exact
        # quotient
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

    A group without values takes the fill value - an empty sequence leaves the
    empty block of the group alone.  The array is built element by element:
    ``np.asanyarray`` of a list of blocks collapses it into a regular array
    whenever the groups happen to be of the same length - and the result would
    lose the sequences it is meant to hold (and the size the caller asked
    for).

    Both backends collect their groups into such an array, and the function
    lives here so that neither of them has to import the other.
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
