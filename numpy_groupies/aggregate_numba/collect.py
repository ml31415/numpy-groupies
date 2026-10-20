"""Collecting the values of every group - the ``array`` function.

``array`` is the one function of this backend whose result is no plain
number per group: it hands out the values of every group, one sequence per
group.  A compiled driver cannot produce an array of sequences - numba has no
object arrays - but it can produce a *list* of them, which is what the driver
of this module returns: a reflected list of the arrays of the groups.

The python-level ``aggregate()`` collects that list in an object array, the
same way the numpy implementation collects the blocks of its own sort, so that
the two hand out the very same object.
"""

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_common import FuncName
from numpy_groupies.aggregate_numba.compose import build_custom_kernel, output_size, rule_object
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numba.steps import group_count, group_place, group_starts
from numpy_groupies.aggregate_numpy import _aggregate_base, blocks_to_object_array
from numpy_groupies.utils import AFillValue, AValue, RavelOrder

# the dtypes the compiled driver can place - the numeric ones the drivers of
# the other operations take as well.  The list is no duplication of what numba
# can type: the other operations let numba refuse a dtype it cannot take (a
# clean NumbaNotImplementedError), while this one has to hand the values over
# to the numpy implementation - and the driver would refuse a float128 in the
# middle of the frontend, with its own source in the traceback.  A float16 is
# out as well: numba has none, and this operation is the one that hands the
# values out unchanged.
_PLACEABLE = frozenset(
    map(
        np.dtype,
        (
            np.bool_,
            *(getattr(np, f"int{w}") for w in (8, 16, 32, 64)),
            *(getattr(np, f"uint{w}") for w in (8, 16, 32, 64)),
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ),
    )
)


def _placeable(dtype: np.dtype | None) -> bool:
    """Whether the compiled driver can place the values of this dtype.

    Numba has no float16 - the numpy implementation copies the values of that
    dtype around, which keeps them exactly as they are.
    """
    return dtype in _PLACEABLE


def _array_driver_factory(nan_variant: bool) -> Any:
    """The compiled driver of the collecting function.

    The values of a group are placed in a contiguous block, in the order they
    appear in the input, so that the sequence of a group is a slice of one
    array.  The blocks are handed out as a list of arrays - the reflected list
    numba can hold where it cannot hold an object array.

    A group without values holds an empty array: the fill value of the
    function is an empty sequence, and a custom one (a scalar or ``None``) is
    applied by the post hook, which knows the groups that received no value.
    """
    # every operation builds its own copies of the shared helpers - the
    # compiled specialization of one must not reach into the other (see
    # compose.build_kernel)
    _count = njit_stable(group_count)
    _starts = njit_stable(group_starts)
    _place = njit_stable(group_place)

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        ddof: int,
    ) -> list[npt.NDArray[np.generic]]:
        # the fill value is of no use here, and the dtypes neither - the
        # sequences carry the dtype of the values
        size = output_size(group_idx, size_arg)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        values = np.empty(group_idx.shape[0], a.dtype)
        _place(group_idx, a, starts.copy(), values)
        return [values[s : s + c] for s, c in zip(starts, counts)]

    # the name carries the variant: numba mangles the symbol of a compiled
    # function from module, qualname and signature, so the plain and the nan
    # variant of one operation must not share it (see compose.build_kernel)
    return njit_stable(driver, name=f"{'nan' if nan_variant else ''}array_driver")


def _numpy_fallback(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: FuncName,
    size: int | tuple[int, ...] | npt.NDArray[np.integer] | None,
    fill_value: AFillValue,
    order: RavelOrder,
    dtype: npt.DTypeLike | None,
    axis: int | None,
) -> npt.NDArray[np.object_]:
    """The numpy implementation, for the dtypes the compiled driver cannot place.

    It copies the values around without knowing their type, which the
    compiled driver cannot do - numba has no float16 and no object arrays.
    """
    return cast(
        "npt.NDArray[np.object_]",
        _aggregate_base(group_idx, a, func, size=size, fill_value=fill_value, order=order, dtype=dtype, axis=axis),
    )


array = build_custom_kernel(
    "array",
    driver_factory=_array_driver_factory,
    rule=rule_object,
    fill_default=0,
    nans=False,
    fallback=_numpy_fallback,
    # the dtypes the driver can take, and the assembly of its result into the
    # object array the caller expects
    placeable=_placeable,
    post=blocks_to_object_array,
    assembles=True,
    # the driver hands out the values of the groups and cannot place a fill
    # value in an absent one - the njit-callable stub says so instead of
    # ignoring the argument
    reject_fill=True,
    doc="The values of every group, one sequence per group",
)
