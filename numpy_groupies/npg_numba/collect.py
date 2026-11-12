"""Collecting the values of every group - the ``array`` function.

One sequence of values per group, returned as a reflected list of arrays (numba
has no object arrays); the python-level ``aggregate()`` assembles that list into
an object array, so it matches the numpy implementation's result exactly.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from numpy_groupies.groups import blocks_to_object_array
from numpy_groupies.npg_numba.compose import build_custom_kernel, output_size, rule_object
from numpy_groupies.npg_numba.fallback import FALLBACK
from numpy_groupies.npg_numba.jit import njit_stable
from numpy_groupies.npg_numba.steps import group_count, group_place, group_starts

# the dtypes the compiled driver can place: the other operations let numba
# refuse a dtype it cannot take (a clean NumbaNotImplementedError), while this
# one hands the values to the numpy implementation - the driver would refuse a
# float128 mid-frontend, with its own source in the traceback.  No float16:
# numba has none, and this operation hands the values out unchanged.
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
    """Whether the compiled driver can place this dtype.

    No float16 - the numpy implementation copies those values around unchanged.
    """
    return dtype in _PLACEABLE


def _array_driver_factory(nan_variant: bool) -> Callable[..., list[npt.NDArray[np.generic]]]:
    """The compiled driver of the collecting function.

    A group's values are placed in a contiguous block, in input order, so the
    sequence of a group is a slice of one array; handed out as a list of arrays
    - the reflected list numba holds where it cannot hold an object array.  An
    absent group holds an empty array: a custom fill value is applied by the
    post hook, which knows the groups that received no value.
    """
    # every operation builds its own copies of the shared helpers - one compiled
    # specialization must not reach into another (see compose.build_kernel)
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
    ) -> list[npt.NDArray[np.generic]]:
        # the fill value and the dtypes are of no use - the sequences carry the
        # dtype of the values
        size = output_size(group_idx, size_arg)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        values = np.empty(group_idx.shape[0], a.dtype)
        _place(group_idx, a, starts.copy(), values)
        return [values[s : s + c] for s, c in zip(starts, counts)]

    # the name carries the variant: numba mangles the symbol from module,
    # qualname and signature, so the plain and nan variants of one operation
    # must not share it (see compose.build_kernel)
    return njit_stable(driver, name=f"{'nan' if nan_variant else ''}array_driver")


array = build_custom_kernel(
    "array",
    driver_factory=_array_driver_factory,
    rule=rule_object,
    fill_default=0,
    nans=False,
    fallback=FALLBACK.as_operation(),
    # the placeable dtypes, and the assembly of the result into the caller's
    # object array
    placeable=_placeable,
    post=blocks_to_object_array,
    assembles=True,
    # an absent group holds an empty sequence - the kernel says so instead of
    # ignoring the fill argument
    reject_fill=True,
    doc="The values of every group, one sequence per group",
)
