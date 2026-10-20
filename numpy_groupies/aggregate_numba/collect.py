"""Collecting the values of every group - the ``array`` function.

``array`` is the one function of this backend whose result is no plain
number per group: it hands out the values of every group, one sequence per
group.  A compiled driver cannot produce an array of sequences - numba has no
object arrays - but it can produce a *list* of them, which is what the driver
of this module returns: a reflected list of the arrays of the groups.

The python-level ``aggregate()`` builds its object array from that list, the
same way the numpy implementation builds it from the blocks of its own sort,
so that the two hand out the very same object.
"""

import typing

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_common import DEFAULT_FILL_VALUE
from numpy_groupies.aggregate_numba.compose import KernelHandle, build_custom_kernel, output_size, rule_object
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numpy import _aggregate_base
from numpy_groupies.utils import (
    AValue,
    RavelOrder,
    check_dtype,
    check_fill_value,
    check_nton_shape,
    input_validation,
    is_duck_array,
    resolve_fill_value,
)

# the dtypes the compiled driver can place - the allowlist is exactly what
# the drivers of the other operations take: numba boxes a float16 at runtime
# and refuses a float128 at typing time, and strings, objects and the rest it
# does not know at all - those are copied around by the numpy implementation,
# which does not have to know their type
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


def _can_place(a: AValue) -> bool:
    """Whether the compiled driver can place the values of this dtype.

    Numba has no float16 - the numpy implementation copies the values of that
    dtype around, which keeps them exactly as they are.
    """
    return getattr(a, "dtype", None) in _PLACEABLE


def _array_driver_factory(nan_variant: bool) -> typing.Any:
    """The compiled driver of the collecting function.

    The values of a group are placed in a contiguous block, in the order they
    appear in the input, so that the sequence of a group is a slice of one
    array.  The blocks are handed out as a list of arrays - the reflected list
    numba can hold where it cannot hold an object array.

    A group without values holds an empty array: the fill value of the
    function is an empty sequence, and a custom one (a scalar or ``None``) is
    applied by the python-level wrapper, which knows the groups that received
    no value.
    """

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        ddof: int,
    ) -> typing.Any:
        # the fill value is of no use here, and the dtypes neither - the
        # sequences carry the dtype of the values
        size = output_size(group_idx, size_arg)
        n = group_idx.shape[0]
        counts = np.zeros(size, np.int64)
        for i in range(n):
            g = group_idx[i]
            if g < 0:
                raise ValueError("negative indices not supported")
            counts[g] += 1
        starts = np.cumsum(counts) - counts
        cursor = starts.copy()
        values = np.empty(n, a.dtype)
        for i in range(n):
            g = group_idx[i]
            values[cursor[g]] = a[i]
            cursor[g] += 1
        return [values[starts[g] : starts[g] + counts[g]] for g in range(size)]

    return njit_stable(driver, name="array_driver")


def _numpy_fallback(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: str,
    size: int | tuple[int, ...] | npt.NDArray[np.integer] | None,
    fill_value: typing.Any,
    order: RavelOrder,
    dtype: npt.DTypeLike | None,
    axis: int | None,
) -> npt.NDArray[np.object_]:
    """The numpy implementation, for the dtypes the compiled driver cannot place.

    It copies the values around without knowing their type, which the
    compiled driver cannot do - numba has no float16 and no object arrays.
    """
    return typing.cast(
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
    # the driver hands out the values of the groups and cannot place a fill
    # value in an absent one - the njit-callable stub says so instead of
    # ignoring the argument
    reject_fill=True,
    doc="The values of every group, one sequence per group",
)


def build_collecting_op(func: str, handle: KernelHandle) -> typing.Callable[..., typing.Any]:
    """The python-level wrapper of the collecting function.

    The result of the function is a sequence per group, which the shared
    wrapper of the other operations cannot hold: the driver hands out a list
    of the arrays of the groups, and the wrapper collects them in an object
    array - the same object the numpy implementation hands out.  A group
    without values takes the fill value, which the driver knows nothing about.
    """
    fallback = handle.fallback
    if fallback is None:
        # the dtypes the driver cannot place are the numpy implementation's
        # business - without it there is nothing to hand them to
        raise ValueError("the collecting function needs a fallback for the dtypes its driver cannot place")

    def aggregate(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None = None,
        fill_value: typing.Any = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        ddof: int = 0,
    ) -> npt.NDArray[np.object_]:
        # the values as an array, so that the dtype below can be asked for -
        # the validation would do the same, but it comes after the decision
        # whether the compiled driver can place the values at all
        if not isinstance(a, (int, float, complex)) and not is_duck_array(a):
            a = typing.cast("AValue", np.asanyarray(a))
        if not _can_place(a):
            # a dtype the compiled driver cannot place - the numpy
            # implementation copies the values without knowing their type
            return typing.cast(
                "npt.NDArray[np.object_]",
                fallback(group_idx, a, func, size, fill_value, order, dtype, axis),
            )

        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False, func=func)
        group_idx, a, flat_size, ndim_idx, size, _ = iv

        # the dtype of the result is the object one - the sequences of the
        # groups are of the dtype of the values, and the array holding them is
        # of none
        out_dtype = check_dtype(dtype, func, a, len(group_idx))
        fill_value = resolve_fill_value(func, fill_value, out_dtype)
        check_fill_value(fill_value, out_dtype, func=func)
        if fill_value is not None and not (np.isscalar(fill_value) or len(fill_value) == 0):
            raise ValueError("fill_value must be None, a scalar or an empty sequence")

        # the sequences of the groups, in the order the groups are numbered -
        # the same construction the numpy implementation uses, so that the two
        # hand out the very same object
        blocks = handle.driver(group_idx, a, flat_size, out_dtype, out_dtype, 0.0, ddof)
        ret = np.empty(len(blocks), dtype=object)
        for g, block in enumerate(blocks):
            ret[g] = block

        if fill_value is None or np.isscalar(fill_value):
            # a group without values takes the fill value - an empty sequence
            # leaves the empty block of the group alone
            untouched = np.ones_like(ret, dtype=bool)
            untouched[group_idx] = False
            ret[untouched] = fill_value

        if ndim_idx > 1:
            # the labels named more than one dimension - the flat result is
            # spread over them again
            check_nton_shape(ret, size, func)
            ret = ret.reshape(size, order=order)
        return ret

    return aggregate
