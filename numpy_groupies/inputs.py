"""The shapes of the answers and the validation of the input of a call:
lists, duck arrays and scalars are narrowed to the arrays the signatures
of the package name.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.dtypes import is_duck_array
from numpy_groupies.typing import AScalar, RavelOrder, ValidatedInput
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    GROUP_IDX_TOO_MANY_DIMS,
    LENGTH_MISMATCH_OR_SCALAR,
    NEGATIVE_INDICES,
    NON_INTEGER_GROUP_IDX,
    OUTPUT_SIZE_NOT_SCALAR,
    TOO_LARGE_FOR_SIZE,
    is_integer_dtype,
)


def check_nton_shape(
    ret: npt.NDArray[np.generic],
    size: int | tuple[int, ...] | npt.NDArray[np.integer],
    func: str | Callable[..., AScalar],
) -> None:
    """Complain early when a one-out-per-in function would have to fill a hole
    (``sort``, ``cumulative``) - otherwise numpy reports a plain reshape error.
    """
    # np.prod takes an int as it is, a tuple or an array of ints
    prod = int(np.prod(size))
    if ret.size != prod:
        name = getattr(func, "__name__", func)
        raise ValueError(
            f"'{name}' gives one value per input item and cannot fill the "
            f"{prod - ret.size} absent group entries of size {_size_tuple(size)}"
        )


def _size_tuple(size: int | tuple[int, ...] | npt.NDArray[np.integer]) -> tuple[int, ...]:
    """The size as a tuple of its dimensions - an int is one dimension."""
    if isinstance(size, tuple):
        return size
    if isinstance(size, int):
        return (size,)
    return tuple(int(s) for s in size)


def _ravel_group_idx(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    axis: int,
    size: Any,
    order: RavelOrder,
    method: str = "ravel",
) -> tuple[Any, Any]:
    ndim_a = a.ndim
    # broadcast-ready multidimensional indexing - purely a convenience
    size_in = int(np.max(group_idx)) + 1 if size is None else size
    group_idx_in = group_idx
    # the labels are collected per input dimension before ravel_multi_index folds them
    labels: list[npt.NDArray[np.integer]] = []
    size = []
    ravel = method == "ravel"
    for ii, s in enumerate(a.shape):
        if ravel:
            if ii == axis:
                ii_idx = group_idx_in
                if ii_idx.ndim == 1:
                    # 1d labels only carry the axis dimension - reshape so
                    # that ravel_multi_index broadcasts them against the
                    # arange-pieces of the other dimensions
                    ii_shape = [1] * ndim_a
                    ii_shape[ii] = s
                    ii_idx = ii_idx.reshape(ii_shape)
            else:
                ii_shape = [1] * ndim_a
                ii_shape[ii] = s
                ii_idx = np.arange(s).reshape(ii_shape)
            labels.append(ii_idx)
        size.append(size_in if ii == axis else s)
    # ravel or offset the labels, and return
    if ravel:
        group_idx = np.ravel_multi_index(labels, size, order=order, mode="raise")
    elif method == "offset":
        group_idx = offset_labels(group_idx_in, a.shape, axis, order, size_in)
    return group_idx, size


def offset_labels(
    group_idx: npt.NDArray[np.integer], inshape: tuple[int, ...], axis: int, order: RavelOrder, size: int
) -> npt.NDArray[np.integer]:
    """Offset group labels by dimension, for reductions over a subset of the
    dimensions of group_idx (reduction dimensions flattened into the last one).
    From https://stackoverflow.com/questions/46256279
    """

    group_idx = np.broadcast_to(group_idx, inshape)
    if axis not in (-1, len(inshape) - 1):
        group_idx = np.moveaxis(group_idx, axis, -1)
    newshape = group_idx.shape[:-1] + (-1,)

    group_idx = group_idx + np.arange(np.prod(newshape[:-1]), dtype=int).reshape(newshape) * size
    if axis not in (-1, len(inshape) - 1):
        return np.moveaxis(group_idx, -1, axis)
    return group_idx


def input_validation(
    # the unvalidated user input, narrowed down by this function
    group_idx: Any,
    a: Any,
    size: Any = None,
    order: RavelOrder = "C",
    axis: int | None = None,
    ravel_group_idx: bool = True,
    check_bounds: bool = True,
    func_name: str | None = None,
) -> ValidatedInput:
    """Extensive checking of group_idx and a; convert ndim-indexing to 1d.

    ``func_name`` is the resolved name of the function - ``arg``-reductions
    and the one-out-per-item answers (``cumulative``, ``sort``) are special-cased;
    no name (a custom callable) is neither.
    """
    if not isinstance(a, (int, float, complex)) and not is_duck_array(a):
        a = np.asanyarray(a)

    if len(group_idx) == 0:
        raise ValueError(EMPTY_GROUP_IDX)
    if not is_duck_array(group_idx):
        group_idx = np.asanyarray(group_idx)

    ndim_idx = group_idx.ndim
    ndim_a = getattr(a, "ndim", 0)

    # the two shape questions of a call without an axis come before the questions
    # of what the two arguments are made of, and in the same order in every
    # backend: the values are one-dimensional, and the labels reach at most one
    # row per dimension of the answer - a row per dimension is as deep as labels
    # ever go, and a dimension beyond that names no group
    if axis is None:
        if ndim_a > 1:
            raise ValueError("a must be scalar or 1 dimensional, use .ravel to flatten. Alternatively specify axis.")
        if ndim_idx > 2:
            raise ValueError(GROUP_IDX_TOO_MANY_DIMS)

    # equivalent to np.issubdtype(group_idx.dtype, np.integer), just cheaper
    if not is_integer_dtype(group_idx.dtype):
        raise TypeError(NON_INTEGER_GROUP_IDX)

    # This check works for multidimensional indexing as well
    if check_bounds and np.any(group_idx < 0):
        raise ValueError(NEGATIVE_INDICES)

    # axis arg: turn 1d indexing into multidimensional indexing along the axis
    if axis is not None:
        if axis >= ndim_a or axis < -ndim_a:
            raise ValueError("axis arg too large for np.ndim(a)")
        axis = axis if axis >= 0 else ndim_a + axis  # negative indexing
        if ndim_idx > 1:
            # multidimensional group labels, e.g. per row - broadcast across
            # the non-axis dimensions of a (issue #74)
            if ndim_idx != ndim_a:
                raise ValueError("when using axis arg, group_idx must be 1d, or of the same dimensionality as a")
            try:
                group_idx = np.broadcast_to(group_idx, a.shape)
            except ValueError as err:
                raise ValueError(
                    f"group_idx with shape {group_idx.shape} cannot be broadcast to a with shape {a.shape}"
                ) from err
            ndim_idx = 1
        if group_idx.ndim == 1 and a.shape[axis] != group_idx.shape[0]:
            raise ValueError("a.shape[axis] doesn't match length of group_idx.")
        if size is not None and not np.isscalar(size):
            raise NotImplementedError("when using axis arg, size must be None or scalar.")
        is_form_3 = ndim_a > 1
        orig_shape = a.shape if is_form_3 else group_idx.shape
        unravel_shape = orig_shape if isinstance(func_name, str) and "arg" in func_name else None

        method = "offset" if axis == ndim_a - 1 and order == "C" else "ravel"
        group_idx, size = _ravel_group_idx(group_idx, a, axis, size, order, method=method)
        flat_size = np.prod(size)
        ndim_idx = ndim_a
        one_to_one = is_form_3 and isinstance(func_name, str) and ("cum" in func_name or func_name == "sort")
        size = orig_shape if one_to_one else size
        ravel_order = order if one_to_one else "C"
        return (
            group_idx.ravel(order=ravel_order),
            a.ravel(order=ravel_order),
            flat_size,
            ndim_idx,
            size,
            unravel_shape,
        )

    if ndim_idx == 1:
        if size is None:
            size = int(np.max(group_idx)) + 1
        else:
            if not np.isscalar(size):
                raise ValueError(OUTPUT_SIZE_NOT_SCALAR)
            # the comparison below needs a number
            scalar_size = cast("int | float | np.number", size)
            if check_bounds and np.any(group_idx > scalar_size - 1):
                raise ValueError(TOO_LARGE_FOR_SIZE.format(size=scalar_size))
        flat_size = size
    else:
        if size is None:
            size = np.max(group_idx, axis=1).astype(int) + 1
        else:
            if np.isscalar(size):
                raise ValueError(f"output size must be of length {len(group_idx)}")
            if len(size) != len(group_idx):
                raise ValueError(f"{len(size)} sizes given, but {len(group_idx)} output dimensions specified in index")
        if ravel_group_idx:
            group_idx = np.ravel_multi_index(group_idx, size, order=order, mode="raise")
        flat_size = np.prod(size)

    if not (ndim_a == 0 or len(a) == group_idx.size):
        raise ValueError(LENGTH_MISMATCH_OR_SCALAR)

    return (group_idx, a, flat_size, ndim_idx, size, None)
