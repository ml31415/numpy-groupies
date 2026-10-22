"""Gather aggregated results back onto the input positions."""

from typing import Any

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.utils_np import AResult


# the annotations of the jitted function below are documentation only - numba
# compiles from the argument types and ignores them, the type checker needs them
@nb.njit(cache=True)
def _gather_1d(
    ret: npt.NDArray[np.generic], group_idx: npt.NDArray[np.integer], out: npt.NDArray[np.generic]
) -> npt.NDArray[np.generic]:
    for i in range(group_idx.size):
        out[i] = ret[group_idx[i]]
    return out


def unpack(group_idx: npt.NDArray[np.integer], ret: AResult | list[Any]) -> AResult:
    """The gather of a result: the result of the group of every value.

    The jitted loop gathers into an array of its own; ``unpack_into`` gathers
    into one the caller provides.
    """
    if not isinstance(ret, np.ndarray):
        # the result of the pure python implementation is a list - it is
        # gathered like any other sequence of values
        ret = np.asarray(ret)
    group_idx = np.asanyarray(group_idx)
    out = np.empty(group_idx.shape + ret.shape[1:], dtype=ret.dtype)
    return unpack_into(group_idx, ret, out)


def unpack_into(
    group_idx: npt.NDArray[np.integer], ret: npt.NDArray[np.generic], out: npt.NDArray[np.generic]
) -> npt.NDArray[np.generic]:
    """Like ``unpack``, but gathers into a caller-provided array.

    ``out`` must match the shape of ``ret[group_idx]`` and the dtype of
    ``ret``.  Contiguous 1d inputs are gathered by a jitted loop (noticeably
    faster than fancy indexing at large sizes), anything else falls back to
    fancy indexing assignment.
    """
    group_idx = np.asanyarray(group_idx)
    shape = group_idx.shape + ret.shape[1:]
    if out.shape != shape:
        raise ValueError(f"out with shape {out.shape} does not match the expected shape {shape}")
    if out.dtype != ret.dtype:
        raise TypeError(f"out dtype {out.dtype} does not match the result dtype {ret.dtype}")
    if group_idx.ndim == 1 and ret.ndim == 1 and group_idx.flags.c_contiguous and out.flags.c_contiguous:
        return _gather_1d(ret, group_idx, out)
    out[...] = ret[group_idx]
    return out
