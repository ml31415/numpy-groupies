"""Gather aggregated results back onto the input positions."""

import numpy as np
import numpy.typing as npt


def _gather_1d(
    ret: npt.NDArray[np.number], group_idx: npt.NDArray[np.integer], out: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    for i in range(group_idx.size):
        out[i] = ret[group_idx[i]]
    return out


def unpack_into(
    group_idx: npt.NDArray[np.integer], ret: npt.NDArray[np.number], out: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
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
