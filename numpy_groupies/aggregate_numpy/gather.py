"""Gather aggregated results back onto the input positions."""

from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.utils_np import AResult


def unpack(group_idx: npt.NDArray[np.integer], ret: AResult | list[Any]) -> AResult:
    """
    Take an aggregate packed array and uncompress it to the size of group_idx. This is equivalent to
    ret[group_idx].
    """
    if not isinstance(ret, np.ndarray):
        # the result of the pure python implementation is a list - it is
        # gathered like any other sequence of values
        ret = np.asarray(ret)
    # fancy indexing is unannotated in numpy, the result is the same dtype as ret
    gathered: AResult = ret[group_idx]
    return gathered
