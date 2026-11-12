"""Gather aggregated results back onto the input positions."""

from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.typing import AResult


def unpack(group_idx: npt.NDArray[np.integer], ret: AResult | list[Any]) -> AResult:
    """Uncompress an aggregate to the size of group_idx, equivalent to
    ret[group_idx]."""
    if not isinstance(ret, np.ndarray):
        # the pure python result is a list, gathered like any other sequence
        ret = np.asarray(ret)
    # fancy indexing is unannotated in numpy, the result is the same dtype as ret
    gathered: AResult = ret[group_idx]
    return gathered
