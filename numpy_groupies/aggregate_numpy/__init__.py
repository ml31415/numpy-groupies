"""The numpy implementation of aggregate: the vectorized operations, one per
function, behind one dispatch."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.base import _aggregate_base
from numpy_groupies.aggregate_numpy.collect import blocks_to_object_array
from numpy_groupies.aggregate_numpy.steps import step_count, step_indices
from numpy_groupies.common import DEFAULT_FILL_VALUE, FuncName, aggregate_common_doc
from numpy_groupies.utils_np import AFillValue, AResult, AScalar, AValue, RavelOrder

__all__ = ["aggregate", "blocks_to_object_array", "step_count", "step_indices"]


def aggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: FuncName | Callable[..., AScalar] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    **kwargs: Any,
) -> AResult:
    return _aggregate_base(
        group_idx,
        a,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        func=func,
        axis=axis,
        **kwargs,
    )


aggregate.__doc__ = (
    """
    This is the pure numpy implementation of aggregate.
    """
    + aggregate_common_doc
)
