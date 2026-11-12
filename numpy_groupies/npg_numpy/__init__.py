"""The numpy implementation of aggregate: the vectorized operations, one per
function, behind one dispatch shared with the other numpy based backend."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.driver import _aggregate_wrap
from numpy_groupies.fills import DEFAULT_FILL_VALUE
from numpy_groupies.functions import (
    aggregate_common_doc,
    bind_functions,
)
from numpy_groupies.npg_numpy.collect import blocks_to_object_array
from numpy_groupies.npg_numpy.gather import unpack
from numpy_groupies.npg_numpy.ops import _dispatch
from numpy_groupies.npg_numpy.steps import step_count, step_indices
from numpy_groupies.registry import (
    FuncName,
    function_names,
)
from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder

__all__ = ["aggregate", "blocks_to_object_array", "step_count", "step_indices", "unpack"]


def aggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: FuncName | Callable[..., AScalar] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    out: AResult | None = None,
    **kwargs: Any,
) -> AResult:
    return _aggregate_wrap(
        _dispatch,
        group_idx,
        a,
        func,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        axis=axis,
        out=out,
        **kwargs,
    )


aggregate.__doc__ = (
    """
    This is the pure numpy implementation of aggregate.
    """
    + aggregate_common_doc
)


# ---------------------------------------------------------------------------
# the functions of the backend, by their name
#
# one entry point per function of the package, calling the implementation
# above with the name of its function

_functions = bind_functions(aggregate, function_names())


def __getattr__(name: str) -> Any:
    """The function of the backend, by its name."""
    try:
        return _functions[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    """The names of the functions of the backend, among the module's own."""
    return sorted(set(globals()) | set(_functions))
