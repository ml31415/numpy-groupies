"""The numpy implementation of aggregate: the vectorized operations, one per
function, behind one dispatch."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.base import _aggregate_base
from numpy_groupies.aggregate_numpy.collect import blocks_to_object_array
from numpy_groupies.aggregate_numpy.gather import unpack
from numpy_groupies.aggregate_numpy.steps import step_count, step_indices
from numpy_groupies.common import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
)
from numpy_groupies.registry import (
    FuncName,
    bind_functions,
    function_names,
)
from numpy_groupies.utils_np import AFillValue, AResult, AScalar, AValue, RavelOrder

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
    # the argument the numba implementation honours and this one does not:
    # there is no compiled loop here to write into an array of the caller -
    # the argument is taken, so that the signature of every backend is the
    # same, and it says what it cannot do
    if out is not None:
        raise NotImplementedError("out= is only supported by the numba implementation")
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


# ---------------------------------------------------------------------------
# the functions of the backend, by their name
#
# One entry point per function of the package, calling the implementation
# above with the name of its function - the same thing the numba module
# offers with its kernels, and the same thing the package itself offers with
# the functions of the best implementation available.

_functions = bind_functions(aggregate, function_names())


def __getattr__(name: str) -> Any:
    """The function of the backend, by its name.

    The names of the functions are not set one by one - they are looked up
    in the table of the entry points, so that they cannot drift from it.
    """
    try:
        return _functions[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    """The names of the functions of the backend, among the module's own."""
    return sorted(set(globals()) | set(_functions))
