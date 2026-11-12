"""The handover of the numba backend to the numpy implementation.

Three calls cannot be served in compiled code and go to the numpy
implementation instead: the median of complex input (numba has no ordering to
partition it), the quantile of a method other than the linear one (numpy's own
interpolation), and the collecting function on float16 or object arrays (no
numba dtype).  The quantile of a temporal input is not part of it: the compiled
kernel holds all five methods of numpy.  The numpy implementation is the module
of another backend, imported in one place only - this backend's ``__init__``,
which hands the callables to the modules that need them.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.typing import AFillValue, AResult, AValue, RavelOrder


class NumpyFallback:
    """The calls of the numba backend that the numpy implementation answers.

    The implementation is set after import by the backend's ``__init__`` - the
    modules holding the kernels are imported before it gets there - and every
    call of it is a call of the numpy implementation of the package.
    """

    def __init__(self) -> None:
        self._aggregate: Callable[..., AResult] | None = None

    def set_implementation(self, aggregate: Callable[..., AResult]) -> None:
        """The entry of the numpy implementation."""
        self._aggregate = aggregate

    def aggregate(
        self,
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        func: str,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None,
        fill_value: AFillValue,
        order: RavelOrder,
        dtype: npt.DTypeLike | None,
        axis: int | None,
        **kwargs: Any,
    ) -> AResult:
        """The answer of the numpy implementation."""
        if self._aggregate is None:
            raise RuntimeError("the numpy implementation of the package is not set")
        return self._aggregate(
            group_idx,
            a,
            func,
            size=size,
            fill_value=fill_value,
            order=order,
            dtype=dtype,
            axis=axis,
            **kwargs,
        )

    def as_operation(self) -> Callable[..., AResult]:
        """The fallback of a kernel: the operation arguments first, then the
        kernel's own - the signature the kernel hands it over with."""

        def fallback(
            group_idx: npt.NDArray[np.integer],
            a: AValue,
            func: str,
            size: int | tuple[int, ...] | npt.NDArray[np.integer] | None,
            fill_value: AFillValue,
            order: RavelOrder,
            dtype: npt.DTypeLike | None,
            axis: int | None,
        ) -> AResult:
            return self.aggregate(group_idx, a, func, size, fill_value, order, dtype, axis)

        return fallback


# the one instance the modules of the backend take
FALLBACK = NumpyFallback()
