"""The handover of the numba backend to the numpy implementation.

Three operations of the numba backend cannot do what the caller asks of
them in compiled code, and they hand the call over to the numpy
implementation instead:

- the median of a complex input cannot be partitioned - numba has no
  ordering for complex values,
- the quantile of a method other than the linear one is an interpolation
  that numpy does in its own way,
- the collecting function cannot place a float16 or an object array, which
  numba has no dtype for.

The quantile of a temporal input is no part of it: the compiled kernel of
this backend holds all five methods of numpy.

The numpy implementation is the module of another backend, and the numba
backend imports it in one place only - its ``__init__`` - which hands the
callables to the modules that need them.  That keeps the dependency of the
backend on the other one in a single spot, where it can be seen and where
it can go away with the operations that need it.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.typing import AFillValue, AResult, AValue, RavelOrder


class NumpyFallback:
    """The calls of the numba backend that the numpy implementation answers.

    The instance is the one of this module - the modules that need it take
    it from here - and the ``__init__`` of the numba backend hands it the
    numpy implementation, which it imports.  Every call of it is a call of
    the numpy implementation of the package.

    The implementation is set after the fact and not at the build of the
    object, because the modules that hold the kernels are imported before
    the ``__init__`` gets to the numpy implementation - and a call of the
    fallback happens only at the call of a kernel, long after that.
    """

    def __init__(self) -> None:
        self._aggregate: Callable[..., AResult] | None = None

    def set_implementation(self, aggregate: Callable[..., AResult]) -> None:
        """The entry of the numpy implementation.

        The quantile of a temporal input is no part of it: the compiled
        kernel of this backend holds all five methods of numpy.
        """
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
        """The fallback of a kernel: the same call behind the signature the
        kernel hands it over with - the arguments of the operation come
        first, then the ones of the kernel.
        """

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
