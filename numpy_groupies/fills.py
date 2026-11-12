"""The fill values of the package: the ``DEFAULT_FILL_VALUE`` sentinel, the
per-function defaults and the resolution of a fill value against a dtype.

This module must not import numpy: the pure python implementation is loaded
before numpy is known to be available, and has to work without it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy_groupies.functions import (
    default_fill_values,
    fill_value_key,
    forced_float_types,
    forced_real_float_types,
    nan,
)
from numpy_groupies.registry import FuncName
from numpy_groupies.validation import (
    check_filling_with_none,
    is_complex_dtype,
    is_inexact_dtype,
    is_integer_dtype,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from numpy_groupies.typing import AFillValue, AScalar


class DefaultFillMarker:
    """The type of the marker that asks for the default fill value of a
    function, resolved against the dtype of the answer (see
    ``resolve_fill_value``).
    """

    def __repr__(self) -> str:
        return "DEFAULT_FILL_VALUE"


DEFAULT_FILL_VALUE = DefaultFillMarker()


def nan_of(dtype: npt.DTypeLike | None) -> AScalar:
    """A nan shaped like ``dtype``: ``nan+nanj`` for complex, plain ``nan``
    otherwise - a poisoned group never stated anything about its parts, so
    ``nan+0j`` would be wrong.
    """
    return nan + nan * 1j if is_complex_dtype(dtype) else nan


def default_fill_value(func: FuncName | Callable[..., AScalar], dtype: npt.DTypeLike | None = None) -> AFillValue:
    """The value ``aggregate`` uses for groups missing from ``group_idx``.

    ``func`` may be a name, an alias or a callable; ``dtype=None`` assumes a
    floating result.
    """
    return resolve_fill_value(func, DEFAULT_FILL_VALUE, float if dtype is None else dtype)


def resolve_fill_value(
    func: str | Callable[..., AScalar], fill_value: AFillValue, dtype: npt.DTypeLike | None
) -> AFillValue:
    """Replace the ``DEFAULT_FILL_VALUE`` sentinel with the default of ``func``;
    anything else is returned untouched, except a ``None``, which is refused
    where no result can hold it. ``nan`` only fits where ``dtype`` can hold
    it, otherwise the value falls back to ``0``.
    """
    if fill_value is None:
        check_filling_with_none(fill_value_key(func))
    if fill_value is not DEFAULT_FILL_VALUE:
        return fill_value
    key = fill_value_key(func)
    # nan only fits where the output can hold it - an explicitly requested
    # integer output holds no nan, whatever the default is
    fits = not is_integer_dtype(dtype) and (
        dtype is None or key in forced_float_types or key in forced_real_float_types or is_inexact_dtype(dtype)
    )
    value = default_fill_values.get(key)
    if value is None:
        # a custom callable has no function specific default
        value = nan if fits else 0
    elif value != value and not fits:
        value = 0
    if (
        isinstance(value, float)
        and value != value
        and dtype is not None
        and is_complex_dtype(dtype)
        and key not in forced_real_float_types
    ):
        # a nan of a complex output has to be a nan in both parts, so the
        # backends agree with the numpy kernels (nan+nanj for an empty sum)
        value = nan_of(dtype)
    return list(value) if isinstance(value, list) else value
