"""The fill values of the package: the sentinel that asks for the default of
a function, the defaults themselves and the resolution of a fill value.

The value an absent group gets depends on the function and on the dtype of
the answer, and a call asks for it with the sentinel
``DEFAULT_FILL_VALUE`` - the way of asking, next to the resolution of what
the answer is.  The type of the marker is named by the types of the
package (``numpy_groupies.typing``), the class and its one instance live
here.

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
    """The type of the marker that asks for the default of a fill value.

    The registry knows the default of every function of the package, and
    the marker of this type is the way of asking for it: a call that carries
    it is answered with the default of its function, resolved against the
    dtype of its answer (see ``resolve_fill_value``).
    """

    def __repr__(self) -> str:
        return "DEFAULT_FILL_VALUE"


DEFAULT_FILL_VALUE = DefaultFillMarker()


def nan_of(dtype: npt.DTypeLike | None) -> AScalar:
    """A nan shaped like ``dtype``: a nan in every part it has.

    A plain ``nan`` written into a complex result lands as ``nan+0j``, which
    states that the part the nan never touched was a zero - and a group that was
    poisoned by one of its values never stated anything about its parts at all.
    The default of a function for a complex result takes this shape, and so does
    the poisoning of a group, for the same reason.
    """
    return nan + nan * 1j if is_complex_dtype(dtype) else nan


def default_fill_value(func: FuncName | Callable[..., AScalar], dtype: npt.DTypeLike | None = None) -> AFillValue:
    """The value ``aggregate`` uses for groups missing from ``group_idx``.

    ``func`` may be a name, an alias or a callable.  Pass the ``dtype`` of your
    actual output to get the value that would really be used, ``None`` assumes
    a floating result (which is what the averaging functions give anyway).
    """
    return resolve_fill_value(func, DEFAULT_FILL_VALUE, float if dtype is None else dtype)


def resolve_fill_value(
    func: str | Callable[..., AScalar], fill_value: AFillValue, dtype: npt.DTypeLike | None
) -> AFillValue:
    """Replace the ``DEFAULT_FILL_VALUE`` sentinel with the default of ``func``.

    Anything else than the sentinel is returned untouched, except a ``None``: a
    ``None`` is refused where no result can hold it.  ``nan`` is only
    handed out if ``dtype`` can represent it, otherwise the value falls back to
    ``0`` - changing the output datatype of e.g. an integer ``min`` would be a
    worse surprise than an unspectacular filling value.
    """
    if fill_value is None:
        check_filling_with_none(fill_value_key(func))
    if fill_value is not DEFAULT_FILL_VALUE:
        return fill_value
    key = fill_value_key(func)
    # nan only fits where the output can hold it - the averaging and
    # dispersion functions coerce their result to a float type anyway, but an
    # explicitly requested integer output holds no nan, whatever the function
    # answers by default
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
        # a nan of a complex output has to be a nan in both parts - the numpy
        # kernels divide an empty sum by zero and so get nan+nanj for free,
        # and filling with nan+0j would make the backends disagree
        value = nan_of(dtype)
    return list(value) if isinstance(value, list) else value
