"""The types the implementations of the package speak in.

The aliases name the values that travel through the dispatchers in one
place, so modules that only need a type can import it under
``TYPE_CHECKING`` without pulling in numpy. The one value without a type of
its own - the default-fill marker - lives with the fill values.

Despite the name this module does not shadow the stdlib ``typing``: the
imports of the package are absolute.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from numpy_groupies.fills import DefaultFillMarker

# an array of any dtype, or the one python scalar standing for that value
# repeated over every label
AValue: TypeAlias = npt.NDArray[np.generic] | int | float | complex | np.bool_

# numbers mostly, bools for 'all'/'any', objects for 'array'
AResult: TypeAlias = npt.NDArray[np.generic]

# a bool is an int, a np.bool_ is neither an int nor an np.number
AScalar: TypeAlias = float | int | complex | np.number | np.bool_

# a number, the default-fill marker, None, or a sequence for object results
AFillValue: TypeAlias = "AScalar | None | Sequence[AScalar] | DefaultFillMarker"

# the memory layout of the output, as every signature spells it
RavelOrder: TypeAlias = Literal["C", "F"]

# (raveled integer indices, the values - possibly a python scalar -, the
# flat output size, the index dimensionality, the (possibly shaped) output
# size, and the shape an arg reduction unravels positions with)
ValidatedInput: TypeAlias = tuple[
    npt.NDArray[np.integer],
    AValue,
    int,
    int,
    int | tuple[int, ...] | npt.NDArray[np.integer],
    tuple[int, ...] | None,
]
