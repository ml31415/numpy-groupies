"""The types the implementations of the package speak in.

The aliases of this module name the values that travel through the
dispatchers: the input of an aggregation, its result, the scalars and the
fill values, and the memory layout the output is written in.

They live here and not next to the code that uses them, so that the type of
a value is named in one place - and so that a module that only needs the
type does not have to pull in the whole of numpy.

The module is called ``typing``, and it does not shadow the module of the
standard library of the same name: the imports of the package are absolute,
so ``from typing import ...`` inside this file reaches the standard library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from numpy_groupies.registry import DefaultFillValue

# the values as the dispatchers carry them on: an array of any dtype, or
# the one python scalar that stands for that value repeated over every label
AValue: TypeAlias = npt.NDArray[np.generic] | int | float | complex | np.bool_

# the result of an aggregation: numbers for most functions, bools for
# 'all' and 'any', objects for 'array' - np.generic covers all three
AResult: TypeAlias = npt.NDArray[np.generic]

# the numbers and flags that can land in an array - a bool is an int, a
# np.bool_ is neither an int nor an np.number
AScalar: TypeAlias = float | int | complex | np.number | np.bool_

# what a fill value can be: a number, the sentinel that asks for the
# default of the function, a None or - for the object result - a sequence
AFillValue: TypeAlias = AScalar | None | Sequence[AScalar] | DefaultFillValue

# the memory layout the output is written in, as every signature of this
# package spells it: the docstring offers 'F' for fortran-style and the code
# only ever ravels and reshapes with these two
RavelOrder: TypeAlias = Literal["C", "F"]

# what input_validation hands back: (raveled integer indices, the values -
# possibly still a python scalar -, the flat output size, the input index
# dimensionality, the (possibly shaped) output size and the shape an arg
# reduction has to unravel its positions with, in that order
ValidatedInput: TypeAlias = tuple[
    npt.NDArray[np.integer],
    AValue,
    int,
    int,
    int | tuple[int, ...] | npt.NDArray[np.integer],
    tuple[int, ...] | None,
]
