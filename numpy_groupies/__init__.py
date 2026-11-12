"""The public API of the package.

``aggregate`` is the best implementation available at import time (numba,
else numpy, else pure python); every function is also a name of its own
(``npg.mean(group_idx, a)``).

Implementations are reached in two ways, both kept:

- their modules ``npg_numpy``, ``npg_numba``, ``npg_purepy``, carrying the
  functions of the implementation as attributes; ``None`` where the needed
  package is not installed.
- the old names: ``aggregate_np``/``aggregate_nb``/``aggregate_py`` (the
  aggregates themselves, documented in the readme) and
  ``aggregate_numpy``/``aggregate_numba``/``aggregate_purepy`` (the modules
  under their old names).

Rule throughout: a name belonging to an implementation is ``None`` where that
implementation is not there; pure python is always there.  The package's own
functions behave like ``aggregate`` with the function name as ``func``;
numba's kernels live only as attributes of the numba module.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from numpy_groupies import npg_purepy
from numpy_groupies.cache import purge_cache
from numpy_groupies.functions import bind_functions
from numpy_groupies.registry import function_names

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    import numpy.typing as npt

    from numpy_groupies.registry import FuncName
    from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder

    class Aggregate(Protocol):
        """The canonical signature of the public ``aggregate`` (bound to the best
        implementation at import time).

        Differences to the most restrictive one: the pure python fallback does
        not take a sequence ``size``; only numba honours ``out``.
        """

        def __call__(
            self,
            group_idx: npt.NDArray[np.integer],
            a: AValue,
            func: FuncName | Callable[..., AScalar] = "sum",
            size: int | Sequence[int] | None = None,
            fill_value: AFillValue = ...,
            order: RavelOrder = "C",
            dtype: npt.DTypeLike | None = None,
            axis: int | None = None,
            out: npt.NDArray[np.number] | None = None,
            **kwargs: Any,
        ) -> AResult: ...

    class EntryPoint(Protocol):
        """Entry point of one function of the package: the ``aggregate`` arguments
        plus the function's own ones (``ddof``, ``q``, ``dx``, ``reverse``),
        passed between ``a`` and ``size`` at runtime and not spelled out here.
        """

        def __call__(
            self,
            group_idx: npt.NDArray[np.integer],
            a: AValue,
            *args: Any,
            size: int | Sequence[int] | None = None,
            fill_value: AFillValue = ...,
            order: RavelOrder = "C",
            dtype: npt.DTypeLike | None = None,
            axis: int | None = None,
            **kwargs: Any,
        ) -> AResult: ...

    # the gather of a result: numpy and numba gather into an array, pure
    # python (without numpy) into a list
    def unpack(group_idx: npt.NDArray[np.integer], ret: AResult | list[Any]) -> AResult | list[Any]: ...

    # tools needing a package the caller may not have: None where absent
    unpack_into: Callable[[npt.NDArray[np.integer], AResult, AResult], AResult] | None
    step_count: Callable[[npt.NDArray[np.integer]], int] | None
    step_indices: Callable[[npt.NDArray[np.integer]], npt.NDArray[np.intp]] | None
    multi_arange: Callable[[npt.NDArray[np.integer]], npt.NDArray[np.intp]] | None
    label_contiguous_1d: Callable[[npt.NDArray[np.number]], npt.NDArray[np.intp]] | None
    relabel_groups_unique: Callable[[npt.NDArray[np.integer]], npt.NDArray[np.intp]] | None
    relabel_groups_masked: Callable[[npt.NDArray[np.integer], npt.NDArray[np.bool_]], npt.NDArray[np.intp]] | None
    default_fill_value: Callable[..., AFillValue] | None


# ---------------------------------------------------------------------------
# the backends
#
# A backend is the module of its implementation: aggregate and per-function
# entry points as attributes (the numba kernels too, which jitted code can
# reach; a plain python object is no numba type, a module attribute is).

# implementation modules - None where the needed package is not installed
npg_numpy: ModuleType | None
npg_numba: ModuleType | None

# implementation aggregates - None where the implementation is absent
aggregate: Aggregate
aggregate_np: Aggregate | None
aggregate_nb: Aggregate | None
# pure python is always there and keeps its own type: it answers with a list
# where the numpy based backends answer with an array
aggregate_py = npg_purepy.aggregate
aggregate_purepy = npg_purepy

# old module name kept in sys.modules, so the import path stays
sys.modules["numpy_groupies.aggregate_purepy"] = npg_purepy

try:
    import numpy as np
except ImportError:
    # no numpy, no numpy implementation: none of its names
    npg_numpy = None
    aggregate_np = None
    aggregate_numpy = None
    multi_arange = label_contiguous_1d = None
    relabel_groups_masked = relabel_groups_unique = None
    default_fill_value = None
    step_count = step_indices = None

    # pure python is the only backend: it answers with a list and gathers a list
    aggregate = cast("Aggregate", aggregate_py)
    from numpy_groupies.npg_purepy import unpack
else:
    # importing the module also sets it as the package attribute
    from numpy_groupies import npg_numpy
    from numpy_groupies.npg_numpy import aggregate, step_count, step_indices, unpack

    aggregate_np = aggregate
    aggregate_numpy = npg_numpy
    if npg_numpy is not None:
        sys.modules["numpy_groupies.aggregate_numpy"] = npg_numpy

    from numpy_groupies.fills import default_fill_value
    from numpy_groupies.groups import (
        label_contiguous_1d,
        multi_arange,
        relabel_groups_masked,
        relabel_groups_unique,
    )

try:
    import numba
except ImportError:
    # no numba, no numba backend (the benchmarks ask for the old name first)
    npg_numba = None
    aggregate_nb = None
    aggregate_numba = None
    unpack_into = None
else:
    # importing the module also sets it as the package attribute
    from numpy_groupies import npg_numba
    from numpy_groupies.npg_numba import aggregate, step_count, step_indices, unpack, unpack_into

    # numba wins as the aggregate of the package
    aggregate_nb = aggregate
    aggregate_numba = npg_numba
    if npg_numba is not None:
        sys.modules["numpy_groupies.aggregate_numba"] = npg_numba


def uaggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    out: AResult | None = None,
    **kwargs: Any,
) -> AResult | list[Any]:
    """
    Aggregate the values of ``a`` by the groups in ``group_idx`` and broadcast the result
    back to the size of ``a``, so that every element carries the result of its own group.
    This is equivalent to ``unpack(group_idx, aggregate(group_idx, a, ...))``, i.e.
    ``aggregate(...)[group_idx]``, and takes the same arguments as ``aggregate``.

    By example:

        group_idx = [3, 0, 0, 1, 0, 3, 5, 5, 0, 4]
        a         = [13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7]
        ret       = [13.3, 2.5, 2.5, -8.2, 2.5, 13.3, 46.05, 46.05,  2.5 , 53.7]

    with ``func='mean'``. A typical use is demeaning within each group, as in
    ``a - uaggregate(group_idx, a, func='mean')``.

    Pass ``out`` to gather into an array provided by the caller, which must match the
    shape of ``ret[group_idx]`` and the dtype of ``ret``. Only the numba implementation
    supports this, the others raise ``NotImplementedError``.

    See also ``aggregate``, ``unpack``, ``unpack_into``.
    """
    ret = aggregate(group_idx, a, **kwargs)
    if out is None:
        return unpack(group_idx, ret)
    if unpack_into is None or aggregate_nb is None or aggregate is not aggregate_nb:
        raise NotImplementedError("out= is only supported by the numba implementation")
    return unpack_into(group_idx, ret, out)


# ---------------------------------------------------------------------------
# the functions of the package
#
# One entry point per function, with the function in the name; built at
# runtime from the entry point table so they cannot drift.  Declared for the
# type checker below; ``test_registry.py`` keeps the two in step.

_functions = bind_functions(aggregate, function_names())
globals().update(_functions)

if TYPE_CHECKING:
    sum: EntryPoint
    prod: EntryPoint
    first: EntryPoint
    last: EntryPoint
    len: EntryPoint
    nunique: EntryPoint
    mean: EntryPoint
    median: EntryPoint
    quantile: EntryPoint
    percentile: EntryPoint
    trapezoid: EntryPoint
    var: EntryPoint
    std: EntryPoint
    skew: EntryPoint
    kurt: EntryPoint
    all: EntryPoint
    any: EntryPoint
    allnan: EntryPoint
    anynan: EntryPoint
    max: EntryPoint
    min: EntryPoint
    argmax: EntryPoint
    argmin: EntryPoint
    sort: EntryPoint
    array: EntryPoint
    topk: EntryPoint
    mode: EntryPoint
    sumofsquares: EntryPoint
    cumsum: EntryPoint
    cumprod: EntryPoint
    cummax: EntryPoint
    cummin: EntryPoint
    nansum: EntryPoint
    nanprod: EntryPoint
    nanfirst: EntryPoint
    nanlast: EntryPoint
    nanlen: EntryPoint
    nannunique: EntryPoint
    nanmean: EntryPoint
    nanmedian: EntryPoint
    nanquantile: EntryPoint
    nanpercentile: EntryPoint
    nantrapezoid: EntryPoint
    nanvar: EntryPoint
    nanstd: EntryPoint
    nanskew: EntryPoint
    nankurt: EntryPoint
    nanall: EntryPoint
    nanany: EntryPoint
    nanmax: EntryPoint
    nanmin: EntryPoint
    nanargmax: EntryPoint
    nanargmin: EntryPoint
    nansumofsquares: EntryPoint
    nancumsum: EntryPoint
    nancumprod: EntryPoint
    nancummax: EntryPoint
    nancummin: EntryPoint


# ``purge_cache()`` removes the cached compiles of the numba backend's cache
# (see ``numpy_groupies.cache.numba_cache_dir``), which numba never cleans up.


try:
    # Version is added only when packaged
    from numpy_groupies._version import __version__
except ImportError:
    try:
        from setuptools_scm import get_version
    except ImportError:
        __version__ = "0.0.0"
    else:
        __version__ = get_version(root="..", relative_to=__file__)
        del get_version
