"""The public API of the package.

The package offers the aggregate of the best implementation available at
import time - numba, else numpy, else pure python - under the name
``aggregate``.  Every function of the package is a name of its own as well:
``npg.mean(group_idx, a)`` is the mean of every group, computed by the best
implementation there is.

The implementations are reached in two ways, and both of them are kept:

- the modules of the implementations: ``aggregate_numpy``, ``aggregate_numba``
  and ``aggregate_purepy`` are the modules the implementations live in, and
  they carry the functions of their implementation as their attributes -
  ``import numpy_groupies.aggregate_numpy as npg`` gives a ``npg`` whose
  ``aggregate`` is the aggregate of the numpy implementation and whose
  ``mean`` is the mean of the numpy implementation.  A module is ``None``
  where the package it needs is not installed.
- the old names, which are the aggregates themselves: ``aggregate_np`` is the
  aggregate of the numpy implementation, ``aggregate_nb`` the one of the numba
  implementation and ``aggregate_py`` the one of the pure python
  implementation.  An aggregate is ``None`` where its implementation is not
  there, exactly like its module.  They are the names the readme of the
  package documents.

The rule runs through the whole package: a name that belongs to an
implementation - its module, its aggregate, its tools - is ``None`` where that
implementation is not there.  The pure python implementation is always there,
and so is the gather of a result, which every implementation has.

The functions of the package itself are the ones of the best implementation
available, and they behave like ``aggregate`` with the name of the function as
the ``func`` argument: ``npg.mean(group_idx, a)`` is ``npg.aggregate(group_idx,
a, func="mean")``.  The kernels of the numba implementation - the lean
functions jitted code calls - are the attributes of the numba module, and are
not what the package hands out here.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from numpy_groupies import aggregate_purepy
from numpy_groupies.registry import bind_functions, function_names
from numpy_groupies.utils_cache import purge

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    import numpy.typing as npt

    from numpy_groupies.registry import FuncName
    from numpy_groupies.utils_np import AFillValue, AResult, AScalar, AValue, RavelOrder

    class Aggregate(Protocol):
        """The canonical signature of the public ``aggregate``.

        The public name is bound to the best implementation available at
        import time (numba, else numpy, else pure python) - they agree on
        everything but the most restrictive ``size`` of the pure python
        fallback, which does not take a sequence, and on the ``out`` keyword
        argument, which only the numba implementation honours (the others
        raise it).
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
        """The entry point of one function of the package.

        It takes the arguments of ``aggregate`` and the arguments of its own
        function - the ``ddof`` of the variance, the ``q`` of the quantile,
        the ``dx`` of the trapezoid, the ``reverse`` of the sort.  Those are
        the arguments between ``a`` and ``size`` at runtime, and they are not
        spelled out here: one entry point covers them all.
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

    # the gather of a result: the numpy and the numba backends gather into an
    # array, the pure python one - the one that runs without numpy - into a
    # list
    def unpack(group_idx: npt.NDArray[np.integer], ret: AResult | list[Any]) -> AResult | list[Any]: ...

    # the tools that need a package the caller may not have: they are None
    # where that package is not installed
    unpack_into: Callable[[npt.NDArray[np.integer], AResult, AResult], AResult] | None
    step_count: Callable[[npt.NDArray[np.integer]], int] | None
    step_indices: Callable[[npt.NDArray[np.integer]], npt.NDArray[np.intp]] | None
    multi_arange: Callable[[npt.NDArray[np.integer]], npt.NDArray[np.intp]] | None
    label_contiguous_1d: Callable[[npt.NDArray[np.number]], npt.NDArray[np.intp]] | None
    relabel_groups_unique: Callable[[npt.NDArray[np.integer]], npt.NDArray[np.intp]] | None
    relabel_groups_masked: Callable[[npt.NDArray[np.integer], npt.NDArray[np.bool_]], npt.NDArray[np.intp]] | None
    default_fill_value: Callable[..., AFillValue] | None


def dummy_no_impl(*args: Any, **kwargs: Any) -> Any:
    """The stand-in for an implementation that is not there.

    The package hands out ``None`` for the names of the implementations it
    does not have; this is the callable that says what is missing when it is
    called, for the caller who wants one.
    """
    raise NotImplementedError(
        "You may need to install another package (numpy or numba) to access a working implementation."
    )


# ---------------------------------------------------------------------------
# the backends
#
# A backend is the module of its implementation: it carries the aggregate of
# the implementation and the entry point of every function of the package as
# its attribute - ``aggregate_numpy.mean(group_idx, a)`` is the mean of every
# group, with the function in the name.  The numba module carries its
# kernels there, which jitted code can reach (a plain python object is no
# numba type, a module attribute is).

# the modules of the implementations - a module is None where the package it
# needs is not installed
aggregate_numpy: ModuleType | None
aggregate_numba: ModuleType | None

# the aggregates of the implementations - the aggregate of an implementation is
# None where the implementation is not there, exactly like its module
aggregate: Aggregate
aggregate_np: Aggregate | None
aggregate_nb: Aggregate | None
# the pure python one is always there, and it keeps its own type: it answers
# with a list where the numpy based backends answer with an array
aggregate_py = aggregate_purepy.aggregate

try:
    import numpy as np
except ImportError:
    # the numpy implementation needs numpy - without it there is none of it:
    # neither its module nor its aggregate nor its tools
    aggregate_numpy = None
    aggregate_np = None
    multi_arange = label_contiguous_1d = None
    relabel_groups_masked = relabel_groups_unique = None
    default_fill_value = None
    step_count = step_indices = None

    # the pure python backend is the only one there is: its aggregate answers
    # with a list, where the protocol of the public aggregate describes the
    # array of the numpy based backends, and its gather gathers a list
    aggregate = cast("Aggregate", aggregate_py)
    from numpy_groupies.aggregate_purepy import unpack
else:
    # the module of the implementation - importing it sets it as the
    # attribute of the package as well
    from numpy_groupies import aggregate_numpy
    from numpy_groupies.aggregate_numpy import aggregate, step_count, step_indices, unpack

    aggregate_np = aggregate

    from numpy_groupies.common import default_fill_value
    from numpy_groupies.utils_np import (
        label_contiguous_1d,
        multi_arange,
        relabel_groups_masked,
        relabel_groups_unique,
    )

try:
    import numba
except ImportError:
    # no numba, no numba backend - the old name says so as well, and the
    # benchmarks ask for it before they measure anything
    aggregate_numba = None
    aggregate_nb = None
    # the gather into an array of the caller is numba work
    unpack_into = None
else:
    # the module of the implementation - importing it sets it as the
    # attribute of the package as well
    from numpy_groupies import aggregate_numba
    from numpy_groupies.aggregate_numba import aggregate, step_count, step_indices, unpack, unpack_into

    # the numba implementation wins: the aggregate of the package is its one
    aggregate_nb = aggregate


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
# One entry point per function of the package, with the function in the name:
# ``npg.mean(group_idx, a)`` is the mean of every group, computed by the best
# implementation available.  They are the entry points of that implementation,
# so they carry the arguments their function has of its own - the ``ddof`` of
# the variance, the ``q`` of the quantile - and they behave like ``aggregate``
# with the name of the function as its ``func`` argument.
#
# The names are built at runtime, out of the table of the entry points - so
# that they cannot drift from it.  The type checker cannot see that table, so
# the names are declared for it below; the test
# ``test_every_function_of_the_package_is_declared_for_the_type_checker`` in
# ``tests/test_registry.py`` keeps the two in step.

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


# ``purge()`` removes the cached compiles of the package - it is the tool for
# the cache of the numba backend, which lives in a directory of its own (see
# ``numpy_groupies.utils_cache.numba_cache_dir``) and which numba never cleans
# up.


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
