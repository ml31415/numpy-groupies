from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from numpy_groupies import aggregate_purepy
from numpy_groupies.registry import function_names

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol, TypeAlias

    import numpy as np
    import numpy.typing as npt

    from numpy_groupies.registry import FuncName
    from numpy_groupies.utils_np import AFillValue, AResult, AScalar, AValue, RavelOrder

    class Aggregate(Protocol):
        """The canonical signature of the public ``aggregate``.

        The public name is bound to the best implementation available at
        import time (numba, else numpy, else pure python) - they agree on
        everything but the most restrictive ``size`` of the pure python
        fallback, which does not take a sequence, and on the ``out`` and
        ``cache`` keyword arguments, which only the numba implementation
        honours (the others raise or ignore them).
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
            cache: bool = True,
            **kwargs: Any,
        ) -> AResult: ...


def dummy_no_impl(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "You may need to install another package (numpy or numba) to access a working implementation."
    )


def _missing_backend(name: str, package: str) -> ModuleType:
    """The stand-in for a backend whose package is not installed.

    Every attribute of it says what is missing, in the words of the error
    the implementation itself would raise.
    """
    module = ModuleType(name)

    def _getattr(attribute: str) -> Any:
        raise NotImplementedError(
            f"You may need to install another package ({package}) to access a working implementation."
        )

    module.__getattr__ = _getattr  # type: ignore[method-assign]
    return module


# ---------------------------------------------------------------------------
# the backends
#
# A backend is the module of its implementation: it carries the aggregate of
# the implementation and the entry point of every function of the package as
# its attribute - ``aggregate_numpy.mean(group_idx, a)`` is the mean of every
# group, with the function in the name.  The numba module carries its
# kernels there, which jitted code can reach (a plain python object is no
# numba type, a module attribute is).

# the pure python backend is always there - it is the one that works without
# numpy
#
# the module of the best implementation available - the functions of the
# package below are the ones of this one
_backend: ModuleType = aggregate_purepy

# declared with the type of the protocol, or the branches below would join
# into a plain Any and lose its signature
aggregate: Aggregate

aggregate_numpy: ModuleType
aggregate_numba: ModuleType | None

try:
    import numpy as np
except ImportError:
    # the numpy implementation needs numpy - without it every entry point
    # says so, the way the function itself does
    aggregate_numpy = _missing_backend("numpy_groupies.aggregate_numpy", "numpy")
    multi_arange = label_contiguous_1d = dummy_no_impl
else:
    from numpy_groupies import aggregate_numpy
    from numpy_groupies.aggregate_numpy import step_count as step_count_np_impl
    from numpy_groupies.aggregate_numpy import step_indices as step_indices_np_impl

    _backend = aggregate_numpy

    # the numba branch below may rebind both, onto other signatures -
    # what they take depends on the backend that won, so say it loosely
    step_count: Callable[[Any], Any] = step_count_np_impl
    step_indices: Callable[[Any], Any] = step_indices_np_impl

    from numpy_groupies.common import default_fill_value
    from numpy_groupies.utils_np import (
        label_contiguous_1d,
        multi_arange,
        relabel_groups_masked,
        relabel_groups_unique,
        unpack,
    )

try:
    import numba
except ImportError:
    aggregate_numba = None
else:
    from numpy_groupies import aggregate_numba as _aggregate_numba
    from numpy_groupies.aggregate_numba import step_count as step_count_nb_impl
    from numpy_groupies.aggregate_numba import step_indices as step_indices_nb_impl
    from numpy_groupies.aggregate_numba import unpack_into

    aggregate_numba = _aggregate_numba
    _backend = _aggregate_numba
    step_count = step_count_nb_impl
    step_indices = step_indices_nb_impl

# the aggregate of the best implementation available, and the functions of
# the package in the package itself: ``npg.mean(group_idx, a)`` is the mean
# of every group, computed by the best implementation there is
aggregate = cast("Aggregate", _backend.aggregate)
globals().update({name: getattr(_backend, name) for name in function_names()})


def uaggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    out: AResult | None = None,
    **kwargs: Any,
) -> AResult:
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
    if aggregate_numba is None or aggregate is not aggregate_numba.aggregate:
        raise NotImplementedError("out= is only supported by the numba implementation")
    return unpack_into(group_idx, ret, out)


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
