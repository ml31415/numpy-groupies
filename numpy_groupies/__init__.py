from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from numpy_groupies.aggregate_purepy import aggregate as aggregate_py_impl
from numpy_groupies.common import Backend, bind_functions, function_names

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


# declared with the type of the protocol, or the branches below would join
# into a plain Any and lose its signature
aggregate: Aggregate
aggregate_np: Aggregate
aggregate_py: Aggregate

# the pure python backend is an object like the others: callable, and it
# carries the entry point of every function as its attribute
aggregate_py = Backend(
    aggregate_py_impl,
    bind_functions(aggregate_py_impl, function_names()),
    name="purepy",
    doc="The pure python implementation of aggregate.",
)

aggregate = cast("Aggregate", aggregate_py)

try:
    import numpy as np
except ImportError:
    # the numpy implementation needs numpy - without it every entry point
    # says so, the way the function itself does
    aggregate_np = Backend(
        dummy_no_impl,
        {name: dummy_no_impl for name in function_names()},
        name="numpy",
        doc="The pure numpy implementation of aggregate.",
    )
    multi_arange = label_contiguous_1d = dummy_no_impl
else:
    from numpy_groupies.aggregate_numpy import aggregate as aggregate_np_impl
    from numpy_groupies.aggregate_numpy import step_count as step_count_np_impl
    from numpy_groupies.aggregate_numpy import step_indices as step_indices_np_impl

    # the numba branch below may rebind both, onto other signatures -
    # what they take depends on the backend that won, so say it loosely
    step_count: Callable[[Any], Any] = step_count_np_impl
    step_indices: Callable[[Any], Any] = step_indices_np_impl

    # the backend is an object: it is callable like the function it
    # replaces, and it carries the entry point of every function as its
    # attribute - ``aggregate.mean(group_idx, a)`` is the mean of every
    # group, with the function in the name
    aggregate_np = Backend(
        aggregate_np_impl,
        bind_functions(aggregate_np_impl, function_names()),
        name="numpy",
        doc="The pure numpy implementation of aggregate.",
    )
    aggregate = cast("Aggregate", aggregate_np)
    from numpy_groupies.common import default_fill_value
    from numpy_groupies.utils_np import (
        label_contiguous_1d,
        multi_arange,
        relabel_groups_masked,
        relabel_groups_unique,
        unpack,
    )

aggregate_nb: Aggregate | None
try:
    import numba
except ImportError:
    aggregate_nb = None
else:
    from numpy_groupies.aggregate_numba import aggregate as aggregate_nb_impl
    from numpy_groupies.aggregate_numba import step_count as step_count_nb_impl
    from numpy_groupies.aggregate_numba import step_indices as step_indices_nb_impl
    from numpy_groupies.aggregate_numba import unpack_into

    # the attributes of a backend are the entry points of its functions -
    # numba cannot reach them from jitted code (a plain python object is no
    # numba type), so the kernels - the same functions, compiled once per
    # dtype - are the attributes of the module of the implementation, which is
    # what jitted code imports
    aggregate_nb = Backend(
        aggregate_nb_impl,
        bind_functions(aggregate_nb_impl, function_names()),
        name="numba",
        doc="The numba implementation of aggregate.",
    )
    step_count = step_count_nb_impl
    step_indices = step_indices_nb_impl
    aggregate = aggregate_nb


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
        ret       = [13.3, 2.5, 2.5, -8.2, 2.5, 13.3, 46.05, 46.05, 2.5, 53.7]

    with ``func='mean'``. A typical use is demeaning within each group, as in
    ``a - uaggregate(group_idx, a, func='mean')``.

    Pass ``out`` to gather into an array provided by the caller, which has to match the
    shape of ``ret[group_idx]`` and the dtype of ``ret``. Only the numba implementation
    supports this, the others raise ``NotImplementedError``.

    See also ``aggregate``, ``unpack``, ``unpack_into``.
    """
    ret = aggregate(group_idx, a, **kwargs)
    if out is None:
        return unpack(group_idx, ret)
    if aggregate is not aggregate_nb:
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
