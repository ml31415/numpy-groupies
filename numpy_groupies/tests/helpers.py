"""Shared helpers for the test suite: the backend registry and the ``impl``
fixture proxy.

Backends signal missing functionality with ``NotImplementedError``. The
``_BackendProxy`` translates that into ``pytest.skip`` at runtime, so a module
running over every backend needs no bookkeeping for the functions one of them
lacks.  What is left of the predicates answers a question inside a test -
whether this backend computes with python numbers - and nothing more: a result
that deviates from the group loop is measured like any other result, never
filtered out of the collection.
"""

from functools import wraps

import numpy as np
import pytest

from numpy_groupies.driver import aliasing
from numpy_groupies.registry import (
    aliasing_py,
    funcs_collecting,
    function_names,
)
from numpy_groupies.tests.reference import aggregate_ref, reference_names


def impl_name(impl):
    """Short backend name ("purepy", "numpy", ...) for a backend module
    or its ``_BackendProxy``.

    The name of the module of a backend carries the name of the backend:
    ``npg_purepy`` is the pure python one, ``npg_numpy`` the numpy one.
    """
    if isinstance(impl, _BackendProxy):
        return impl._name
    return impl.__name__.rsplit(".", 1)[-1].removeprefix("npg_")


def random_group_idx(rng, size, n_groups):
    """Random group labels in ``[0, n_groups)`` with the given shape.

    Pass the ``rng`` fixture as generator.
    """
    return rng.integers(0, n_groups, size=size)


def random_values(rng, size):
    """Random normally distributed float values with the given shape.

    Pass the ``rng`` fixture as generator.
    """
    return rng.standard_normal(size)


def is_supported(impl, func):
    """True if the backend runs ``func``.

    The dispatch table of the backend answers it; the answer is a lookup and
    nothing else - asking the backend to run the function would make the numba
    backend compile a kernel per function in every worker of the parallel run,
    which is where the minutes of a cold run went.  A func that names no
    function of the package is the caller's own and is never deselected.
    Accepts a backend module or its ``_BackendProxy``, and a function name
    or callable.
    """
    if isinstance(impl, _BackendProxy):
        impl = impl._module
    if not isinstance(func, str):
        # a func that cannot be a key of a dict cannot key the dispatch table
        # either, and no func of one's own is ever deselected
        func = getattr(func, "__name__", None)
        if not isinstance(func, str):
            return True
    # every backend holds a dispatch table: a table of the aliases of the
    # functions it knows and the implementation of its own behind every one
    # of them - and one that has none knows no function at all
    table = getattr(impl, "_dispatch", None)
    if table is None:
        return True
    resolved = table.get(func)
    return resolved is not None and resolved.op is not None


def _wrap_notimplemented_skip(impl, name=None):
    """Backends raise ``NotImplementedError`` for missing functionality.
    Translate that into ``pytest.skip`` instead of a test error."""

    @wraps(impl)
    def try_skip(*args, **kwargs):
        try:
            return impl(*args, **kwargs)
        except NotImplementedError:
            pytest.skip("Functionality not implemented")

    if name:
        try_skip.__name__ = name
    return try_skip


class _BackendProxy:
    """Proxy around a backend module.

    - ``aggregate`` is wrapped so that ``NotImplementedError`` becomes a
      ``pytest.skip``.
    - every other attribute is forwarded to the module; a missing attribute
      (e.g. ``step_count`` on backends that do not provide it) also becomes
      a ``pytest.skip``.
    """

    def __init__(self, module):
        self._module = module
        self._name = module.__name__.rsplit(".", 1)[-1].removeprefix("npg_")
        self.aggregate = _wrap_notimplemented_skip(module.aggregate, f"aggregate_{self._name}")

    @property
    def _dispatch(self):
        """The dispatch table of the backend under test."""
        return self._module._dispatch

    def __getattr__(self, name):
        if name.startswith("_"):
            # keep dunder/private probes (copy, pickle, introspection) honest
            raise AttributeError(name)
        try:
            return getattr(self._module, name)
        except AttributeError:
            pytest.skip(f"{self._name} backend does not provide {name!r}")


# The functions the sweeps of the suite ask for: every function the package
# knows whose value the group loop of the reference computes as well.  The
# list is derived from the registry of the package, so a function that is
# added there is asked for from the start - the three nan variants of the
# cumulative functions are the ones the loop does not know yet.
func_list = tuple(name for name in function_names() if name in set(reference_names()))


def assert_result_dtype(impl, res, dtype):
    """The dtype of the answer of a backend.

    The pure python implementation carries no dtype of its own - its python
    numbers are the answer - so the dtype is asked of the backends that have
    one.  The question is asked in one place, so that the tests over the
    complex values do not answer it again and again.
    """
    if not is_purepy(impl):
        assert np.asarray(res).dtype == np.dtype(dtype)


def assert_same_result(got, want, func, rtol=10**-10, atol=0.0):
    """Compare the results of two backends, whatever the function returns.

    The collecting functions hand out a sequence per group, the others one
    number - the comparison looks at the one it is given, so that the tests
    over ``func_list`` need no exception for the shape of the result.

    ``atol`` is the absolute tolerance of the answers that are zero: the
    variance of a group whose values are all the same is zero, and the two
    ways of computing it leave a rounding residue behind - one of them the
    size of the sum of the values, the other one of its square.
    """
    if func in funcs_collecting:
        _assert_same_groups(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def assert_matches_the_group_loop(impl, func, block, values, rtol=10**-10, atol=0.0):
    """Compare the result of a backend against the group loop.

    The yardstick of the suite: a backend is measured against the group loop
    of :mod:`.reference`, never against itself.  The values are the ones of
    the block - the plain ones, the ones with nans in them or the complex
    ones.
    """
    got = impl.aggregate(block.group_idx, values, func=func, size=block.size)
    want = aggregate_ref(block.group_idx, values, func=func, size=block.size)
    assert_same_result(got, want, func, rtol=rtol, atol=atol)


def _assert_same_groups(got, want):
    """Compare two collections of per-group sequences."""
    groups = list(got) if isinstance(got, list) else np.ravel(got)
    expected = list(want) if isinstance(want, list) else np.ravel(want)
    assert len(groups) == len(expected), f"{len(groups)} groups against {len(expected)}"
    for g, (one, other) in enumerate(zip(groups, expected)):
        if one is None or other is None:
            assert one is None and other is None, f"group {g}: {one!r} != {other!r}"
        else:
            np.testing.assert_array_equal(one, other, err_msg=f"group {g}")


def is_purepy(impl):
    """True for the implementation that computes with python numbers.

    Those carry no dtype: purepy neither honours a ``dtype`` argument nor
    hands back a result of the dtype of its input.  Its other gaps have their
    own names - the temporal ones the ``skip_purepy`` fixture, the
    multidimensional output the refusal of the backend itself.
    """
    return impl_name(impl) == "purepy"


def is_numba(impl):
    """True for the implementation that compiles what it is given as func.

    A compiled function and a callable object are two different things to that
    implementation, where the other two only call what they are handed.
    """
    return impl_name(impl) == "numba"


def aliasing_of(impl):
    """The alias table of the backend under test.

    The pure python backend knows fewer aliases than the numpy based ones -
    it has no numpy to take them from - so the table belongs to the backend
    and not to the package.
    """
    if impl_name(impl) == "purepy":
        return aliasing_py
    return aliasing


def assert_matches_the_group_loop_values(impl, group_idx, a, func, size=None, rtol=10**-10):
    """The result of a backend against the group loop, for the values of the
    caller.

    The counts compare exactly - an integer count has no tolerance - and the
    moments and the quantiles with the tolerance of their own.
    """
    got = impl.aggregate(group_idx, a, func=func, size=size)
    want = aggregate_ref(group_idx, a, func=func, size=size)
    np.testing.assert_allclose(got, want, rtol=rtol, equal_nan=True)


# The sizes of the data the generic sweep runs over: the small one is below
# the floor of the parallel path of the numba backend (``MT_MIN_VALUES`` is
# 100_000), the big one above it.  The sweep measures the sequential path and
# the parallel one with the very same test - the backends that have no
# threads answer the same at both sizes.
SWEEP_SIZES = {"small": 1_000, "large": 200_000}
