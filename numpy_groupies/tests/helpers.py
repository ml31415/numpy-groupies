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
from typing import Any

import numpy as np
import pytest

from numpy_groupies import aggregate_numpy, aggregate_purepy
from numpy_groupies.driver import aliasing
from numpy_groupies.registry import (
    aliasing_py,
    funcs_collecting,
    function_names,
)
from numpy_groupies.tests.reference import aggregate_ref, reference_names

aggregate_numba: Any

try:
    from numpy_groupies import aggregate_numba
except ImportError:
    aggregate_numba = None

# backend registry: module -> short name, in fixture-parametrization order
_backend_names = {
    module: name
    for name, module in (
        ("purepy", aggregate_purepy),
        ("numpy", aggregate_numpy),
        ("numba", aggregate_numba),
    )
    if module is not None
}
_implementations = list(_backend_names)


def impl_name(impl):
    """Short backend name ("purepy", "numpy", ...) for a backend module
    or its ``_BackendProxy``."""
    if isinstance(impl, _BackendProxy):
        return impl._name
    return _backend_names.get(impl)


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

    The dispatch table of the backend answers it: a function the table does
    not hold, or one it holds without an implementation behind it, is a
    function this backend does not have.

    The answer is a lookup and nothing else.  Asking the backend to run the
    function would be the same question with a price: the numba backend would
    compile the kernel of the function, and the collection of the suite would
    compile a kernel per function in every worker of the parallel run - which
    is where the minutes of a cold run went.

    A func that names no function of the package is the caller's own and is
    never deselected.

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
    table = _dispatch_of(impl)
    if table is None:
        return True
    resolved = table.get(func)
    return resolved is not None and resolved.op is not None


def _dispatch_of(impl):
    """The dispatch table of a backend, or None for one that has none.

    Every backend holds one: a table of the aliases of the functions it knows
    and the implementation of its own behind every one of them.
    """
    if impl is aggregate_purepy:
        return aggregate_purepy._dispatch
    if impl is aggregate_numpy:
        return aggregate_numpy.ops._dispatch
    if aggregate_numba is not None and impl is aggregate_numba:
        return aggregate_numba.ops._dispatch
    return None


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
        self._name = _backend_names[module]
        self.aggregate = _wrap_notimplemented_skip(module.aggregate, f"aggregate_{self._name}")

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

    Those carry no dtype: purepy neither honours a ``dtype`` argument nor hands
    back a result of the dtype of its input.
    """
    return impl_name(impl) == "purepy"


def is_numba(impl):
    """True for the implementation that compiles what it is given as func.

    A compiled function and a callable object are two different things to that
    implementation, where the other two only call what they are handed.
    """
    return impl_name(impl) == "numba"


# The functions with a well-defined complex result, following numpy semantics -
# the order statistics included, as numpy orders complex values lexicographically.
# median is the one order statistic missing: over incomparable values numpy's own
# partition and its sort disagree, so the group loop cannot tell what a poisoned
# group mediates to - what its answer has to look like is pinned by
# test_median_of_a_poisoned_group instead, and a clean group by test_median
complex_funcs = (
    "sum",
    "prod",
    "mean",
    "var",
    "std",
    "trapezoid",
    "sumofsquares",
    "first",
    "last",
    "nansum",
    "nanprod",
    "nanmean",
    "nanvar",
    "nanstd",
    "nanmedian",
    "nantrapezoid",
    "nansumofsquares",
    "cumsum",
    "all",
    "any",
    "allnan",
    "anynan",
    "nanall",
    "nanany",
    "len",
    "nanlen",
    "nanfirst",
    "nanlast",
    "min",
    "max",
    "nanmin",
    "nanmax",
    "mode",
    "argmin",
    "argmax",
    "nanargmin",
    "nanargmax",
)
# their complex result is the real squared magnitude, not complex a*a
complex_real_out_funcs = {"var", "std", "sumofsquares", "nanvar", "nanstd", "nansumofsquares"}
# these count, test or point at a position, so their result is not a number of
# the input type
complex_bool_funcs = {"all", "any", "allnan", "anynan", "nanall", "nanany"}
complex_arg_funcs = ("argmin", "argmax", "nanargmin", "nanargmax")
complex_int_funcs = {"len", "nanlen", *complex_arg_funcs}
# order statistics over the values themselves, as opposed to their positions
complex_order_funcs = ("min", "max", "nanmin", "nanmax", "mode")
# conjugating the input does not simply conjugate the result for these: it
# flips the order of two values sharing a real part
complex_asymmetric_funcs = complex_order_funcs + complex_arg_funcs
# functions whose output dtype a complex input does not fix
complex_dtype_funcs = tuple(complex_real_out_funcs | complex_bool_funcs | complex_int_funcs)


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
