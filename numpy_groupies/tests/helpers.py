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

from functools import cache, wraps
from typing import Any

import numpy as np
import pytest

from numpy_groupies import aggregate_numpy, aggregate_purepy
from numpy_groupies.registry import (
    funcs_collecting,
)

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

    Probed with a tiny real-valued and a tiny complex-valued input: a func
    counts as supported when the backend resolves it for either, so that
    dtype-specific gaps (a backend implementing sumofsquares only for
    complex input, say) do not deselect testable combos. Combos failing for only
    one dtype still skip at runtime via the backend proxy.

    Accepts a backend module or its ``_BackendProxy``, and a function name
    or callable.
    """
    if isinstance(impl, _BackendProxy):
        impl = impl._module
    if not isinstance(func, str):
        # a func that cannot be a key of a dict cannot key the probe cache
        # either, and no func of one's own is ever deselected
        func = getattr(func, "__name__", None)
        if not isinstance(func, str):
            return True
    return _is_supported(impl, func)


@cache
def _is_supported(impl, func):
    """The probe itself, keyed by a backend module and a function name."""
    group_idx = np.array([0, 0, 1, 1])
    inputs = (np.arange(4.0), np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]))
    for a in inputs:
        try:
            impl.aggregate(group_idx, a, func=func)
        except (NotImplementedError, ValueError):
            continue
        except Exception:  # noqa: BLE001 - unexpected, but the func runs
            return True
        return True
    return False


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


func_list = (
    "sum",
    "prod",
    "min",
    "max",
    "all",
    "any",
    "mean",
    "median",
    "quantile",
    "percentile",
    "trapezoid",
    "std",
    "var",
    "skew",
    "kurt",
    "len",
    "nunique",
    "argmin",
    "argmax",
    "anynan",
    "allnan",
    "cumsum",
    "sumofsquares",
    "nansum",
    "nanprod",
    "nanmin",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nanpercentile",
    "nantrapezoid",
    "nanstd",
    "nanvar",
    "nanskew",
    "nankurt",
    "nanlen",
    "nannunique",
    "nanargmin",
    "nanargmax",
    "nansumofsquares",
    "array",
)


def assert_same_result(got, want, func, rtol=10**-10):
    """Compare the results of two backends, whatever the function returns.

    The collecting functions hand out a sequence per group, the others one
    number - the comparison looks at the one it is given, so that the tests
    over ``func_list`` need no exception for the shape of the result.
    """
    if func in funcs_collecting:
        _assert_same_groups(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=rtol)


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


def is_purepy(impl, *args, **kwargs):
    """True for the implementation that computes with python numbers.

    Those carry no dtype: purepy neither honours a ``dtype`` argument nor hands
    back a result of the dtype of its input.
    """
    return impl_name(impl) == "purepy"


def is_numba(impl, *args, **kwargs):
    """True for the implementation that compiles what it is given as func.

    A compiled function and a callable object are two different things to that
    implementation, where the other two only call what they are handed.
    """
    return impl_name(impl) == "numba"
