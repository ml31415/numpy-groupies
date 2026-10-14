from functools import wraps
from typing import Any

import pytest

from .. import aggregate_numpy, aggregate_numpy_ufunc, aggregate_purepy

aggregate_numba: Any
aggregate_pandas: Any

try:
    from .. import aggregate_numba
except ImportError:
    aggregate_numba = None
try:
    from .. import aggregate_pandas
except ImportError:
    aggregate_pandas = None

_implementations = [
    aggregate_purepy,
    aggregate_numpy_ufunc,
    aggregate_numpy,
    aggregate_numba,
    aggregate_pandas,
]
_implementations = [i for i in _implementations if i is not None]


def _impl_name(impl):
    if not impl or type(impl).__name__ == "NotSetType":
        return
    return impl.__name__.rsplit("aggregate_", 1)[1].rsplit("_", 1)[-1]


_implemented_by_impl_name = {
    "numpy": {"not_implemented": ("cumprod", "cummax", "cummin")},
    "purepy": {"not_implemented": ("cumsum", "cumprod", "cummax", "cummin", "sumofsquares")},
    "numba": {"not_implemented": ("array", "list", "sort")},
    "pandas": {
        "not_implemented": ("array", "list", "sort", "sumofsquares", "nansumofsquares", "trapezoid", "nantrapezoid")
    },
    "ufunc": {
        "implemented": (
            "sum",
            "prod",
            "min",
            "max",
            "len",
            "all",
            "any",
            "anynan",
            "allnan",
        )
    },
}


def _is_implemented(impl_name, funcname):
    func_description = _implemented_by_impl_name[impl_name]
    not_implemented = func_description.get("not_implemented", [])
    implemented = func_description.get("implemented", [])
    if impl_name == "purepy" and funcname.startswith("nan"):
        return False
    if funcname in not_implemented:
        return False
    return not (implemented and funcname not in implemented)


def _wrap_notimplemented_skip(impl, name=None):
    """Some implementations lack some functionality. That's ok, let's skip that instead of raising errors."""

    @wraps(impl)
    def try_skip(*args, **kwargs):
        try:
            return impl(*args, **kwargs)
        except NotImplementedError:
            impl_name = impl.__module__.split("_")[-1]
            func = kwargs.pop("func", None)
            if callable(func):
                func = func.__name__
            if not _is_implemented(impl_name, func):
                pytest.skip("Functionality not implemented")

    if name:
        try_skip.__name__ = name
    return try_skip


class _BackendProxy:
    """Proxy around a backend module that wraps ``aggregate`` with the
    ``NotImplementedError`` -> ``pytest.skip`` translation, and delegates
    every other attribute (e.g. ``step_count``, ``step_indices``) to the
    underlying module so tests can use ``impl.aggregate(...)``,
    ``impl.step_count(...)`` etc. without caring whether the attribute
    is wrapped or not.
    """

    def __init__(self, module, name):
        self._module = module
        self.aggregate = _wrap_notimplemented_skip(module.aggregate, f"aggregate_{name}")

    def __getattr__(self, name):
        return getattr(self._module, name)

    @property
    def __name__(self):
        return self._module.__name__


func_list = (
    "sum",
    "prod",
    "min",
    "max",
    "all",
    "any",
    "mean",
    "median",
    "trapezoid",
    "std",
    "var",
    "len",
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
    "nantrapezoid",
    "nanstd",
    "nanvar",
    "nanlen",
    "nanargmin",
    "nanargmax",
    "nansumofsquares",
)


def _deselect_purepy(impl, *args, **kwargs):
    # purepy implementations does not handle nan values and ndim correctly.
    # So it needs to be excluded from several tests."""
    return impl.__name__.endswith("purepy")


def _deselect_purepy_and_pandas(impl, *args, **kwargs):
    # purepy and pandas implementation handle some nan cases differently.
    # So they need to be excluded from several tests."""
    return impl.__name__.endswith(("pandas", "purepy"))


def _deselect_purepy_and_invalid_axis(impl, func, size, axis):
    impl_name = impl.__name__.split("_")[-1]
    if impl_name == "purepy":
        # purepy does not handle axis parameter
        return True
    if axis >= len(size):
        return True
    return not _is_implemented(impl_name, func)


def _deselect_purepy_axis(impl, axis):
    # purepy does not support axis, so deselect it whenever an axis is
    # requested. Use this for tests that do not need the full
    # _deselect_purepy_and_invalid_axis bookkeeping.
    return impl.__name__.endswith("purepy") and axis is not None


def _deselect_not_implemented(impl, func, *args, **kwargs):
    impl_name = impl.__name__.split("_")[-1]
    return not _is_implemented(impl_name, func)


def _deselect_no_step_count(impl, *args, **kwargs):
    # Only numpy and numba backends expose step_count / step_indices.
    return not hasattr(impl, "step_count")
