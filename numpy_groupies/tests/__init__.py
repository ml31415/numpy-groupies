from functools import wraps

import pytest

from .. import aggregate_numpy, aggregate_numpy_ufunc, aggregate_purepy

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
    "purepy": {
        "not_implemented": ("cumsum", "cumprod", "cummax", "cummin", "sumofsquares")
    },
    "numba": {"not_implemented": ("array", "list", "sort")},
    "pandas": {
        "not_implemented": ("array", "list", "sort", "sumofsquares", "nansumofsquares")
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
    if implemented and funcname not in implemented:
        return False
    return True


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


func_list = (
    "sum",
    "prod",
    "min",
    "max",
    "all",
    "any",
    "mean",
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
    "nanstd",
    "nanvar",
    "nanlen",
    "nanargmin",
    "nanargmax",
    "nansumofsquares",
)
