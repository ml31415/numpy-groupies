"""Names, aliases and single-lookup dispatch for the functions of the package.

Properties live in ``numpy_groupies.functions``; the default sentinel and its
resolution in ``numpy_groupies.fills``.  This module must not import numpy:
the pure python implementation loads before numpy is known to be available.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias, cast

if TYPE_CHECKING:
    from numpy_groupies.typing import AScalar


# fmt: off
funcs_common = [
    "sum",
    "prod",
    "first",
    "last",
    "len",
    "nunique",
    "mean",
    "median",
    "quantile",
    "percentile",
    "trapezoid",
    "var",
    "std",
    "skew",
    "kurt",
    "all",
    "any",
    "allnan",
    "anynan",
    "max",
    "min",
    "argmax",
    "argmin",
    "sort",
    "array",
    "topk",
    "mode",
    "sumofsquares",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
]


funcs_no_separate_nan = frozenset(["sort", "array", "allnan", "anynan", "topk", "mode"])


funcs_collecting = frozenset(["array"])


FuncName: TypeAlias = Literal[
    "add",
    "all",
    "allnan",
    "amax",
    "amin",
    "and",
    "any",
    "anynan",
    "argmax",
    "argmin",
    "array",
    "asort",
    "asorted",
    "count",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "dsort",
    "first",
    "last",
    "len",
    "max",
    "maximum",
    "nannunique",
    "mean",
    "median",
    "min",
    "minimum",
    "mode",
    "multiply",
    "nanall",
    "nanany",
    "nanargmax",
    "nanargmin",
    "nancummax",
    "nancummin",
    "nancumprod",
    "nancumsum",
    "nanfirst",
    "nanlast",
    "nanlen",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanprod",
    "nanquantile",
    "nanskew",
    "nankurt",
    "nanstd",
    "nansum",
    "nansumofsquares",
    "nantrapezoid",
    "nanvar",
    "or",
    "skew",
    "kurt",
    "kurtosis",
    "percentile",
    "plus",
    "prod",
    "product",
    "quantile",
    "rsorted",
    "sort",
    "sorted",
    "splice",
    "split",
    "std",
    "sum",
    "sumofsquares",
    "times",
    "topk",
    "trapezoid",
    "var",
    "nunique",
]
# fmt: on


_alias_str = {
    "or": "any",
    "and": "all",
    "add": "sum",
    "count": "len",
    "plus": "sum",
    "multiply": "prod",
    "product": "prod",
    "times": "prod",
    "amax": "max",
    "maximum": "max",
    "amin": "min",
    "minimum": "min",
    "split": "array",
    "splice": "array",
    "sorted": "sort",
    "asort": "sort",
    "asorted": "sort",
    "rsorted": "sort",
    "dsort": "sort",
    "kurtosis": "kurt",
}


_alias_builtin: dict[str | Callable[..., object], str] = {
    all: "all",
    any: "any",
    len: "len",
    max: "max",
    min: "min",
    sum: "sum",
    sorted: "sort",
    slice: "array",
    list: "array",
}


def get_aliasing(*extra: dict[str | Callable[..., object], str]) -> dict[str | Callable[..., object], str]:
    """Map strings and functions to the canonical function names,
    e.g. ``alias['add'] = 'sum'`` and ``alias[sorted] = 'sort'``.  Import time only.
    """
    alias: dict[str | Callable[..., object], str] = {k: k for k in funcs_common}
    alias.update(_alias_str)
    alias.update((fn, fn) for fn in _alias_builtin.values())
    alias.update(_alias_builtin)
    for d in extra:
        alias.update(d)
    alias.update((k, k) for k in set(alias.values()))
    # Treat nan-functions as firstclass member and add them directly
    for key in set(alias.values()):
        if key not in funcs_no_separate_nan and not key.startswith("nan"):
            key = "nan" + key
            alias[key] = key
    return alias


# alias table without objects, so the pure python backend can use it without
# numpy; numpy-based backends add their function aliases
aliasing_py = get_aliasing()


def can_name_a_function(obj: object) -> bool:
    """Whether ``obj`` can be a dict key (and so name a function): an object
    defining ``__eq__`` without ``__hash__`` cannot, nor can a list or dict.
    """
    return type(obj).__hash__ is not None


def resolve_name(func: Any, aliasing: Mapping[str | Callable[..., object], str] = aliasing_py) -> str | None:
    """The canonical name of ``func``, or ``None`` where it names no function."""
    return aliasing.get(func) if can_name_a_function(func) else None


def function_names() -> list[str]:
    """All function names: plain ones and their nan variants (functions without
    a separate nan variant appear once).
    """
    names = list(funcs_common)
    names += ["nan" + name for name in funcs_common if name not in funcs_no_separate_nan]
    return names


def missing_impl_error(name: str | Callable[..., AScalar]) -> Exception:
    """The error for a function that is known but not implemented by this backend."""
    if isinstance(name, str) and name.startswith("nan") and name[3:] in funcs_no_separate_nan:
        return ValueError(f"{name[3:]} does not have a nan-version")
    return NotImplementedError("No such function available")


def custom_callable(func: Any) -> Callable[..., AScalar]:
    """The callable behind a ``func`` that is no key of the dispatch table,
    or an error if it is no callable either.  Cold path only.
    """
    if callable(func):
        return cast("Callable[..., AScalar]", func)
    raise ValueError(f"func {func} is neither a valid function string nor a callable object")


def build_dispatch(
    op_dict: Mapping[str, Callable[..., Any]],
    aliasing: Mapping[Any, str],
    temporal_op_dict: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[Any, Resolved]:
    """Fuse an aliasing and the implementation tables for single-lookup dispatch:
    each alias holds the canonical name plus the numeric and temporal ops.
    ``(name, None)`` marks an unimplemented name (``missing_impl_error``); the
    temporal op is None where the numeric op handles temporal input itself.
    """
    return {
        alias: Resolved(name, op_dict.get(name), (temporal_op_dict or {}).get(name)) for alias, name in aliasing.items()
    }


class Resolved(NamedTuple):
    """What a func resolved to: canonical name (or caller's own callable), plus
    the numeric and temporal ops - each None where the backend has none.
    """

    funcname: str | Callable[..., AScalar]
    op: Callable[..., Any] | None
    temporal_op: Callable[..., Any] | None = None


class Registry(dict[str | Callable[..., object], Resolved]):
    """One backend's registry: every alias holds the canonical name and the
    implementation (None where unimplemented).  One plain-subscript lookup
    answers known names and caller callables alike - the latter through
    ``__missing__``, and ``custom`` for funcs that cannot be dict keys at all.
    """

    def __init__(
        self,
        op_dict: Mapping[str, Callable[..., Any]],
        aliasing: Mapping[str | Callable[..., object], str],
        custom: Callable[..., Resolved],
        temporal_op_dict: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        super().__init__(build_dispatch(op_dict, aliasing, temporal_op_dict))
        self._custom_factory = custom

    def __missing__(self, key: Any) -> Resolved:
        """Resolve an unknown key: the same lookup as a known name, via the factory."""
        return self._custom_factory(key)

    def custom(self, func: Any) -> Resolved:
        """Resolve a func that cannot be a dict key (no ``__hash__``), via the factory."""
        return self._custom_factory(func)
