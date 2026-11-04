"""The registry of the functions of the package: their names, their aliases
and the operations of a backend.

Every function of the package has a name, a set of aliases and a set of
properties - the result type it settles, the default that fills a group it
never saw, the arguments it has of its own.  The names and the aliases live
here, and so does the dispatch that resolves one of them to the operations of
a backend: one lookup answers the canonical name of the function and the two
operations that do the work.

The properties of the functions live in ``numpy_groupies.functions``, next to
the sentinel that asks for the default of a function, which lives in
``numpy_groupies.fills`` with the resolution of the fill value.

This module must not import numpy: the pure python implementation is loaded
before numpy is known to be available, and has to work without it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias, cast

if TYPE_CHECKING:
    from numpy_groupies.typing import AScalar


# fmt: on


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
    "sumofsquares",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
]


funcs_no_separate_nan = frozenset(["sort", "array", "allnan", "anynan", "topk"])


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
    """
    Assembles a dictionary that maps both strings and functions to a list of supported function names.

    Examples:
        alias['add'] = 'sum'
        alias[sorted] = 'sort'

    This function should only be called during import.
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


aliasing_py = get_aliasing()


def can_name_a_function(obj: object) -> bool:
    """Whether ``obj`` can be a key of the alias table, and so can name a function.

    The table is a dict, which asks its keys for a hash, and an object that
    defines ``__eq__`` without a ``__hash__`` does not have one (a dataclass with
    ``eq=True`` is the everyday case), as little as a list or a dict has one.
    Asked of the type in front of the lookup, this is all that is needed to make
    the lookup itself unable to fail: what cannot be a key names no function,
    and does not have to be looked up to know that.
    """
    return type(obj).__hash__ is not None


def resolve_name(func: Any, aliasing: Mapping[str | Callable[..., object], str] = aliasing_py) -> str | None:
    """The canonical name of ``func``, or ``None`` where it names no function.

    Every name and every alias is a key of ``aliasing`` and every key of it is
    hashable (see ``get_aliasing``, which is the only writer), so this is one
    lookup that can neither miss nor fail - a func that cannot be a key is not a
    function anybody knows, and is not asked for one.
    """
    return aliasing.get(func) if can_name_a_function(func) else None


def function_names() -> list[str]:
    """The names of the functions of the package: the plain ones and the
    nan-skipping ones beside them.

    The four functions that do not skip nans have no nan variant of their
    own, so they appear once.
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
    """The callable behind a ``func`` that is no key of the resolved dispatch table.

    A miss of that table says all there is to know: ``func`` is not one of the
    functions anybody knows, so it is either a callable of the caller's own -
    which needs no hashability to be run, a dataclass instance is the everyday
    case - or it named nothing at all and says so.  Only the cold path comes
    here; the hot path is the lookup itself.
    """
    if callable(func):
        return cast("Callable[..., AScalar]", func)
    raise ValueError(f"func {func} is neither a valid function string nor a callable object")


def build_dispatch(
    op_dict: Mapping[str, Callable[..., Any]],
    aliasing: Mapping[Any, str],
    temporal_op_dict: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[Any, Resolved]:
    """Fuse an aliasing and the implementation tables for single-lookup dispatch.

    Every alias of every known function is a key, holding the canonical name
    of the function and the two operations of it - the numeric one and the
    temporal one - so naming a function is one subscript and a func missing
    from the table is not a function anybody knows (``custom_callable``).  An
    alias whose name this backend does not implement holds ``(name, None)``,
    which the caller reports with ``missing_impl_error`` - so the table
    answers that question too, without a second lookup.

    The temporal operation is None where the backend has none: the numeric
    operation of such a function answers the temporal input on its own.
    """
    return {
        alias: Resolved(name, op_dict.get(name), (temporal_op_dict or {}).get(name)) for alias, name in aliasing.items()
    }


class Resolved(NamedTuple):
    """What a func resolved to: the func it names and the operations that do the work.

    The funcname is the canonical name of a known function, or the callable of
    the caller's own - which is what tells the two apart.  The operation is
    None where the backend knows the function but does not implement it.

    The temporal operation is the other half of the operation of a function:
    it answers the datetime64 and the timedelta64 input of it, and it is
    None where the backend has none - the numeric operation of such a
    function answers the temporal input on its own.
    """

    funcname: str | Callable[..., AScalar]
    op: Callable[..., Any] | None
    temporal_op: Callable[..., Any] | None = None


class Registry(dict[str | Callable[..., object], Resolved]):
    """The registry of one backend: the functions it knows and the callables of the caller.

    Every alias of every known function is a key of the table, holding the
    canonical name and the implementation of this backend - None where the
    backend does not implement the function.  A func that names no known
    function is resolved by the custom factory of the backend, which turns a
    callable of the caller's own into an operation of the same shape.  One
    lookup therefore answers both kinds of func, and the caller does not have
    to know which kind it got.

    The lookup is a plain subscript, so that the hot path pays for a dict
    lookup and nothing else: the factory is reached through ``__missing__``,
    which the dict calls for a key it does not hold.  A func that cannot be a
    key of a dict at all - an object that defines ``__eq__`` without
    ``__hash__`` - is answered by ``custom``, which the caller reaches when
    the subscript raises.

    The temporal operations of the backend live in the registry as well: the
    operation of a function for the datetime64 and timedelta64 input, built
    like the numeric one and put beside it.  They are no aliases of the
    public functions - they are the other half of the operation of a
    function, and they are reached by the same lookup as the numeric one.
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
        """The operation of a func that names no known function.

        The dict calls this for a key it does not hold, so that a callable of
        the caller's own is resolved by the same lookup that resolves a name.
        """
        return self._custom_factory(key)

    def custom(self, func: Any) -> Resolved:
        """The operation of a func that the table cannot hold as a key.

        A dict asks its keys for a hash, which an object that defines ``__eq__``
        without ``__hash__`` does not have - such a func names no known
        function either, and is answered by the same factory.
        """
        return self._custom_factory(func)
