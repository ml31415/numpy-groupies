"""Function registry, aliasing and fill-value logic shared by every implementation.

This module must not import numpy - the pure python implementation is loaded
before numpy is known to be available, and has to work without it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, SupportsInt, TypeAlias, cast

from numpy_groupies.validation import FILL_NOT_BOOLEAN

if TYPE_CHECKING:
    import numpy.typing as npt

    from numpy_groupies.utils_np import AFillValue, AScalar

nan = float("nan")  # nan is just this float

# integers beyond this magnitude are not exactly representable in float64, so
# once an exact integer sum can leave this window, a float64-accumulated mean
# silently rounds (2**53 itself is still exact, hence the inclusive limit)
EXACT_FLOAT64_INTEGER_LIMIT = 2**53
# largest magnitude a two's-complement signed 64-bit deviation represents
# exactly; integer variance centers each group on its first value and needs
# every deviation to stay within this bound
INT64_DEVIATION_LIMIT = 2**63 - 1
INT64_MIN = -(2**63)


def check_integer_mean_overflow(count: int, bound: int) -> None:
    """Refuse integer means that float64 cannot compute exactly.

    The mean of ``count`` integers of magnitude up to ``bound`` is exact in
    float64 only while their sums stay inside the window float64 represents
    exactly.  Every implementation applies this rule, so the backends never
    disagree silently on inexact integer means.
    """
    if count * bound > EXACT_FLOAT64_INTEGER_LIMIT:
        raise OverflowError(
            f"cannot compute an exact mean of {count} integer values with magnitude up to {bound}: "
            f"their sums can exceed +/-{EXACT_FLOAT64_INTEGER_LIMIT}, the range float64 represents "
            f"exactly - cast the input to a float dtype to accept rounding, or use sum and len"
        )


def check_integer_variance_span_overflow(low: int, high: int) -> None:
    """Refuse integer variances whose exact centering would overflow int64.

    The deviations of the values of a group from its first value have to stay
    inside the signed 64-bit range for the variance to be exact.
    """
    if high - low > INT64_DEVIATION_LIMIT:
        raise OverflowError(
            f"cannot compute an exact variance: the integer values span [{low}, {high}], wider than "
            f"+/-{INT64_DEVIATION_LIMIT}, so centering each group on its first value would overflow "
            f"the signed 64-bit deviation - cast the input to a float dtype first"
        )


aggregate_common_doc = """
    See readme file at https://github.com/ml31415/numpy-groupies for a full
    description.  Below we reproduce the "Full description of inputs"
    section from that readme, note that the text below makes references to
    other portions of the readme that are not shown here.

    group_idx:
        this is an array of non-negative integers, to be used as the "labels"
        with which to group the values in ``a``. Although we have so far
        assumed that ``group_idx`` is one-dimensional, and the same length as
        ``a``, it can in fact be two-dimensional (or some form of nested
        sequences that can be converted to 2D).  When ``group_idx`` is 2D, the
        size of the 0th dimension corresponds to the number of dimensions in
        the output, i.e. ``group_idx[i,j]`` gives the index into the ith
        dimension in the output
        for ``a[j]``.  Note that ``a`` should still be 1D (or scalar), with
        length matching ``group_idx.shape[1]``.
    a:
        this is the array of values to be aggregated.  See above for a
        simple demonstration of what this means.  ``a`` will normally be a
        one-dimensional array, however it can also be a scalar in some cases.
    func: default='sum'
        the function to use for aggregation.  See the section above for
        details. Note that the simplest way to specify the function is using a
        string (e.g. ``func='max'``) however a number of aliases are also
        defined (e.g. you can use the ``func=np.max``, or even ``func=max``,
        where ``max`` is the
        builtin function).  To check the available aliases see ``utils.py``.
    size: default=None
        the shape of the output array. If ``None``, the maximum value in
        ``group_idx`` will set the size of the output.  Note that for
        multidimensional output you need to list the size of each dimension
        here, or give ``None``.
    fill_value: default=DEFAULT_FILL_VALUE
        in the example above, group 2 does not have any data, so requires some
        kind of filling value.  By default the value is chosen per function,
        following what the corresponding numpy function returns for an empty
        slice: ``0`` for sums and counts, ``1`` for products, ``False`` for the
        boolean functions, ``nan`` for the averaging ones (mean, median, var,
        std, and min/max/first/last on floating input), ``0`` for the
        trapezoidal integral (as it is for a single sample), ``-1`` for
        argmax/argmin, and an empty sequence for array/sort.  Integer output
        cannot hold ``nan``, so those functions fall back to ``0``.  For complex
        output the default ``nan`` fills with a nan in *both* parts, so that an
        empty group looks the same in every implementation.  Use
        ``utils.default_fill_value(func, dtype)`` to query the value, or pass
        your own.  ``None`` is a value and not a request: it fills the object
        result of ``array`` and is refused for every other function, since the
        request for a default is made with ``DEFAULT_FILL_VALUE``.  Note that
        there are some subtle interactions between what is
        permitted for ``fill_value`` and the input/output ``dtype`` - exceptions
        should be raised in most cases to alert the programmer if issues arise.
    order: default='C'
        this is relevant only for multidimensional output.  It controls the
        layout of the output array in memory, can be ``'F'`` for fortran-style.
    dtype: default=None
        the ``dtype`` of the output.  By default something sensible is chosen
        based on the input, aggregation function, and ``fill_value``.
        Complex input keeps the complex dtype wherever the result is complex
        (sum, prod, mean, median, trapezoid, first, last, sort, cumsum), while
        var, std and sumofsquares measure squared magnitudes and return a real
        dtype - matching numpy.  Requesting a non-complex ``dtype`` for complex
        input is refused for every function that does return a complex result,
        since the imaginary part would be lost.  The order-dependent functions
        (min, max, argmax, argmin and their ``nan`` counterparts) follow numpy's
        lexicographic ordering - the real part decides, the imaginary part only
        breaks a tie - in every implementation, including numba, which compares
        the two parts itself because neither python nor numba orders complex
        numbers.
    axis: default=None
        allows aggregation to be performed along a single axis of a
        multi-dimensional array ``a``.  In that case ``group_idx`` must be 1D
        with length matching ``a.shape[axis]``, or have the same
        dimensionality as ``a`` with a shape broadcastable to ``a.shape``
        (e.g. to use separate group labels for each row).  In either case the
        groups are broadcast out along the remaining axes of ``a``.  Not
        supported by the pure python implementation.
    reverse: default=False
        only relevant for ``func='sort'`` - sorts the items within each group
        in descending order instead of ascending.
    ddof: default=0
        passed through into calculations of variance and standard deviation
        (see above).
    dx: default=1.0
        passed through into the calculation of the trapezoidal integral
        ``trapezoid`` (see above), where it is the sample spacing.
    q: default=0.5
        the quantile to compute - a number between 0 and 1 for
        ``func='quantile'``, between 0 and 100 for ``func='percentile'``.  The
        median is the quantile 0.5, which is what the two functions answer
        when no quantile is asked for.
    method: default='linear'
        how to interpolate between the two values a quantile falls between:
        'linear', 'lower', 'higher', 'midpoint' or 'nearest', as in
        ``np.quantile``.
"""

funcs_common = [
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
    "allnan",
    "anynan",
    "max",
    "min",
    "argmax",
    "argmin",
    "sumofsquares",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
]
funcs_no_separate_nan = frozenset(["sort", "array", "allnan", "anynan"])

# the functions that collect the values of a group instead of reducing them to
# one number: their result is a sequence per group, which the comparisons that
# work on numbers cannot take in one go
funcs_collecting = frozenset(["array"])

# the names the known functions answer to - their canonical names and the
# string aliases of the alias table.  The literal type names them for the
# type checker (a typo is a type error) and for the editor.
# fmt: off
FuncName: TypeAlias = Literal[
    "add", "all", "allnan", "amax", "amin", "and",
    "any", "anynan", "argmax", "argmin", "array", "asort",
    "asorted", "count", "cummax", "cummin", "cumprod", "cumsum",
    "dsort", "first", "last", "len", "max", "maximum", "nannunique",
    "mean", "median", "min", "minimum", "multiply", "nanall",
    "nanany", "nanargmax", "nanargmin", "nancummax", "nancummin", "nancumprod",
    "nancumsum", "nanfirst", "nanlast", "nanlen", "nanmax", "nanmean",
    "nanmedian", "nanmin", "nanpercentile", "nanprod", "nanquantile",
    "nanskew", "nankurt", "nanstd", "nansum", "nansumofsquares",
    "nantrapezoid", "nanvar", "or", "skew", "kurt", "kurtosis",
    "percentile", "plus", "prod", "product", "quantile",
    "rsorted", "sort", "sorted", "splice", "split", "std",
    "sum", "sumofsquares", "times", "trapezoid", "var", "nunique",
]
# fmt: on


class _DefaultFillValue:
    """Sentinel for ``fill_value``, asking for the function specific default."""

    def __repr__(self) -> str:
        return "DEFAULT_FILL_VALUE"


DEFAULT_FILL_VALUE = _DefaultFillValue()

# The value an absent group gets, per aggregation function, following the
# convention that it is whatever numpy returns for an empty slice - where the
# output datatype can represent it (see ``resolve_fill_value``).
_default_fill_values: dict[str, AFillValue] = {
    "sum": 0,
    "prod": 1,
    "all": False,
    "any": False,
    "allnan": False,
    "anynan": False,
    "len": 0,
    "nunique": 0,
    "mean": nan,
    "median": nan,
    "quantile": nan,
    "percentile": nan,
    # an empty domain has a zero integral, just like a single sample does
    "trapezoid": 0,
    "var": nan,
    "std": nan,
    "skew": nan,
    "kurt": nan,
    "min": nan,
    "max": nan,
    "first": nan,
    "last": nan,
    "argmax": -1,
    "argmin": -1,
    "sumofsquares": 0,
    "cumsum": 0,
    "cumprod": 1,
    "cummax": 0,
    "cummin": 0,
    # 'array' and 'sort' ask for an empty sequence, which their implementations
    # interpret as 'leave the empty group as it is' - a fresh list per call is
    # handed out below so that callers cannot alias each other's results.
    "array": [],
    "sort": [],
}


def _fill_value_key(func: str | Callable[..., AScalar]) -> str:
    """Map a function (name, alias or callable) onto a ``_default_fill_values`` key."""
    name = resolve_name(func)
    if name is None:
        # a custom callable is keyed by the name it carries, which it usually
        # does not carry at all (a lambda, a bound method of one's own)
        name = getattr(func, "__name__", "")
    if name not in _default_fill_values and str(name).startswith("nan"):
        # nan-variants fill like their plain counterparts
        name = name[3:]
    return name


def _is_inexact_dtype(dtype: npt.DTypeLike | None) -> bool:
    """True for everything that can hold nan - numpy dtypes by kind, python types by identity."""
    kind = getattr(dtype, "kind", None)
    if kind is not None:  # a numpy dtype
        return kind in "fc"
    if isinstance(dtype, str):
        return dtype.startswith(("float", "complex"))
    return isinstance(dtype, type) and issubclass(dtype, (float, complex))


def _is_complex_dtype(dtype: npt.DTypeLike | None) -> bool:
    """True for complex numpy dtypes, the complex builtin and complex dtype strings."""
    kind: str | None = getattr(dtype, "kind", None)  # the kind of a numpy dtype is a one letter string
    if kind is not None:  # a numpy dtype
        return kind == "c"
    if isinstance(dtype, str):
        return dtype.startswith("complex")
    # NumPy's complex64 and clongdouble do not inherit from Python's complex.
    # Inspect the public scalar base without importing the optional dependency.
    return isinstance(dtype, type) and any(
        base is complex or (base.__module__ == "numpy" and base.__name__ == "complexfloating") for base in dtype.__mro__
    )


def nan_of(dtype: npt.DTypeLike | None) -> AScalar:
    """A nan shaped like ``dtype``: a nan in every part it has.

    A plain ``nan`` written into a complex result lands as ``nan+0j``, which
    states that the part the nan never touched was a zero - and a group that was
    poisoned by one of its values never stated anything about its parts at all.
    The default of a function for a complex result takes this shape, and so does
    the poisoning of a group, for the same reason.
    """
    return nan + nan * 1j if _is_complex_dtype(dtype) else nan


# the functions whose result is settled by the function and holds no ``None``:
# everything the table of defaults names, except the one function that returns
# an object array - a sort returns the values of a group, so it takes their
# dtype, and a count or a flag holds no ``None`` either
_no_none_result_functions = frozenset(_default_fill_values) - {"array"}


def check_filling_with_none(func: str | Callable[..., AScalar]) -> None:
    """Refuse to fill with ``None`` where no result can hold it.

    ``None`` is a value and not a request: the request for the default of a
    function is made with ``DEFAULT_FILL_VALUE``, and ``None`` is what an object
    result - the sequence of ``func='array'`` - is filled with, nothing else.  A
    number, a flag, a count, a position and the values a sort returns cannot hold
    a ``None``, and the three implementations answered the request three
    different ways: the numpy one complained about an ``isnan`` of an object, the
    numba one failed to type a kernel, and the pure python one wrote the ``None``
    into a column of floats, which turned the result into an object array.
    """
    if _fill_value_key(func) in _no_none_result_functions:
        raise TypeError(
            "fill_value=None fills an object result, which is what func='array' returns and no other function does (the default of a function is asked for with DEFAULT_FILL_VALUE)"
        )


def default_fill_value(func: FuncName | Callable[..., AScalar], dtype: npt.DTypeLike | None = None) -> AFillValue:
    """The value ``aggregate`` uses for groups missing from ``group_idx``.

    ``func`` may be a name, an alias or a callable.  Pass the ``dtype`` of your
    actual output to get the value that would really be used, ``None`` assumes
    a floating result (which is what the averaging functions give anyway).
    """
    return resolve_fill_value(func, DEFAULT_FILL_VALUE, float if dtype is None else dtype)


def resolve_fill_value(
    func: str | Callable[..., AScalar], fill_value: AFillValue, dtype: npt.DTypeLike | None
) -> AFillValue:
    """Replace the ``DEFAULT_FILL_VALUE`` sentinel with the default of ``func``.

    Anything else than the sentinel is returned untouched, except a ``None``: a
    ``None`` is refused where no result can hold it.  ``nan`` is only
    handed out if ``dtype`` can represent it, otherwise the value falls back to
    ``0`` - changing the output datatype of e.g. an integer ``min`` would be a
    worse surprise than an unspectacular filling value.
    """
    if fill_value is None:
        check_filling_with_none(func)
    if fill_value is not DEFAULT_FILL_VALUE:
        return fill_value
    key = _fill_value_key(func)
    # nan only fits where the output can hold it - the averaging and
    # dispersion functions coerce their result to a float type anyway
    fits = dtype is None or key in _forced_float_types or key in _forced_real_float_types or _is_inexact_dtype(dtype)
    value = _default_fill_values.get(key)
    if value is None:
        # a custom callable has no function specific default
        value = nan if fits else 0
    elif value != value and not fits:
        value = 0
    if (
        isinstance(value, float)
        and value != value
        and dtype is not None
        and _is_complex_dtype(dtype)
        and key not in _forced_real_float_types
    ):
        # a nan of a complex output has to be a nan in both parts - the numpy
        # kernels divide an empty sum by zero and so get nan+nanj for free,
        # and filling with nan+0j would make the backends disagree
        value = nan_of(dtype)
    return list(value) if isinstance(value, list) else value


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
    op_dict: Mapping[str, Callable[..., Any]], aliasing: Mapping[Any, str]
) -> dict[Any, tuple[str, Callable[..., Any] | None]]:
    """Fuse an aliasing and an implementation table for single-lookup dispatch.

    Every alias of every known function is a key, holding a
    ``(canonical name, implementation)`` tuple, so naming a function is one
    subscript and a func missing from the table is not a function anybody knows
    (``custom_callable``).  An alias whose name this backend does not implement
    holds ``(name, None)``, which the caller reports with ``missing_impl_error``
    - so the table answers that question too, without a second lookup.
    """
    return {alias: (name, op_dict.get(name)) for alias, name in aliasing.items()}


class Resolved(NamedTuple):
    """What a func resolved to: the func it names and the operation that does the work.

    The funcname is the canonical name of a known function, or the callable of
    the caller's own - which is what tells the two apart.  The operation is
    None where the backend knows the function but does not implement it.
    """

    funcname: str | Callable[..., AScalar]
    op: Callable[..., Any] | None


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
    """

    def __init__(
        self,
        op_dict: Mapping[str, Callable[..., Any]],
        aliasing: Mapping[str | Callable[..., object], str],
        custom: Callable[..., Resolved],
    ) -> None:
        super().__init__({alias: Resolved(name, op) for alias, (name, op) in build_dispatch(op_dict, aliasing).items()})
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


_forced_float_types = {
    "mean",
    "median",
    "skew",
    "kurt",
    "nanskew",
    "nankurt",
    "quantile",
    "percentile",
    "trapezoid",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nanpercentile",
    "nantrapezoid",
}

# var/std (and sumofsquares) measure the squared magnitude of the deviations,
# which is real even for complex input - like np.var returning a real dtype
_forced_real_float_types = {
    "var",
    "std",
    "nanvar",
    "nanstd",
    "sumofsquares",
    "nansumofsquares",
}


def check_boolean(x: AFillValue) -> None:
    if x not in (0, 1):
        raise ValueError(FILL_NOT_BOOLEAN)


# functions whose result type the function settles rather than taking it from
# the values aggregated: a boolean answer, a count, a position, and the object
# array of ``array``.  The python counterpart of each result type is here, where
# no numpy is imported; the numpy dtypes of them in utils, which also knows how
# wide a count is on the platform it runs on
_fixed_result_types: dict[str, type] = {
    "array": object,
    "all": bool,
    "any": bool,
    "nanall": bool,
    "nanany": bool,
    "allnan": bool,
    "anynan": bool,
    "len": int,
    "nanlen": int,
    "nunique": int,
    "nannunique": int,
    "argmax": int,
    "argmin": int,
    "nanargmin": int,
    "nanargmax": int,
}


def check_fixed_fill_value(func: str | Callable[..., AScalar], fill_value: AFillValue) -> None:
    """Refuse a ``fill_value`` that the result type of ``func`` cannot hold, for
    those functions whose result type the function itself settles.

    This is the rule ``check_fill_value`` tells against a numpy dtype, told
    without one: a boolean answer is filled with 0 or 1 - a nan would pass for
    ``bool``, which is why the two values are asked for themselves - and a count
    or a position with whatever turns into an integer, which a nan does not.
    """
    target = _fixed_result_types.get(_fill_value_key(func))
    if target is bool:
        check_boolean(fill_value)
    elif target is int:
        try:
            int(cast("SupportsInt", fill_value))
        except ValueError:
            raise ValueError("fill_value must be convertible into an integer")


# functions whose result is not complex even for complex input - only these may
# be asked for a non-complex dtype, everything else would throw the imaginary
# part of the values away
# the names of the dtype table in utils (which holds the numpy dtypes
# themselves) - everything here may aggregate complex input into a
# non-complex dtype without throwing the imaginary part away
_complex_real_result = (
    frozenset(
        [
            "array",
            "all",
            "any",
            "nanall",
            "nanany",
            "len",
            "nanlen",
            "allnan",
            "anynan",
            "argmax",
            "argmin",
            "nanargmin",
            "nanargmax",
        ]
    )
    | _forced_real_float_types
)


def check_complex_dtype(
    a_dtype: npt.DTypeLike, dtype: npt.DTypeLike | None, func_str: str | Callable[..., AScalar]
) -> None:
    """Complain when a non-complex ``dtype`` is requested for complex input.

    ``resolve_output_dtype`` applies this for the backends that honour ``dtype``;
    the pure python implementation needs it on its own, since python numbers
    carry no dtype and it would otherwise silently ignore the request.
    """
    if (
        dtype is not None
        and _is_complex_dtype(a_dtype)
        and not _is_complex_dtype(dtype)
        and func_str not in _complex_real_result
    ):
        raise TypeError(
            f"function {func_str} cannot aggregate complex values into the non-complex dtype "
            f"{dtype} - pass a complex dtype to keep the imaginary part"
        )


def is_scalar_input(a: Any) -> bool:
    """Whether the aggregated data is a single value instead of a sequence.

    Kept numpy free on purpose, since the pure python implementation uses it
    too: a python number has no ``ndim`` at all, while a numpy scalar and a 0d
    array both report 0 - ``isinstance(a, (int, float, complex))`` alone misses
    every numpy scalar that is not a python number (``np.float32``, ``np.int64``,
    ``np.bool_``, ``np.complex64``).
    """
    return getattr(a, "ndim", None) == 0 or isinstance(a, (int, float, complex))


# the arguments a function has of its own, beyond the size and the fill value
# that every function takes: the name it carries in the signature, the
# annotation and the default.  The njit kernels and the python entry points
# of a backend take the same arguments, so both read them from here.
FUNC_EXTRAS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "std": (("ddof", "int", 0),),
    "var": (("ddof", "int", 0),),
    "quantile": (("q", "float", 0.5),),
    "percentile": (("q", "float", 50.0),),
    "trapezoid": (("dx", "float", 1.0),),
    "sort": (("reverse", "bool", False),),
}


# the marker for an argument without a default - None is a default of its own
_NO_DEFAULT = object()


def func_extras(name: str) -> tuple[tuple[str, str, Any], ...]:
    """The arguments the function has of its own.

    The nan-skipping variant of a function has the same arguments as the
    plain one, so the name is stripped of its prefix before the lookup.
    """
    return FUNC_EXTRAS.get(name.removeprefix("nan"), ())


class Backend:
    """The aggregate of one backend: callable like the function it replaces,
    with the entry point of every function as its attribute.

    ``backend(group_idx, a, func="mean")`` does what it always did, and
    ``backend.mean(group_idx, a)`` is the same thing with the function in
    the name - a plain python function of the backend, in every backend.  The
    kernels of the numba implementation are the attributes of its module
    (``numpy_groupies.aggregate_numba``), which is the one place jitted code
    can reach them.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        functions: Mapping[str, Callable[..., Any]],
        name: str = "",
        doc: str | None = None,
    ) -> None:
        self._func = func
        self._functions = dict(functions)
        self._name = name
        if doc is not None:
            self.__doc__ = doc

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # the attributes are the functions of the backend - they are looked
        # up in the dict instead of being set one by one, so that the object
        # stays small and the names cannot drift from the dict
        try:
            return self._functions[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}") from None

    def __dir__(self) -> list[str]:
        return sorted(self._functions)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self._name} with {len(self._functions)} functions>"


def bind_functions(
    func: Callable[..., Any],
    names: Iterable[str],
    doc: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """One entry point per function name, with the function in the name.

    The entry points call the backend's ``aggregate`` with the name of their
    function, and each of them carries the arguments its function has of its
    own - the ``ddof`` of the variance, the ``q`` of the quantile, the ``dx``
    of the trapezoid, the ``reverse`` of the sort.
    """
    return {name: _bind_one(func, name, doc) for name in names}


def _bind_one(func: Callable[..., Any], name: str, doc: str | None) -> Callable[..., Any]:
    """The entry point of one function of a backend.

    The signature is spelled out from the description of the function's own
    arguments, so that the entry point of the quantile takes its ``q`` and
    the entry point of the variance its ``ddof``.
    """
    arguments: list[tuple[str, str, Any]] = [
        ("group_idx", "npt.NDArray[np.integer]", _NO_DEFAULT),
        ("a", "AValue", _NO_DEFAULT),
    ]
    arguments += [(extra, annotation, default) for extra, annotation, default in FUNC_EXTRAS.get(name, ())]
    arguments += [
        ("size", "int | Sequence[int] | None", None),
        ("fill_value", "AFillValue", DEFAULT_FILL_VALUE),
        ("order", "RavelOrder", '"C"'),
        ("dtype", "npt.DTypeLike | None", None),
        ("axis", "int | None", None),
    ]
    signature = ", ".join(
        n + ": " + ann + ("" if default is _NO_DEFAULT else f" = {default}") for n, ann, default in arguments
    )
    # the arguments of their own go to the call of the backend
    own = "".join(f", {n}={n}" for n, _, _ in FUNC_EXTRAS.get(name, ()))
    source = f"""
def _entry_point({signature}):
    return _func(group_idx, a, func=_name, size=size, fill_value=fill_value,
                 order=order, dtype=dtype, axis=axis{own})
"""
    namespace: dict[str, Any] = {"_func": func, "_name": name, "DEFAULT_FILL_VALUE": DEFAULT_FILL_VALUE}
    exec(source, namespace)  # noqa: S102 - the signature is spelled out in one place
    entry_point: Callable[..., Any] = namespace["_entry_point"]
    entry_point.__name__ = name
    entry_point.__qualname__ = name
    entry_point.__module__ = __name__
    entry_point.__doc__ = doc or f"The {name} of every group - see the documentation of the package."
    return entry_point


def function_names() -> list[str]:
    """The names of the functions of the package: the plain ones and the
    nan-skipping ones beside them.

    The four functions that do not skip nans have no nan variant of their
    own, so they appear once.
    """
    names = list(funcs_common)
    names += ["nan" + name for name in funcs_common if name not in funcs_no_separate_nan]
    return names
