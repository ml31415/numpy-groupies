"""Function registry, aliasing and fill-value logic shared by every implementation.

This module must not import numpy - the pure python implementation is loaded
before numpy is known to be available, and has to work without it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

nan = float("nan")  # nan is just this float


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
        your own.  Note that there are some subtle interactions between what is
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
"""

funcs_common = [
    "first",
    "last",
    "len",
    "mean",
    "median",
    "trapezoid",
    "var",
    "std",
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


class _DefaultFillValue:
    """Sentinel for ``fill_value``, asking for the function specific default."""

    def __repr__(self):
        return "DEFAULT_FILL_VALUE"


DEFAULT_FILL_VALUE = _DefaultFillValue()

# The value an absent group gets, per aggregation function, following the
# convention that it is whatever numpy returns for an empty slice - where the
# output datatype can represent it (see ``resolve_fill_value``).
_default_fill_values = {
    "sum": 0,
    "prod": 1,
    "all": False,
    "any": False,
    "allnan": False,
    "anynan": False,
    "len": 0,
    "mean": nan,
    "median": nan,
    # an empty domain has a zero integral, just like a single sample does
    "trapezoid": 0,
    "var": nan,
    "std": nan,
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


def _fill_value_key(func: str | Callable[..., Any]) -> str:
    """Map a function (name, alias or callable) onto a ``_default_fill_values`` key."""
    try:
        name = aliasing_py[func]
    except (KeyError, TypeError):
        name = getattr(func, "__name__", "")
    if name not in _default_fill_values and str(name).startswith("nan"):
        # nan-variants fill like their plain counterparts
        name = name[3:]
    return name


def _is_inexact_dtype(dtype: Any) -> bool:
    """True for everything that can hold nan - numpy dtypes by kind, python types by identity."""
    kind = getattr(dtype, "kind", None)
    if kind is not None:  # a numpy dtype
        return kind in "fc"
    if isinstance(dtype, str):
        return dtype.startswith(("float", "complex"))
    return isinstance(dtype, type) and issubclass(dtype, (float, complex))


def _is_complex_dtype(dtype: Any) -> bool:
    """True for complex numpy dtypes, the complex builtin and complex dtype strings."""
    kind = getattr(dtype, "kind", None)
    if kind is not None:  # a numpy dtype
        return kind == "c"
    if isinstance(dtype, str):
        return dtype.startswith("complex")
    return isinstance(dtype, type) and issubclass(dtype, complex)


def resolve_fill_value(func: str | Callable[..., Any], fill_value: Any, dtype: Any) -> Any:
    """Replace the ``DEFAULT_FILL_VALUE`` sentinel with the default of ``func``.

    Anything else than the sentinel is returned untouched.  ``nan`` is only
    handed out if ``dtype`` can represent it, otherwise the value falls back to
    ``0`` - changing the output datatype of e.g. an integer ``min`` would be a
    worse surprise than an unspectacular filling value.
    """
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
        value = nan + nan * 1j
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
}

_alias_builtin = {
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


def get_aliasing(*extra: dict[Any, str]) -> dict[Any, str]:
    """
    Assembles a dictionary that maps both strings and functions to a list of supported function names.

    Examples:
        alias['add'] = 'sum'
        alias[sorted] = 'sort'

    This function should only be called during import.
    """
    alias: dict[Any, str] = {k: k for k in funcs_common}
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


def get_func(
    func: str | Callable[..., Any], aliasing: dict[Any, Any], implementations: Any
) -> str | Callable[..., Any]:
    """Return the key of a found implementation or the func itself"""
    try:
        func_str = aliasing[func]
    except KeyError:
        if callable(func):
            return func
    else:
        if func_str in implementations:
            return func_str
        if func_str.startswith("nan") and func_str[3:] in funcs_no_separate_nan:
            raise ValueError(f"{func_str[3:]} does not have a nan-version")
        else:
            raise NotImplementedError("No such function available")
    raise ValueError(f"func {func} is neither a valid function string nor a callable object")


def build_dispatch(
    impl_dict: Mapping[str, Callable[..., Any]], aliasing: Mapping[Any, str]
) -> dict[Any, tuple[str, Callable[..., Any]]]:
    """Fuse an aliasing and an implementation table for single-lookup dispatch.

    Returns a dict mapping every alias (string or callable) onto a
    ``(canonical name, implementation)`` tuple.  Aliases without an
    implementation are omitted, so lookups missing there fall through to
    ``get_func``, which keeps handling custom callables and error reporting.
    """
    dispatch = {}
    for alias, name in aliasing.items():
        impl = impl_dict.get(name)
        if impl is not None:
            dispatch[alias] = (name, impl)
    return dispatch


_forced_float_types = {
    "mean",
    "median",
    "trapezoid",
    "nanmean",
    "nanmedian",
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


def check_complex_dtype(a_dtype: Any, dtype: Any, func_str: str) -> None:
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
