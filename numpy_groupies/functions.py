"""The properties of the functions of the package, and the entry points a
backend offers for them.

Per function: the aliases (via the registry), the default fill value, the
result type and dtype, and its own arguments - kept here so implementations
can ask for them without knowing of each other.  Also the binding of names
to the aggregate of a backend.

This module must not import numpy: the pure python implementation is loaded
before numpy is known to be available, and has to work without it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from numpy_groupies.registry import resolve_name

if TYPE_CHECKING:
    from numpy_groupies.typing import AFillValue, AScalar

nan = float("nan")  # nan is just this float

# the value an absent group gets, per function - whatever numpy returns for
# an empty slice, where the output dtype can represent it
# (see ``numpy_groupies.fills``)
default_fill_values: dict[str, AFillValue] = {
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
    # a fresh list per call so callers cannot alias each other's results
    "array": [],
    "sort": [],
    # fills with the dtype default: nan where the output can hold one, else 0
    "topk": nan,
    # a group never seen, or holding only missing values, has no mode
    "mode": nan,
}


# functions answering with a float dtype whatever the input: averaging and
# dispersion
forced_float_types = {
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

# var/std (and sumofsquares) measure squared magnitudes, real even for
# complex input - like np.var returning a real dtype
forced_real_float_types = {
    "var",
    "std",
    "nanvar",
    "nanstd",
    "sumofsquares",
    "nansumofsquares",
}


# functions whose result type the function settles rather than taking it
# from the input values.  Python types here (no numpy is imported); the
# numpy dtypes live in ``numpy_groupies.dtypes``
fixed_result_types: dict[str, type] = {
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


# functions whose result is not complex even for complex input - only these
# may be asked for a non-complex dtype without throwing the imaginary part
# away; the dtype tables themselves live in ``numpy_groupies.dtypes``
complex_real_result = (
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
    | forced_real_float_types
)


# a function's own arguments beyond size and fill value: name, annotation,
# default.  The njit kernels and the python entry points read them from here.
FUNC_EXTRAS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "std": (("ddof", "int", 0),),
    "var": (("ddof", "int", 0),),
    "quantile": (("q", "float", 0.5), ("method", "str", "linear")),
    "percentile": (("q", "float", 50.0), ("method", "str", "linear")),
    "trapezoid": (("dx", "float", 1.0),),
    "sort": (("reverse", "bool", False),),
    # how many values per group, largest or smallest
    "topk": (("k", "int", 5), ("largest", "bool", True)),
    # the bare temporal topk kernel on the int64 ticks - no package function,
    # but it takes the topk arguments the same way
    "tick_topk": (("k", "int", 5), ("largest", "bool", True)),
    # the bare temporal quantile kernel on the int64 ticks - no package
    # function, but it takes q and method the same way
    "tick_quantile": (("q", "float", 0.5), ("method", "str", "linear")),
}


# how a quantile interpolates between its neighbours - numpy's method names;
# shared by the backends
QUANTILE_METHODS = frozenset({"linear", "lower", "higher", "midpoint", "nearest"})


def fill_value_key(func: str | Callable[..., AScalar]) -> str:
    """Map a function (name, alias or callable) onto a ``default_fill_values`` key."""
    name = resolve_name(func)
    if name is None:
        # a custom callable is keyed by its __name__ (often absent: a lambda,
        # a bound method)
        name = getattr(func, "__name__", "")
    if name not in default_fill_values and str(name).startswith("nan"):
        # nan-variants fill like their plain counterparts
        name = name[3:]
    return name


def func_extras(name: str) -> tuple[tuple[str, str, Any], ...]:
    """The function's own arguments; nan variants share the plain form's."""
    return FUNC_EXTRAS.get(name.removeprefix("nan"), ())


def func_argument_default(func: str, argument: str) -> Any:
    """The default of one of the function's own arguments, from the table
    that is the single place knowing them."""
    for name, _annotation, default in func_extras(func):
        if name == argument:
            return default
    raise KeyError(f"{func!r} has no argument {argument!r}")


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
        builtin function).  To check the available aliases see ``registry.py``.
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
        ``common.default_fill_value(func, dtype)`` to query the value, or pass
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
    k: default=5
        only relevant for ``func='topk'`` - how many values of every group the
        answer holds.  The answer is a matrix with one row per group and k
        columns.
    largest: default=True
        only relevant for ``func='topk'`` - keeps the largest values of every
        group, or the smallest ones when it is false.
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


def bind_functions(
    func: Callable[..., Any],
    names: Iterable[str],
    doc: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """One entry point per function name, calling the backend's ``aggregate``
    with the name baked in and passing on the function's own arguments
    (``ddof``, ``q``, ``dx``, ``reverse``, ...)."""
    return {name: _bind_one(func, name, doc) for name in names}


def _bind_one(func: Callable[..., Any], name: str, doc: str | None) -> Callable[..., Any]:
    """The entry point of one function: the aggregate with ``func=name``
    baked in; the arguments behind the values are keyword arguments."""

    def entry_point(group_idx: Any, a: Any, **kwargs: Any) -> Any:
        return func(group_idx, a, func=name, **kwargs)

    entry_point.__name__ = name
    entry_point.__qualname__ = name
    entry_point.__module__ = getattr(func, "__module__", __name__)
    entry_point.__doc__ = doc or f"The {name} of every group - see the documentation of the package."
    # the signature of the aggregate, without ``func`` and with the
    # function's own arguments, so the help says what it takes
    vars(entry_point)["__signature__"] = _signature_of(func, name)
    return entry_point


def _signature_of(func: Callable[..., Any], name: str) -> inspect.Signature:
    """The aggregate's signature without ``func`` and with the function's
    own arguments added as keyword-only."""
    signature = inspect.signature(func)
    parameters: list[inspect.Parameter] = []
    behind_the_values = False
    for parameter in signature.parameters.values():
        if parameter.name == "func":
            continue
        if parameter.name == "cache":
            # a named callable has a kernel of its own and no loop to cache
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parameters.append(parameter)
            continue
        if parameter.name == "a":
            behind_the_values = True
            parameters.append(parameter)
            parameters.extend(
                inspect.Parameter(extra, inspect.Parameter.KEYWORD_ONLY, default=default)
                for extra, _annotation, default in FUNC_EXTRAS.get(name, ())
            )
            continue
        if behind_the_values:
            parameters.append(parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY))
        else:
            parameters.append(parameter)
    return signature.replace(parameters=parameters)
