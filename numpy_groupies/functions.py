"""The properties of the functions of the package, and the entry points a
backend offers for them.

Every function of the package has a name and a set of aliases - the registry
knows them - and properties besides: the value that fills a group the
function never saw, the type of its answer, the dtype it answers with, and
the arguments it has of its own.  The properties live here, so that the
implementations can ask for them without knowing of each other.

The binding of the names to the aggregate of a backend lives here as well,
with the documentation of the arguments that the aggregate takes.

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


# The value an absent group gets, per aggregation function, following the
# convention that it is whatever numpy returns for an empty slice - where the
# output datatype can represent it (see ``numpy_groupies.fills``).
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
    # 'array' and 'sort' ask for an empty sequence, which their implementations
    # interpret as 'leave the empty group as it is' - a fresh list per call is
    # handed out below so that callers cannot alias each other's results.
    "array": [],
    "sort": [],
    # the topk fills the places a group does not have values for with the
    # default of the output dtype - a nan where the output can hold one, and
    # a zero where it cannot
    "topk": nan,
    # the mode answers one of the values of the group; a group it never saw,
    # and one that holds nothing but missing values, has no mode and takes
    # the default of the output dtype
    "mode": nan,
}


# the functions that answer with a float dtype, whatever the dtype of their
# input: the averaging and the dispersion ones
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

# var/std (and sumofsquares) measure the squared magnitude of the deviations,
# which is real even for complex input - like np.var returning a real dtype
forced_real_float_types = {
    "var",
    "std",
    "nanvar",
    "nanstd",
    "sumofsquares",
    "nansumofsquares",
}


# functions whose result type the function settles rather than taking it from
# the values aggregated: a boolean answer, a count, a position, and the object
# array of ``array``.  The python counterpart of each result type is here, where
# no numpy is imported; the numpy dtypes of them in ``numpy_groupies.dtypes``,
# which also knows how wide a count is on the platform it runs on
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


# functions whose result is not complex even for complex input - only these may
# be asked for a non-complex dtype, everything else would throw the imaginary
# part of the values away
# the names of the dtype table in ``numpy_groupies.dtypes`` (which holds the
# numpy dtypes themselves) - everything here may aggregate complex input into
# a non-complex dtype without throwing the imaginary part away
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


# the arguments a function has of its own, beyond the size and the fill value
# that every function takes: the name it carries in the signature, the
# annotation and the default.  The njit kernels and the python entry points
# of a backend take the same arguments, so both read them from here.
FUNC_EXTRAS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "std": (("ddof", "int", 0),),
    "var": (("ddof", "int", 0),),
    "quantile": (("q", "float", 0.5), ("method", "str", "linear")),
    "percentile": (("q", "float", 50.0), ("method", "str", "linear")),
    "trapezoid": (("dx", "float", 1.0),),
    "sort": (("reverse", "bool", False),),
    # the topk: how many values of every group it keeps, and whether it keeps
    # the largest ones or the smallest
    "topk": (("k", "int", 5), ("largest", "bool", True)),
    # the bare kernel of the topk of the ticks - the topk of a temporal
    # input, on the int64 ticks of its values.  It is no function of the
    # package - it is the kernel behind its temporal answer - but it takes
    # the arguments of the topk the same way.
    "tick_topk": (("k", "int", 5), ("largest", "bool", True)),
    # the bare kernel of the quantile of the ticks - the quantile of a
    # temporal input, on the int64 ticks of its values.  It is no function
    # of the package - it is the kernel behind its temporal answer - but it
    # takes the quantile and the method the same way the quantile of the
    # package does.
    "tick_quantile": (("q", "float", 0.5), ("method", "str", "linear")),
}


# the ways a quantile interpolates between the two values it falls between -
# the names numpy uses for them.  The backends share the list: the numpy one
# computes the quantile of the numbers and of the ticks with it, the numba one
# the quantile of the ticks.
QUANTILE_METHODS = frozenset({"linear", "lower", "higher", "midpoint", "nearest"})


def fill_value_key(func: str | Callable[..., AScalar]) -> str:
    """Map a function (name, alias or callable) onto a ``default_fill_values`` key."""
    name = resolve_name(func)
    if name is None:
        # a custom callable is keyed by the name it carries, which it usually
        # does not carry at all (a lambda, a bound method of one's own)
        name = getattr(func, "__name__", "")
    if name not in default_fill_values and str(name).startswith("nan"):
        # nan-variants fill like their plain counterparts
        name = name[3:]
    return name


def func_extras(name: str) -> tuple[tuple[str, str, Any], ...]:
    """The arguments the function has of its own.

    The nan-skipping variant of a function has the same arguments as the
    plain one, so the name is stripped of its prefix before the lookup.
    """
    return FUNC_EXTRAS.get(name.removeprefix("nan"), ())


def func_argument_default(func: str, argument: str) -> Any:
    """The default of one argument of a function - the table holds them.

    The table is the one place that knows the arguments a function has of
    its own and their defaults, so a caller that needs one of them asks
    there instead of writing the number down a second time.
    """
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
    """One entry point per function name, with the function in the name.

    The entry points call the backend's ``aggregate`` with the name of their
    function, and pass on the arguments they are given - the ``ddof`` of the
    variance, the ``q`` of the quantile, the ``dx`` of the trapezoid, the
    ``reverse`` of the sort.
    """
    return {name: _bind_one(func, name, doc) for name in names}


def _bind_one(func: Callable[..., Any], name: str, doc: str | None) -> Callable[..., Any]:
    """The entry point of one function of a backend.

    It is the aggregate of the backend with the name of the function in it:
    the entry point of the quantile is the aggregate with ``func="quantile"``.
    The arguments behind the values are keyword arguments - the ones of the
    function itself among them.
    """

    def entry_point(group_idx: Any, a: Any, **kwargs: Any) -> Any:
        return func(group_idx, a, func=name, **kwargs)

    entry_point.__name__ = name
    entry_point.__qualname__ = name
    entry_point.__module__ = getattr(func, "__module__", __name__)
    entry_point.__doc__ = doc or f"The {name} of every group - see the documentation of the package."
    # the signature of the entry point: the aggregate's, without the name of
    # the function and with the arguments the function has of its own - so
    # that the help of the entry point says what it takes
    vars(entry_point)["__signature__"] = _signature_of(func, name)
    return entry_point


def _signature_of(func: Callable[..., Any], name: str) -> inspect.Signature:
    """The signature an entry point shows.

    It is the signature of the aggregate, without the name of the function -
    that is what the entry point carries - and with the arguments the
    function has of its own, which the entry point takes as keyword
    arguments.
    """
    signature = inspect.signature(func)
    parameters: list[inspect.Parameter] = []
    behind_the_values = False
    for parameter in signature.parameters.values():
        if parameter.name == "func":
            # the name of the function is in the entry point, not in a
            # parameter of it
            continue
        if parameter.name == "cache":
            # the cache is the policy of the compiled loop of a callable of
            # the caller - a named function has a kernel of its own and no
            # loop to cache
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
            # everything behind the values is a keyword argument
            parameters.append(parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY))
        else:
            parameters.append(parameter)
    return signature.replace(parameters=parameters)
