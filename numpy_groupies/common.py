"""The fill values of the package and the binding of the functions of a backend.

What every implementation shares: the documentation of the functions, the
fill value they use for a group they never saw, and the backend that carries
the functions of an implementation as its attributes.

This module must not import numpy - the pure python implementation is loaded
before numpy is known to be available, and has to work without it.

The names of the functions live in ``numpy_groupies.registry``, the checks of
the input of a call in ``numpy_groupies.validation``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy_groupies.registry import (
    DEFAULT_FILL_VALUE,
    FuncName,
    default_fill_values,
    fill_value_key,
    forced_float_types,
    forced_real_float_types,
)
from numpy_groupies.validation import (
    check_filling_with_none,
    is_complex_dtype,
    is_inexact_dtype,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from numpy_groupies.utils_np import AFillValue, AScalar

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


# The value an absent group gets, per aggregation function, following the
# convention that it is whatever numpy returns for an empty slice - where the
# output datatype can represent it (see ``resolve_fill_value``).
def nan_of(dtype: npt.DTypeLike | None) -> AScalar:
    """A nan shaped like ``dtype``: a nan in every part it has.

    A plain ``nan`` written into a complex result lands as ``nan+0j``, which
    states that the part the nan never touched was a zero - and a group that was
    poisoned by one of its values never stated anything about its parts at all.
    The default of a function for a complex result takes this shape, and so does
    the poisoning of a group, for the same reason.
    """
    return nan + nan * 1j if is_complex_dtype(dtype) else nan


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
        check_filling_with_none(fill_value_key(func))
    if fill_value is not DEFAULT_FILL_VALUE:
        return fill_value
    key = fill_value_key(func)
    # nan only fits where the output can hold it - the averaging and
    # dispersion functions coerce their result to a float type anyway
    fits = dtype is None or key in forced_float_types or key in forced_real_float_types or is_inexact_dtype(dtype)
    value = default_fill_values.get(key)
    if value is None:
        # a custom callable has no function specific default
        value = nan if fits else 0
    elif value != value and not fits:
        value = 0
    if (
        isinstance(value, float)
        and value != value
        and dtype is not None
        and is_complex_dtype(dtype)
        and key not in forced_real_float_types
    ):
        # a nan of a complex output has to be a nan in both parts - the numpy
        # kernels divide an empty sum by zero and so get nan+nanj for free,
        # and filling with nan+0j would make the backends disagree
        value = nan_of(dtype)
    return list(value) if isinstance(value, list) else value
