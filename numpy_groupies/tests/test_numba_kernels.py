"""Tests for the kernels of ``numpy_groupies.aggregate_numba``.

The kernels share their compiled drivers with the python-level
``aggregate()``, so value-parity with the (elsewhere exhaustively tested)
layer-2 results is the contract here; every kernel is therefore called from
jitted code *and* from python, which is the other way into the same driver.
The explicitly documented deviations (the dtype approximations of the kernels,
the fill defaults, the input checks of the python call) get their own tests.
"""

from functools import cache
from importlib import import_module
from inspect import signature

import numba as nb
import numpy as np
import pytest

from numpy_groupies import aggregate, aggregate_numba, aggregate_numpy
from numpy_groupies.aggregate_numba import compose
from numpy_groupies.registry import (
    func_extras,
)
from numpy_groupies.tests.reference import aggregate_ref
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    LENGTH_MISMATCH,
    LENGTH_MISMATCH_OR_SCALAR,
    NEGATIVE_INDICES,
    TOO_LARGE_FOR_SIZE,
)

GROUP_IDX = np.array([0, 0, 1, 1, 1, 3])  # group 2 stays empty
FLOAT_A = np.array([3.0, 1.0, np.nan, 2.0, 5.0, 4.0])
INT_A = np.array([3, 1, 7, 2, 5, 4], dtype=np.int64)
BOOL_A = np.array([True, False, True, True, False, True])

# the kernels of ``make_stub``, which is where the input checks are written - as
# many of the tests below ask them all.  ``step_count`` and ``step_indices`` are
# jitted functions with a signature of their own, ``unpack`` and ``unpack_into``
# gather a result back onto the input, the percentile is the quantile in percent
# and so an entry point of the backend, and ``aggregate`` is the python-level
# dispatcher of the module: none of them is such an entry point
_ALL_KERNELS = [
    name
    for name in aggregate_numba.__all__
    if name
    not in (
        "aggregate",
        "step_count",
        "step_indices",
        "unpack",
        "unpack_into",
        "percentile",
        "nanpercentile",
    )
]

# The input checks of a kernel are written in one place - the stub of the
# operation, which every kernel has - so the faults of every kernel are the
# faults of one of them.  The tests below therefore ask a representative set
# of the kernels, one of every family: the plain reduction, its nan variant,
# the order statistic, the sort, a function with an argument of its own (the
# quantile, the variance and the trapezoid - one per builder of a variant),
# the cumulative one, the collecting one, the count of the distinct values
# and the arg reduction.
#
# Every kernel of the backend is asked once whether it refuses the empty
# labels - the smoke test of the whole family, which a new kernel joins
# without anyone adding it to a list.
_FAULT_KERNELS = [
    "sum",
    "nansum",
    "median",
    "sort",
    "quantile",
    "var",
    "trapezoid",
    "cumsum",
    "array",
    "nunique",
    "argmax",
]

# (kernel name, kwargs, values) - the kernels run inside a jitted caller and as a
# plain python call, and both are compared against the python-level
# aggregate() of the same name
_FLOAT_OPS = [
    ("sum", {}),
    ("nansum", {}),
    ("prod", {}),
    ("nanprod", {}),
    ("mean", {}),
    ("nanmean", {}),
    ("var", {"ddof": 1}),
    ("nanvar", {"ddof": 1}),
    ("std", {}),
    ("nanstd", {}),
    ("min", {}),
    ("nanmin", {}),
    ("max", {}),
    ("nanmax", {}),
    ("first", {}),
    ("nanfirst", {}),
    ("last", {}),
    ("nanlast", {}),
    ("median", {}),
    ("nanmedian", {}),
    ("quantile", {}),
    ("nanquantile", {}),
    ("quantile", {"q": 0.9}),
    ("nanquantile", {"q": 0.9}),
    ("trapezoid", {"dx": 1.0}),
    ("nantrapezoid", {"dx": 1.0}),
    ("sumofsquares", {}),
    ("nansumofsquares", {}),
    ("nunique", {}),
    ("nannunique", {}),
    ("argmin", {}),
    ("nanargmin", {}),
    ("argmax", {}),
    ("nanargmax", {}),
    ("cumsum", {}),
    ("nancumsum", {}),
    ("cummax", {}),
    ("nancummax", {}),
    ("sort", {}),
    ("sort", {"reverse": True}),
]
_INT_OPS = [
    ("sum", {}),
    ("prod", {}),
    ("len", {}),
    ("nanlen", {}),
    ("nunique", {}),
    ("nannunique", {}),
    ("min", {}),
    ("max", {}),
    ("first", {}),
    ("last", {}),
    ("median", {}),
    ("quantile", {}),
    ("argmin", {}),
    ("argmax", {}),
    ("cumsum", {}),
    ("cumprod", {}),
    ("cummax", {}),
    ("cummin", {}),
    ("sort", {}),
    ("sort", {"reverse": True}),
]
_BOOL_OPS = [
    ("all", {}),
    ("nanall", {}),
    ("any", {}),
    ("nanany", {}),
    ("sort", {}),
    ("sort", {"reverse": True}),
]


def _kernel_arguments(name, kwargs):
    """The argument the function of ``name`` has of its own, as a tuple.

    Every kernel carries the signature of its function, so the argument it has
    of its own comes first, the size and the fill value after it.  It is always
    passed, with its default if the test does not ask for another one.
    """
    extras = func_extras(name)
    own = next(iter(kwargs.values()), None) if extras else None
    if own is None and extras:
        own = extras[0][2]
    return () if own is None else (own,)


# The outward signature of a kernel is one: the python call and the jitted
# call take the same arguments under the same names.  It is the stub of the
# kernel that spells it out for python and the typing function of its overload
# that spells it out for jitted code - numba binds the arguments of a jitted
# call against the signature of that typing function - and the two have to be
# the same signature, down to the name of the argument an operation has of its
# own.  An operation without such an argument has no extra argument at all.
def test_the_python_and_the_jitted_signature_of_a_kernel_are_one():
    for name, typing_function in compose._TYPING_FUNCTIONS.items():
        stub = getattr(aggregate_numba, name)
        assert signature(stub) == signature(typing_function), name


def test_a_kernel_carries_the_arguments_of_its_function_and_no_others():
    for name, typing_function in compose._TYPING_FUNCTIONS.items():
        extras = [extra for extra, _annotation, _default in func_extras(name)]
        expected = ["group_idx", "a", *extras, "size", "fill_value"]
        assert list(signature(typing_function).parameters) == expected, name


def test_the_argument_of_a_kernel_is_the_one_of_its_function():
    """The argument an operation has of its own is the one of the function.

    Same name, same annotation and same default - the kernel is the function
    in a compiled dress, and a caller that reads the signature of one of them
    reads the signature of the other.  The one exception is the sample
    spacing of the trapezoid, which the kernel carries as the integer 1 and
    not as the float 1.0 (see ``compose._KERNEL_ARGUMENT_DEFAULTS``).
    """
    for name, typing_function in compose._TYPING_FUNCTIONS.items():
        parameters = signature(typing_function).parameters
        for extra, annotation, default in func_extras(name):
            parameter = parameters[extra]
            assert parameter.annotation.__name__ == annotation, name
            expected = compose._KERNEL_ARGUMENT_DEFAULTS.get(extra, default)
            assert parameter.default == expected, name
            assert type(parameter.default) is type(expected), name


def _make_jitted_caller(kernel, name):
    """A jitted caller of one kernel, with the name the kernel has.

    A caller that reaches a kernel from jitted code is a dispatcher of its
    own, and a dispatcher compiles the first time it is called.  The callers
    are therefore built once per kernel and shared by the tests - built per
    test, every test would compile its own copy of the very same wrapper,
    which is the most expensive thing the suite does.

    Every caller carries the name of the kernel it calls: two callers that
    differ only in their closure share the symbol numba mangles their code
    into, and one of them may bind the calls of the other - the same types
    and the wrong kernel.
    """

    def caller(group_idx, values, size):
        return kernel(group_idx, values, size=size, fill_value=None)

    caller.__qualname__ = caller.__name__ = name
    return nb.njit(caller)


def _make_jitted_caller_with_extra(kernel, extra, name):
    """A jitted caller of one kernel, with the argument of its own baked in.

    The kernel of a function that has an argument of its own takes it behind
    the two arrays, and the value of it is a constant of the caller - the
    kernel is called with it, not with a parameter of its own.
    """
    if extra is None:

        def caller(group_idx, a, size, fill_value):
            return kernel(group_idx, a, size=size, fill_value=fill_value)

    else:
        value = extra

        def caller(group_idx, a, size, fill_value):
            return kernel(group_idx, a, value, size=size, fill_value=fill_value)

    caller.__qualname__ = caller.__name__ = name
    return nb.njit(caller)


@cache
def _jitted_caller(name, own):
    """The jitted caller of one kernel, with the argument of its own baked in.

    The caller is built once per kernel and shared: a dispatcher compiles the
    first time it is called, so a caller built per test would compile a copy
    of itself for every test that asks for it.
    """
    stub = getattr(aggregate_numba, name)
    return _make_jitted_caller_with_extra(stub, own, name=f"caller_{name}_{own!r}")


def _kernel_result(name, values, **kwargs):
    """Call the kernel of ``name`` from jitted code (fill default via None)."""
    own = _kernel_arguments(name, kwargs)
    caller = _jitted_caller(name, own[0] if own else None)
    return caller(GROUP_IDX, values, -1, None)


def _python_kernel_result(name, values, **kwargs):
    """Call the kernel of ``name`` from python - one argument per position.

    The other way into the very same kernel: the same signature, the same
    compiled driver, only no jitted caller around it.
    """
    return getattr(aggregate_numba, name)(GROUP_IDX, values, *_kernel_arguments(name, kwargs), -1, None)


def _layer2_result(name, values, **kwargs):
    """The python-level aggregate of the same function and the same arguments."""
    return aggregate_numba.aggregate(GROUP_IDX, values, func=name, **kwargs)


@pytest.mark.parametrize("name,kwargs", _FLOAT_OPS)
def test_float_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, FLOAT_A, **kwargs)
    np.testing.assert_allclose(_kernel_result(name, FLOAT_A, **kwargs), expected, equal_nan=True)
    np.testing.assert_allclose(_python_kernel_result(name, FLOAT_A, **kwargs), expected, equal_nan=True)


@pytest.mark.parametrize("name,kwargs", _INT_OPS)
def test_int_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, INT_A, **kwargs)
    np.testing.assert_allclose(_kernel_result(name, INT_A, **kwargs), expected)
    np.testing.assert_allclose(_python_kernel_result(name, INT_A, **kwargs), expected)


@pytest.mark.parametrize("name,kwargs", _BOOL_OPS)
def test_bool_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, BOOL_A, **kwargs)
    np.testing.assert_array_equal(_kernel_result(name, BOOL_A, **kwargs), expected)
    np.testing.assert_array_equal(_python_kernel_result(name, BOOL_A, **kwargs), expected)


def test_collecting_kernel_matches_layer2():
    """The collecting kernel hands out the sequences of the groups.

    Its result is a list of arrays where every other kernel returns one
    number per group - the python-level aggregate() collects them in an
    object array, and the two hold the same sequences.
    """
    expected = aggregate_numba.aggregate(GROUP_IDX, FLOAT_A, func="array")
    # the jitted and the python call of the kernel hand out the same sequences
    for result in (_kernel_result("array", FLOAT_A), _python_kernel_result("array", FLOAT_A)):
        assert len(result) == len(expected)
        for got, other in zip(result, expected):
            np.testing.assert_array_equal(got, other)
            assert got.dtype == other.dtype
        # the group without values holds an empty sequence
        assert len(result[2]) == 0


def test_collecting_kernel_refuses_a_fill_value():
    """The collecting kernel cannot place a fill value.

    An absent group holds an empty sequence, and the driver cannot put
    anything else there - so the kernel says so instead of ignoring the
    argument, as no other kernel does.
    """

    @nb.njit
    def caller(group_idx, a, fill_value):
        return aggregate_numba.array(group_idx, a, -1, fill_value)

    with pytest.raises(nb.TypingError, match="has no fill_value"):
        caller(GROUP_IDX, FLOAT_A, 7.0)


def test_njit_composition():
    """Kernels compose within jitted code - the motivating use case."""

    @nb.njit
    def zscore(group_idx, values):
        m = aggregate_numba.mean(group_idx, values)
        s = aggregate_numba.std(group_idx, values)
        return (values - m[group_idx]) / s[group_idx]

    result = zscore(np.array([0, 0, 1, 1]), np.array([1.0, 3.0, 10.0, 14.0]))
    np.testing.assert_allclose(result, [-1.0, 1.0, -1.0, 1.0])


def test_pure_python_call_refuses_a_fill_value_as_jitted_code_does():
    """The refusal of the collecting kernel is the same sentence in both worlds.

    Jitted code is answered with a typing error, a python call with the plain
    ValueError of its own world - but with the words of the same complaint.
    """
    with pytest.raises(ValueError, match="has no fill_value"):
        aggregate_numba.array(GROUP_IDX, FLOAT_A, -1, 7.0)


# the three ways a caller reaches a kernel: as a python call, from jitted code,
# and through the python-level aggregate() that shares the driver with them.
# The input checks run once per call in each of them - at the entry point of the
# kernel for the first two (the body of the stub and the implementation of its
# overload), in utils_np.input_validation for the third - and no call ever runs
# them twice, which is what the fast path of the wrapper exists to avoid.  Each
# style is a function of the kernel it is asked of: a kernel is an overload, and
# an overload cannot be passed into jitted code as an argument.
@cache
def _styles(kernel_name):
    """The three ways a caller reaches the kernel of ``kernel_name``.

    The three of them are built once per kernel and shared by the tests: the
    jitted one is a dispatcher of its own, and a dispatcher compiles the first
    time it is called - built per test, every test would compile its own copy
    of the very same wrapper.
    """
    kernel = getattr(aggregate_numba, kernel_name)

    def call_python(group_idx, values, size):
        return kernel(group_idx, values, size=size, fill_value=None)

    call_jitted = _make_jitted_caller(kernel, name=f"call_jitted_{kernel_name}")

    def call_aggregate(group_idx, values, size):
        return aggregate(group_idx, values, func=kernel_name, size=size)

    return [call_python, call_jitted, call_aggregate]


_SAME_WORDS = [
    # (group_idx, values, size, the words of the kernel, the words of aggregate())
    #
    # the kernel and the python-level call do not say the very same thing
    # about a length that does not match: the call knows that a scalar is a
    # value of its own, the kernel does not
    (GROUP_IDX[:0], FLOAT_A[:0], -1, EMPTY_GROUP_IDX, EMPTY_GROUP_IDX),
    (np.array([0, 1, 2]), FLOAT_A[:2], -1, LENGTH_MISMATCH, LENGTH_MISMATCH_OR_SCALAR),
    (GROUP_IDX, np.append(FLOAT_A, 1.0), -1, LENGTH_MISMATCH, LENGTH_MISMATCH_OR_SCALAR),
    (np.array([0, -1]), FLOAT_A[:2], -1, NEGATIVE_INDICES, NEGATIVE_INDICES),
    (GROUP_IDX, FLOAT_A, 1, TOO_LARGE_FOR_SIZE.format(size=1), TOO_LARGE_FOR_SIZE.format(size=1)),
]

# what only a kernel is strict about: aggregate() spreads a multidimensional a
# over an axis, coerces a dtype it can not index with and repeats a single value
# over every label, and a kernel has none of those freedoms.  Jitted code is
# answered by the typing function of the kernel, which knows the shape and the
# dtype at compile time and answers with a TypingError.
#
# The complaint about the labels is the one the python-level call makes as
# well - a TypeError there and here, in the same words.
_STRICT_ONLY = [
    (np.tile(GROUP_IDX, (2, 1)), FLOAT_A, "flat one-dimensional", -1, ValueError),
    (GROUP_IDX, np.tile(FLOAT_A, (2, 1)), "flat one-dimensional", -1, ValueError),
    (GROUP_IDX.astype(np.float64), FLOAT_A, "group_idx must be of integer type", -1, TypeError),
    (GROUP_IDX.astype(bool), FLOAT_A, "group_idx must be of integer type", -1, TypeError),
    (GROUP_IDX, FLOAT_A.astype(np.str_), "numeric values", -1, ValueError),
    (GROUP_IDX, np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]"), "numeric values", -1, ValueError),
]


@pytest.mark.parametrize("kernel_name", _ALL_KERNELS)
def test_every_kernel_refuses_the_empty_labels(kernel_name):
    """Every kernel of the backend refuses the empty labels.

    The check is written in the stub of the operation, and every stub has it -
    but a kernel that loses it would go unnoticed if only a few of them were
    asked.  This test asks all of them, once; the jitted way into a kernel is
    asked of the representatives of :data:`_FAULT_KERNELS`, where the whole
    matrix of the faults is checked.
    """
    call = _styles(kernel_name)[0]
    with pytest.raises(ValueError, match=EMPTY_GROUP_IDX):
        call(GROUP_IDX[:0], FLOAT_A[:0], -1)


@pytest.mark.parametrize("style", [0, 1, 2], ids=["python", "jitted", "aggregate()"])
@pytest.mark.parametrize("kernel_name", _FAULT_KERNELS)
@pytest.mark.parametrize("group_idx, values, size, kernel_words, call_words", _SAME_WORDS)
def test_every_call_refuses_the_input_the_driver_cannot_read(
    kernel_name, style, group_idx, values, size, kernel_words, call_words
):
    """The value checks of a kernel call, in both worlds and through aggregate().

    The drivers take one value per label and numba never looks at the bounds of
    what it reads: a short ``a`` is answered with the values behind the end of
    the array, an empty one with nothing at all.  The entry points of the kernel
    therefore ask the two lengths and the bounds of the labels before the driver
    runs - the same questions ``input_validation`` asks the python-level call.

    The words of the complaint are the ones of ``numpy_groupies.validation``,
    and every way of calling a kernel says the very same ones - the python
    call of the kernel, the jitted one and the python-level aggregate() alike.
    """
    call = _styles(kernel_name)[style]
    with pytest.raises(ValueError) as excinfo:
        call(group_idx, values, size)
    assert str(excinfo.value) == (call_words if style == 2 else kernel_words)


@pytest.mark.parametrize("style", [0, 1], ids=["python", "jitted"])
@pytest.mark.parametrize("kernel_name", _FAULT_KERNELS)
@pytest.mark.parametrize("group_idx, values, message, size, python_error", _STRICT_ONLY)
def test_every_kernel_refuses_the_input_it_cannot_take(
    kernel_name, style, group_idx, values, message, size, python_error
):
    """The shape and dtype checks, as a python call and as a jitted one.

    A two-dimensional ``group_idx`` the driver walks by row is answered with a
    quietly wrong result (the labels of one dimension index as well as the flat
    ones), a float or a bool dtype with labels that are not labels, and a dtype
    the drivers were never written for with a numba error that is harder to read
    than this one.
    """
    call = _styles(kernel_name)[style]
    # the jitted style is answered before anything runs, by the typing function
    # of the kernel, which knows the shape and the dtype at compile time
    error = nb.TypingError if style == 1 else python_error
    with pytest.raises(error, match=message):
        call(group_idx, values, size)


# the fill value of a call has to fit the output it fills - the same rule the
# python-level call applies (utils_np.check_fill_value), in the same words:
# a nan does not fit an integer output, a boolean output takes the two
# booleans, and a value that is no number at all fits nothing.
_FILL_CASES = [
    ("sum", INT_A, np.nan, "fill_value must be convertible into int64"),
    ("sum", INT_A, 2.0**63, "fill_value must be convertible into int64"),
    ("all", BOOL_A, 2.0, "Value not boolean"),
    ("any", BOOL_A, 0.5, "Value not boolean"),
]


@pytest.mark.parametrize("style", [0, 1, 2], ids=["python", "jitted", "aggregate()"])
@pytest.mark.parametrize("kernel_name, values, fill_value, message", _FILL_CASES)
def test_every_call_refuses_a_fill_value_the_output_cannot_hold(kernel_name, style, values, fill_value, message):
    """The fill value is checked like the python-level call checks it.

    A nan in an integer output is not a number the output can hold: numba
    writes the integer garbage of a converted nan into the group, and the
    python-level call refuses the value before it reaches the driver.
    """
    kernel = getattr(aggregate_numba, kernel_name)
    if style == 0:

        def call():
            return kernel(GROUP_IDX, values, -1, fill_value)

    elif style == 1:

        @nb.njit
        def call_jitted(group_idx, a, fill):
            return kernel(group_idx, a, -1, fill)

        def call():
            return call_jitted(GROUP_IDX, values, fill_value)

    else:

        def call():
            return aggregate(GROUP_IDX, values, func=kernel_name, fill_value=fill_value)

    with pytest.raises(ValueError, match=message):
        call()


def test_the_jitted_call_refuses_a_fill_value_that_is_no_number():
    """The one check the jitted world can make before anything runs.

    The typing function of the kernel sees the type of the fill value, not its
    value - a fill that is not a number cannot go into any output, and the
    typing error says so instead of leaving it to the driver.
    """
    with pytest.raises(nb.TypingError, match="fill_value must be convertible into int64"):

        @nb.njit
        def caller(group_idx, a, fill):
            return aggregate_numba.sum(group_idx, a, -1, fill)

        caller(GROUP_IDX, INT_A, "x")


def test_the_scalar_input_is_refused_with_its_own_words():
    """A scalar is one value, and a kernel takes one value per label.

    The python-level call repeats a scalar over every label; a kernel has
    none of that freedom and says so - in its own words, not in the words of
    a length mismatch, which is a different fault.
    """
    with pytest.raises(ValueError, match="one value per label"):
        aggregate_numba.sum(GROUP_IDX, 3.0)


def test_the_python_call_answers_in_the_dtype_of_the_jitted_one():
    """The python call resolves the dtype as the jitted one does.

    The values are compared against the python-level aggregate() by the tests
    above; what is left to name is the dtype they arrive in, which the kernel
    takes from its own rule and not from the length-dependent one of
    ``aggregate()``.
    """
    for name in ("sum", "prod", "mean", "var", "std", "min", "first", "len", "all", "cumsum", "nunique", "median"):
        jitted = np.asarray(_kernel_result(name, INT_A))
        python = np.asarray(_python_kernel_result(name, INT_A))
        assert jitted.dtype == python.dtype, name


def test_the_moments_answer_from_njit_like_the_reference():
    # the skewness and the kurtosis are available inside jitted code as well
    @nb.njit
    def wrapped(group_idx, a):
        return aggregate_numba.skew(group_idx, a), aggregate_numba.kurt(group_idx, a)

    group_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([1.0, 2.0, 3.0, 10.0, 1.0, 2.0, 3.0, 4.0])
    skew, kurt = wrapped(group_idx, a)
    np.testing.assert_allclose(skew, aggregate_ref(group_idx, a, func="skew", size=2))
    np.testing.assert_allclose(kurt, aggregate_ref(group_idx, a, func="kurt", size=2))


def test_median_orders_complex_from_njit_like_the_reference():
    # the median driver selects the middle of every group with the ordering
    # of the package, which for complex values is the lexicographic order
    # numpy uses - so the kernel compiles for them and answers like the
    # reference does.  The backends themselves are compared against the
    # reference by the generic tests of the suite.
    @nb.njit
    def wrapped(group_idx, a):
        return aggregate_numba.median(group_idx, a)

    group_idx = np.array([0, 0, 1])
    a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_ref(group_idx, a, func="median", size=2))


def test_quantile_orders_complex_from_njit_like_the_reference():
    # the quantile driver selects the wanted quantile out of the block of every
    # group with the ordering of the package, which for complex values is the
    # lexicographic order numpy uses - so the kernel compiles for them and
    # answers like the reference does.
    @nb.njit
    def wrapped(group_idx, a):
        return aggregate_numba.quantile(group_idx, a, 0.5, 2, None)

    group_idx = np.array([0, 0, 1])
    a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_ref(group_idx, a, func="quantile", q=0.5, size=2))


def test_quantile_of_a_complex_group_with_a_nan_poisons_it_from_njit():
    # a nan in either part of a complex value makes the value incomparable, and
    # the whole group answers with a nan - the plain quantile is contagious
    @nb.njit
    def wrapped(group_idx, a):
        return aggregate_numba.quantile(group_idx, a, 0.5, 2, None)

    group_idx = np.array([0, 0, 1])
    a = np.array([1 + 2j, complex(np.nan, 4.0), 5 + 6j])
    np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_ref(group_idx, a, func="quantile", q=0.5, size=2))


def test_nunique_counts_the_distinct_values_from_njit_like_the_reference():
    # the driver of the distinct values orders the block of every group, so
    # that the values that differ from their predecessor are the distinct
    # ones - for complex values the order is the lexicographic one numpy
    # uses, and the nans are one value among themselves
    @nb.njit
    def wrapped(group_idx, a):
        return aggregate_numba.nunique(group_idx, a)

    group_idx = np.array([0, 0, 0, 1])
    a = np.array([1 + 2j, 1 + 2j, 3 + 4j, complex(np.nan, 1.0)])
    np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_ref(group_idx, a, func="nunique", size=2))


def test_nannunique_leaves_the_nans_out_from_njit_like_the_reference():
    @nb.njit
    def wrapped(group_idx, a):
        return aggregate_numba.nannunique(group_idx, a)

    group_idx = np.array([0, 0, 0, 1])
    a = np.array([1.0, 1.0, 3.0, np.nan])
    np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_ref(group_idx, a, func="nannunique", size=2))


def test_sort_orders_complex_from_njit_like_numpy():
    # the kernel jitted code reaches orders complex like numpy does: numba
    # compares the real part first, the imaginary part on a tie, and leaves a
    # value it cannot compare with - one with a nan in either part, even when
    # the real parts differ - behind everything it is reached by
    @nb.njit
    def wrapped(group_idx, a):
        return aggregate_numba.sort(group_idx, a)

    for a in (
        np.array([1 + 5j, 1 + complex(np.nan, 0), 3 + 0j, 1 + 2j]),
        np.array([3 + 0j, 1 + 2j, 1 + complex(np.nan, 0), 1 + 5j]),
        np.array([1 + complex(0, np.nan), 2 + 1j, 2 + 0j, 1 + 0j]),
    ):
        group_idx = np.zeros(a.size, dtype=int)
        np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_numpy.aggregate(group_idx, a, func="sort"))


def test_sum_bool_widens_int64():
    """Documented njit-kernel deviation: bool input widens to int64 instead of
    the python-level length-dependent overflow guesses."""

    @nb.njit
    def caller(group_idx, a):
        return aggregate_numba.sum(group_idx, a)

    assert caller(GROUP_IDX, BOOL_A).dtype == np.int64


def test_min_default_fill_depends_on_output_dtype():
    """An omitted fill_value resolves like DEFAULT_FILL_VALUE: nan for inexact
    outputs, 0 otherwise - never a garbage nan cast into an integer array."""

    @nb.njit
    def caller(group_idx, a):
        return aggregate_numba.min(group_idx, a)

    group_idx = np.array([0, 0, 2, 2])  # group 1 empty
    result = caller(group_idx, np.array([5.0, 2.0, 7.0, 3.0]))
    assert np.isnan(result[1])
    result = caller(group_idx, np.array([5, 2, 7, 3], dtype=np.int64))
    assert result[1] == 0


def test_fill_value_passthrough():
    """An explicit fill_value reaches the untouched groups unchanged."""

    @nb.njit
    def caller(group_idx, a, fill_value):
        return aggregate_numba.sum(group_idx, a, -1, fill_value)

    result = caller(GROUP_IDX, INT_A, 42)
    assert result[2] == 42


def test_size_too_small_raises():

    @nb.njit
    def caller(group_idx, a, size):
        return aggregate_numba.sum(group_idx, a, size, None)

    with pytest.raises(ValueError, match="too large"):
        caller(GROUP_IDX, FLOAT_A, 1)


def test_integer_mean_overflow_from_njit():
    """The exactness gates run inside the kernels as well."""

    @nb.njit
    def caller(group_idx, a):
        return aggregate_numba.mean(group_idx, a)

    with pytest.raises(OverflowError, match="exact mean"):
        caller(np.array([0, 0]), np.array([2**60, 2**60 + 1], dtype=np.int64))


@pytest.mark.parametrize(
    "kernel_name, values, expected_dtype",
    [
        ("mean", FLOAT_A.astype(np.float32), np.float32),
        ("mean", INT_A, np.float64),
        ("var", (FLOAT_A + 1j * FLOAT_A).astype(np.complex64), np.float32),
        ("var", INT_A, np.float64),
        ("cumsum", INT_A.astype(np.int8), np.int64),
        ("min", INT_A, np.int64),
    ],
)
def test_dtype_rules(kernel_name, values, expected_dtype):
    stub = getattr(aggregate_numba, kernel_name)

    @nb.njit
    def caller(group_idx, a):
        return stub(group_idx, a)

    assert caller(GROUP_IDX, values).dtype == expected_dtype


def test_a_float16_input_is_the_one_refusal_the_typing_function_cannot_name():
    """The dtype no world of a kernel can be asked about.

    ``aggregate()`` widens float16 values to float32 on the way to the driver,
    a kernel has no such step in front of it: the numba type of the values is
    what the entry point of both of its worlds starts from, and numba answers
    float16 - the one numeric dtype it has no type for - with a
    NotImplementedError of its own.
    """
    values = FLOAT_A.astype(np.float16)
    with pytest.raises(NotImplementedError, match="float16"):
        aggregate_numba.mean(GROUP_IDX, values)

    @nb.njit
    def caller(group_idx, a):
        return aggregate_numba.mean(group_idx, a)

    with pytest.raises(NotImplementedError, match="float16"):
        caller(GROUP_IDX, values)


def test_layer1_layer2_share_the_driver():
    """The python-level aggregate() and the kernels use one compiled artifact
    - spot-check that both paths return bit-identical values."""
    name, values = "var", FLOAT_A
    expected = _layer2_result(name, values, ddof=1)
    result = _kernel_result(name, values, ddof=1)
    assert result.dtype == expected.dtype
    assert result.tobytes() == expected.tobytes()


_OP_MODULES = ("reductions", "averaging", "cumulative", "argreductions", "specials", "collect")


def _all_drivers():
    """``(operation name, driver)`` of every operation of the numba backend."""
    for module_name in _OP_MODULES:
        module = import_module(f"numpy_groupies.aggregate_numba.{module_name}")
        for pair in vars(module).values():
            if type(pair).__name__ != "KernelPair":
                continue
            for handle in (pair.plain, pair.nan_variant):
                if handle is not None:
                    yield handle.name, handle.driver


def _identity(function):
    return function.__module__, function.__qualname__, function.__code__.co_firstlineno


def test_no_two_operations_share_a_compiled_identity():
    """Numba mangles the symbol of a compiled function from module, qualname and
    signature - two operations that share those share one symbol, and the driver
    loaded from disk then binds its building blocks to the other operation's
    environment (observed for ``last`` against ``nanlast``, both taking an int64
    and a float64 array).  Distinct qualnames are what keeps them apart.
    """
    identities = {}
    for name, driver in _all_drivers():
        identity = _identity(driver.py_func)
        assert identity not in identities, f"{name} shares its compiled identity with {identities[identity]}"
        identities[identity] = name
    # plain and nan variant of every operation, so more names than operations
    assert len(identities) > 24


def test_the_plain_and_nan_variant_of_one_operation_are_distinct():
    """The nan variant is a second compiled artifact of the same source block -
    the collision of ``last`` and ``nanlast`` came out of exactly that.
    """
    drivers = dict(_all_drivers())
    for name in ("last", "first", "sum", "mean"):
        assert _identity(drivers[name].py_func) != _identity(drivers[f"nan{name}"].py_func)
        assert drivers[f"nan{name}"].py_func.__qualname__ == f"nan{name}_driver"
