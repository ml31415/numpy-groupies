"""Tests for the njit-callable kernels of ``aggregate_numba.kernels``.

The kernels share their compiled drivers with the python-level
``aggregate()``, so value-parity with the (elsewhere exhaustively tested)
layer-2 results is the contract here; the explicitly documented deviations
(njit-only dtype approximations, fill defaults, pure-python refusal) get
their own tests.
"""

from importlib import import_module

import numba as nb
import numpy as np
import pytest

from numpy_groupies import aggregate_numba, aggregate_numpy
from numpy_groupies.aggregate_numba import kernels
from numpy_groupies.common import func_extras
from numpy_groupies.tests.reference import aggregate_ref

GROUP_IDX = np.array([0, 0, 1, 1, 1, 3])  # group 2 stays empty
FLOAT_A = np.array([3.0, 1.0, np.nan, 2.0, 5.0, 4.0])
INT_A = np.array([3, 1, 7, 2, 5, 4], dtype=np.int64)
BOOL_A = np.array([True, False, True, True, False, True])

# (kernel name, kwargs, values) - the kernels run inside a jitted caller and
# are compared against the python-level aggregate() of the same name
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


def _kernel_result(name, values, **kwargs):
    """Call the kernel of ``name`` from jitted code (fill default via None).

    Every kernel carries the signature of its function, so the arguments the
    function has of its own are passed by their name.
    """
    stub = getattr(kernels, name)
    # the argument the function has of its own comes first, the size and the
    # fill value after it - the signature of the kernel.  It is always
    # passed, with its default if the test does not ask for another one
    extras = func_extras(name)
    own = next(iter(kwargs.values()), None) if extras else None
    if own is None and extras:
        own = extras[0][2]
    arguments = "" if own is None else f", {own!r}"
    source = f"""
def _caller(group_idx, a, size, fill_value):
    return _stub(group_idx, a{arguments}, size, fill_value)
"""
    namespace = {"_stub": stub}
    exec(source, namespace)  # noqa: S102 - the signature of the kernel is spelled out in one place
    caller = nb.njit(namespace["_caller"])
    return caller(GROUP_IDX, values, -1, None)


def _layer2_result(name, values, **kwargs):
    """The python-level aggregate of the same function and the same arguments."""
    return aggregate_numba.aggregate(GROUP_IDX, values, func=name, **kwargs)


@pytest.mark.parametrize("name,kwargs", _FLOAT_OPS)
def test_float_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, FLOAT_A, **kwargs)
    result = _kernel_result(name, FLOAT_A, **kwargs)
    np.testing.assert_allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize("name,kwargs", _INT_OPS)
def test_int_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, INT_A, **kwargs)
    result = _kernel_result(name, INT_A, **kwargs)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("name,kwargs", _BOOL_OPS)
def test_bool_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, BOOL_A, **kwargs)
    result = _kernel_result(name, BOOL_A, **kwargs)
    np.testing.assert_array_equal(result, expected)


def test_collecting_kernel_matches_layer2():
    """The collecting kernel hands out the sequences of the groups.

    Its result is a list of arrays where every other kernel returns one
    number per group - the python-level aggregate() collects them in an
    object array, and the two hold the same sequences.
    """
    expected = aggregate_numba.aggregate(GROUP_IDX, FLOAT_A, func="array")
    result = _kernel_result("array", FLOAT_A)
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
        return kernels.array(group_idx, a, -1, fill_value)

    with pytest.raises(nb.TypingError, match="has no fill_value"):
        caller(GROUP_IDX, FLOAT_A, 7.0)


def test_njit_composition():
    """Kernels compose within jitted code - the motivating use case."""

    @nb.njit
    def zscore(group_idx, values):
        m = kernels.mean(group_idx, values)
        s = kernels.std(group_idx, values)
        return (values - m[group_idx]) / s[group_idx]

    result = zscore(np.array([0, 0, 1, 1]), np.array([1.0, 3.0, 10.0, 14.0]))
    np.testing.assert_allclose(result, [-1.0, 1.0, -1.0, 1.0])


def test_pure_python_call_raises():
    with pytest.raises(NotImplementedError, match="jitted"):
        kernels.mean(GROUP_IDX, FLOAT_A)


def test_the_moments_answer_from_njit_like_the_reference():
    # the skewness and the kurtosis are available inside jitted code as well
    @nb.njit
    def wrapped(group_idx, a):
        return kernels.skew(group_idx, a), kernels.kurt(group_idx, a)

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
        return kernels.median(group_idx, a)

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
        return kernels.quantile(group_idx, a, 0.5, 2, None)

    group_idx = np.array([0, 0, 1])
    a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_ref(group_idx, a, func="quantile", q=0.5, size=2))


def test_quantile_of_a_complex_group_with_a_nan_poisons_it_from_njit():
    # a nan in either part of a complex value makes the value incomparable, and
    # the whole group answers with a nan - the plain quantile is contagious
    @nb.njit
    def wrapped(group_idx, a):
        return kernels.quantile(group_idx, a, 0.5, 2, None)

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
        return kernels.nunique(group_idx, a)

    group_idx = np.array([0, 0, 0, 1])
    a = np.array([1 + 2j, 1 + 2j, 3 + 4j, complex(np.nan, 1.0)])
    np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_ref(group_idx, a, func="nunique", size=2))


def test_nannunique_leaves_the_nans_out_from_njit_like_the_reference():
    @nb.njit
    def wrapped(group_idx, a):
        return kernels.nannunique(group_idx, a)

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
        return kernels.sort(group_idx, a)

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
        return kernels.sum(group_idx, a)

    assert caller(GROUP_IDX, BOOL_A).dtype == np.int64


def test_min_default_fill_depends_on_output_dtype():
    """An omitted fill_value resolves like DEFAULT_FILL_VALUE: nan for inexact
    outputs, 0 otherwise - never a garbage nan cast into an integer array."""

    @nb.njit
    def caller(group_idx, a):
        return kernels.min(group_idx, a)

    group_idx = np.array([0, 0, 2, 2])  # group 1 empty
    result = caller(group_idx, np.array([5.0, 2.0, 7.0, 3.0]))
    assert np.isnan(result[1])
    result = caller(group_idx, np.array([5, 2, 7, 3], dtype=np.int64))
    assert result[1] == 0


def test_fill_value_passthrough():
    """An explicit fill_value reaches the untouched groups unchanged."""

    @nb.njit
    def caller(group_idx, a, fill_value):
        return kernels.sum(group_idx, a, -1, fill_value)

    result = caller(GROUP_IDX, INT_A, 42)
    assert result[2] == 42


def test_size_too_small_raises():

    @nb.njit
    def caller(group_idx, a, size):
        return kernels.sum(group_idx, a, size, None)

    with pytest.raises(ValueError, match="too large"):
        caller(GROUP_IDX, FLOAT_A, 1)


def test_integer_mean_overflow_from_njit():
    """The exactness gates run inside the kernels as well."""

    @nb.njit
    def caller(group_idx, a):
        return kernels.mean(group_idx, a)

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
    stub = getattr(kernels, kernel_name)

    @nb.njit
    def caller(group_idx, a):
        return stub(group_idx, a)

    assert caller(GROUP_IDX, values).dtype == expected_dtype


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
