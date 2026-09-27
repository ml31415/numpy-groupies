"""Tests for the numba fast path (jitted entry)."""

import numpy as np
import pytest

from ..aggregate_numba import aggregate
from ..utils import resolve_output_dtype


@pytest.fixture(params=[np.float64, np.float32, np.int64, np.int32, np.int8, np.bool_], ids=str)
def adtype(request):
    return request.param


@pytest.fixture(
    params=[
        "sum",
        "prod",
        "len",
        "min",
        "max",
        "first",
        "last",
        "mean",
        "median",
        "std",
        "var",
        "sumofsquares",
        "any",
        "anynan",
        "allnan",
        "trapezoid",
    ]
)
def func(request):
    return request.param


def _make_inputs(func, adtype, rng):
    group_idx = rng.integers(0, 6, 300)
    if adtype is np.bool_:
        if func in ("mean", "median", "std", "var", "trapezoid"):
            pytest.skip("float functions do not accept bool input")
        a = rng.integers(0, 2, 300).astype(adtype)
    elif np.issubdtype(adtype, np.integer):
        if func in ("mean", "median", "std", "var", "trapezoid"):
            a = rng.random(300).astype(np.float64)
            group_idx = group_idx.astype(np.int32)
            return group_idx, a
        a = rng.integers(0, 4, 300).astype(adtype)
    else:
        a = (rng.random(300) * 2 - 1).astype(adtype)
    return group_idx, a


def test_matches_slow_path(func, adtype):
    rng = np.random.default_rng(42)
    group_idx, a = _make_inputs(func, adtype, rng)
    # a numpy-integer size bypasses the fast path and runs the full machinery
    ref = aggregate(group_idx, a, func=func, size=np.int64(6))
    for size in (6, None):
        fast = aggregate(group_idx, a, func=func, size=size)
        assert fast.dtype == ref.dtype
        np.testing.assert_array_equal(fast, ref)


def test_nan_variants(func, adtype):
    if func in ("len", "prod", "sumofsquares", "trapezoid", "anynan", "allnan"):
        pytest.skip("no separate nan variant or covered elsewhere")
    rng = np.random.default_rng(43)
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    a[group_idx == 2] = np.nan
    nfunc = "nan" + func
    ref = aggregate(group_idx, a, func=nfunc, size=np.int64(6))
    for size in (6, None):
        fast = aggregate(group_idx, a, func=nfunc, size=size)
        assert fast.dtype == ref.dtype
        np.testing.assert_array_equal(fast, ref)


def test_argreductions():
    rng = np.random.default_rng(44)
    group_idx = rng.integers(0, 8, 400)
    a = rng.random(400)
    for func in ("argmax", "argmin", "nanargmax", "nanargmin"):
        ref = aggregate(group_idx, a, func=func, size=np.int64(8))
        np.testing.assert_array_equal(aggregate(group_idx, a, func=func), ref)
        np.testing.assert_array_equal(
            aggregate(group_idx, a, func=func, size=10), aggregate(group_idx, a, func=func, size=np.int64(10))
        )


def test_size_detection_matches_explicit():
    rng = np.random.default_rng(45)
    group_idx = rng.integers(0, 9, 500)
    a = rng.random(500)
    ref = aggregate(group_idx, a, func="sum", size=int(group_idx.max()) + 1)
    np.testing.assert_array_equal(aggregate(group_idx, a, func="sum"), ref)


def test_errors_match_slow_path():
    rng = np.random.default_rng(46)
    group_idx = rng.integers(0, 5, 100)
    a = rng.random(100)
    # index too large for a given size
    with pytest.raises(ValueError, match="too large"):
        aggregate(group_idx, a, func="sum", size=3)
    with pytest.raises(ValueError, match="too large"):
        aggregate(group_idx, a, func="sum", size=np.int64(3))
    # negative indices
    with pytest.raises(ValueError, match="negative"):
        aggregate(group_idx - 1, a, func="sum")
    with pytest.raises(ValueError, match="negative"):
        aggregate(group_idx - 1, a, func="sum", size=np.int64(5))
    # empty group_idx
    with pytest.raises(ValueError, match="empty"):
        aggregate(group_idx[:0], a[:0], func="sum")
    # non-integer group_idx
    with pytest.raises(TypeError, match="integer"):
        aggregate(group_idx.astype(float), a, func="sum")
    # length mismatch
    with pytest.raises(ValueError, match="same length"):
        aggregate(group_idx, a[:-1], func="sum")


def test_integer_sum_dtype_guess():
    # the 'sum' overflow guess is length dependent and must agree between
    # paths and with the length-dependent dtype resolution itself, for
    # several input lengths (the plan cache is keyed on the length)
    rng = np.random.default_rng(47)
    for adtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32, np.bool_):
        for n in (10, 1000):
            group_idx = rng.integers(0, 4, n)
            a = (rng.random(n) > 0.5).astype(adtype)
            expected_dtype = resolve_output_dtype(None, "sum", a.dtype, n)
            ref = aggregate(group_idx, a, func="sum", size=np.int64(4))
            fast = aggregate(group_idx, a, func="sum", size=4)
            assert fast.dtype == ref.dtype == expected_dtype
            np.testing.assert_array_equal(fast, ref)


def test_slow_path_still_reachable():
    """Inputs the fast path does not take must keep their behaviour."""
    rng = np.random.default_rng(50)
    group_idx = rng.integers(0, 4, 100)
    a = rng.random(100)
    # scalar input
    np.testing.assert_array_equal(
        aggregate(group_idx, 2.0, func="sum"), [2.0 * np.sum(group_idx == i) for i in range(4)]
    )
    # custom fill value forces the finalize machinery
    ref = aggregate(group_idx, a, func="sum", fill_value=-1.0, size=np.int64(5))
    np.testing.assert_array_equal(aggregate(group_idx, a, func="sum", fill_value=-1.0, size=5), ref)
    # 2d group_idx / axis handling
    group_idx_2d = rng.integers(0, 3, (2, 40))
    a2 = rng.random(40)
    ref2 = aggregate(group_idx_2d, a2, func="sum", size=[3, 3])
    np.testing.assert_array_equal(aggregate(group_idx_2d, a2, func="sum", size=None), ref2)
    # axis aggregation
    a3 = rng.random((100, 3))
    ref3 = aggregate(group_idx, a3, func="sum", axis=0, size=np.int64(4))
    np.testing.assert_array_equal(aggregate(group_idx, a3, func="sum", axis=0, size=4), ref3)
