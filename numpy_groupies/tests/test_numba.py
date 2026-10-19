"""Tests for the numba backend: the jitted fast path, the slow-path
fallback, the collecting function, the callable cache, the ``out=`` parameter
and the lock around the shared on-disk compilation cache.
"""

import itertools
import os
from threading import Thread

import numpy as np
import pytest

from numpy_groupies import uaggregate
from numpy_groupies.aggregate_numba import aggregate
from numpy_groupies.aggregate_numba.jit import _ProcessSafeCacheFile
from numpy_groupies.aggregate_numba.ops import _dispatch
from numpy_groupies.aggregate_numpy import aggregate as aggregate_np
from numpy_groupies.utils import resolve_output_dtype


@pytest.fixture(
    params=[np.float64, np.float32, np.float16, np.int64, np.int32, np.int8, np.bool_],
    ids=lambda t: np.dtype(t).name,
)
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


def test_matches_slow_path(rng, func, adtype):
    group_idx, a = _make_inputs(func, adtype, rng)
    # a numpy-integer size bypasses the fast path and runs the full machinery
    ref = aggregate(group_idx, a, func=func, size=np.int64(6))
    for size in (6, None):
        fast = aggregate(group_idx, a, func=func, size=size)
        assert fast.dtype == ref.dtype
        np.testing.assert_array_equal(fast, ref)


def test_nan_variants(rng, func, adtype):
    if func in ("len", "prod", "sumofsquares", "trapezoid", "anynan", "allnan"):
        pytest.skip("no separate nan variant or covered elsewhere")
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    a[group_idx == 2] = np.nan
    nfunc = "nan" + func
    ref = aggregate(group_idx, a, func=nfunc, size=np.int64(6))
    for size in (6, None):
        fast = aggregate(group_idx, a, func=nfunc, size=size)
        assert fast.dtype == ref.dtype
        np.testing.assert_array_equal(fast, ref)


def test_argreductions(rng):
    group_idx = rng.integers(0, 8, 400)
    a = rng.random(400)
    for func in ("argmax", "argmin", "nanargmax", "nanargmin"):
        ref = aggregate(group_idx, a, func=func, size=np.int64(8))
        np.testing.assert_array_equal(aggregate(group_idx, a, func=func), ref)
        np.testing.assert_array_equal(
            aggregate(group_idx, a, func=func, size=10), aggregate(group_idx, a, func=func, size=np.int64(10))
        )


def test_size_detection_matches_explicit(rng):
    group_idx = rng.integers(0, 9, 500)
    a = rng.random(500)
    ref = aggregate(group_idx, a, func="sum", size=int(group_idx.max()) + 1)
    np.testing.assert_array_equal(aggregate(group_idx, a, func="sum"), ref)


@pytest.mark.parametrize("func", ["argmin", "argmax", "nanargmin", "nanargmax"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("size", [3, np.int64(3)])
def test_integer_argreductions_preserve_comparisons(func, dtype, size):
    base = int(np.iinfo(dtype).max) - 8
    a = np.array([base + 1, base + 2, base + 2, base + 4, base + 3], dtype=dtype)
    group_idx = np.array([0, 0, 0, 1, 1])
    expected = [0, 4, -1] if "min" in func else [1, 3, -1]
    np.testing.assert_array_equal(aggregate(group_idx, a, func=func, size=size), expected)


@pytest.mark.parametrize("func", ["argmin", "argmax", "nanargmin", "nanargmax"])
def test_integer_argreductions_along_axis(rng, func):
    a = np.array([[2**53 + 1, 2**53, 2**53 + 2], [-(2**53), -(2**53) - 1, -(2**53) - 2]])
    expected = [[1, 2, -1], [1, 2, -1]] if "min" in func else [[0, 2, -1], [0, 2, -1]]
    result = aggregate(np.array([0, 0, 1]), a, func=func, axis=1, size=3)
    np.testing.assert_array_equal(result, expected)


def test_errors_match_slow_path(rng):
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


def test_integer_sum_dtype_guess(rng):
    # the 'sum' overflow guess is length dependent, so check several lengths
    for adtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32, np.bool_):
        for n in (10, 1000):
            group_idx = rng.integers(0, 4, n)
            a = (rng.random(n) > 0.5).astype(adtype)
            expected_dtype = resolve_output_dtype(None, "sum", a.dtype, n)
            ref = aggregate(group_idx, a, func="sum", size=np.int64(4))
            fast = aggregate(group_idx, a, func="sum", size=4)
            assert fast.dtype == ref.dtype == expected_dtype
            np.testing.assert_array_equal(fast, ref)


def test_slow_path_still_reachable(rng):
    """Inputs the fast path does not take must keep their behaviour."""
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


def test_numba_callable_cache_construction(monkeypatch):
    pytest.importorskip("numba")
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 1, 1, 2])
    a = np.arange(4.0)
    func = lambda x: np.sum(x)

    constructed = []
    orig_build = aggregate_numba.build_generic_aggregate

    def counting(func):
        constructed.append(func)
        return orig_build(func)

    monkeypatch.setattr(aggregate_numba, "build_generic_aggregate", counting)

    cache = {}
    aggregate_numba.aggregate(group_idx, a, func=func, cache=cache)
    for _ in range(3):
        aggregate_numba.aggregate(group_idx, a, func=func, cache=cache)

    # one cache entry, and the cached op is reused instead of being
    # re-constructed on every call (dict.setdefault used to do exactly that)
    assert len(cache) == 1
    assert len(constructed) == 1


def test_aggregate_out(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    ref = aggregate(group_idx, a, func="sum")
    out = np.empty(6)
    ret = aggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # wrong shape or dtype is rejected
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="sum", out=np.empty(5))
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="sum", out=np.empty(6, dtype=np.float32))
    # custom callables do not support out
    with pytest.raises(NotImplementedError, match="out="):
        aggregate(group_idx, a, func=lambda x: np.sum(x), out=out)


def test_every_known_function_is_implemented():
    """A function the aliasing knows and this backend does not have.

    The dispatch table holds such a name too, with no implementation behind it,
    so the answer is about a function that is not available here rather than
    about a func that named nothing at all.  This backend knows them all - the
    collecting function included, which has no compiled kernel but a python
    implementation of its own.
    """
    missing = sorted({name for name, impl in _dispatch.values() if impl is None})
    assert missing == []


def test_a_func_that_named_nothing():
    """A func that is no key of the dispatch table at all.

    It is neither a function anybody knows nor a callable of the caller's own,
    so it says so - the error of a callable is a different one.
    """
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([1.0, 3.0, 7.0, 2.0])
    with pytest.raises(ValueError, match="neither a valid function string"):
        aggregate(group_idx, a, func="no_such_function", size=2)


def test_uaggregate_out(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    ref = uaggregate(group_idx, a, func="sum")
    out = np.empty(300)
    ret = uaggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # shape of ret[group_idx] is enforced
    with pytest.raises(ValueError, match="expected shape"):
        uaggregate(group_idx, a, func="sum", out=np.empty(299))
    with pytest.raises(TypeError, match="result dtype"):
        uaggregate(group_idx, a, func="sum", out=np.empty(300, dtype=np.int64))
    # multidimensional group_idx falls back to fancy indexing assignment
    group_idx_2d = rng.integers(0, 3, (2, 40))
    a2 = rng.random(40)
    ref2 = uaggregate(group_idx_2d, a2, func="sum")
    out2 = np.empty(ref2.shape)
    ret2 = uaggregate(group_idx_2d, a2, func="sum", out=out2)
    assert ret2 is out2
    np.testing.assert_array_equal(out2, ref2)


def test_aggregate_out_complex(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300) + 1j * rng.random(300)
    ref = aggregate(group_idx, a, func="sum")
    out = np.empty(6, dtype=np.complex128)
    ret = aggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # a real out= would throw the imaginary part away, so it is refused
    with pytest.raises(TypeError, match="dtype complex128"):
        aggregate(group_idx, a, func="sum", out=np.empty(6))
    # var measures a real squared magnitude, so a real out= is the right shape
    ref_var = aggregate(group_idx, a, func="var")
    out_var = np.empty(6)
    aggregate(group_idx, a, func="var", out=out_var)
    np.testing.assert_allclose(out_var, ref_var)


def test_uaggregate_out_complex(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300) + 1j * rng.random(300)
    ref = uaggregate(group_idx, a, func="sum")
    out = np.empty(300, dtype=np.complex128)
    ret = uaggregate(group_idx, a, out=out, func="sum")
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    with pytest.raises(TypeError, match="does not match the result dtype"):
        uaggregate(group_idx, a, out=np.empty(300), func="sum")


def test_float16_follows_the_float32_plan():
    # numba has no float16: the input is widened to float32, the kernel runs
    # in the float32 plan and an inexact result is rounded back once.  The
    # numpy twin instead rounds after every step in float16, so on values
    # whose partial sums a float16 cannot represent the two legitimately
    # differ - this pins the float32 plan as the numba behaviour.
    group_idx = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float16)

    ref = aggregate(group_idx, a.astype(np.float32), func="cumsum").astype(np.float16)
    np.testing.assert_array_equal(aggregate(group_idx, a, func="cumsum"), ref)


# the collecting function - the one function of this backend without a
# compiled kernel, its result being a sequence per group


def _same_groups(got, expected):
    """Compare two collections of per-group sequences."""
    assert len(got) == len(expected)
    for g, (one, other) in enumerate(zip(got, expected)):
        if one is None or other is None:
            assert one is other, f"group {g}: {one!r} != {other!r}"
        else:
            np.testing.assert_array_equal(one, other, err_msg=f"group {g}")


def test_collecting_matches_numpy():
    group_idx = np.array([3, 1, 3, 0, 1, 3])
    a = np.arange(6.0)
    res = aggregate(group_idx, a, func="array")
    ref = aggregate_np(group_idx, a, func="array")
    _same_groups(res, ref)
    # the values of a group keep the order they appear in
    np.testing.assert_array_equal(res[3], [0.0, 2.0, 5.0])
    # a group without values holds an empty sequence
    assert len(res[2]) == 0


@pytest.mark.parametrize("func", ["array", list, "split", "splice", np.array, np.asarray])
def test_collecting_aliases(func):
    group_idx = np.array([0, 1, 1])
    a = np.arange(3.0)
    res = aggregate(group_idx, a, func=func)
    np.testing.assert_array_equal(res[1], [1.0, 2.0])


@pytest.mark.parametrize(
    ["fill_value", "expected"],
    [(None, None), (-1, -1), (np.nan, np.nan)],
    ids=["none", "int", "nan"],
)
def test_collecting_fill_value(fill_value, expected):
    # a group without values takes the fill value
    group_idx = np.array([0, 2, 2])
    a = np.arange(3.0)
    res = aggregate(group_idx, a, func="array", fill_value=fill_value)
    assert res[1] is expected or res[1] == expected


def test_collecting_empty_fill_value():
    # an empty sequence leaves the empty block of the group alone
    group_idx = np.array([0, 2, 2])
    a = np.arange(3.0)
    res = aggregate(group_idx, a, func="array", fill_value=[])
    assert len(res[1]) == 0
    ref = aggregate_np(group_idx, a, func="array", fill_value=[])
    _same_groups(res, ref)


def test_collecting_multidimensional_labels():
    group_idx = np.array(list(itertools.product(range(3), range(4)))).T[:, :-1]
    a = np.arange(11, dtype=float)
    res = aggregate(group_idx, a, func="array", size=(3, 4))
    assert res.shape == (3, 4)
    assert len(res[2, 3]) == 0
    ref = aggregate_np(group_idx, a, func="array", size=(3, 4))
    _same_groups(res.ravel(), ref.ravel())


def test_collecting_keeps_the_dtype():
    # the values are handed out as they are - the float16 of the input is
    # not widened to float32 the way the kernels widen it
    group_idx = np.array([0, 0, 1])
    a = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    res = aggregate(group_idx, a, func="array")
    assert res[0].dtype == np.float16
    ref = aggregate_np(group_idx, a, func="array")
    _same_groups(res, ref)


def test_collecting_out():
    group_idx = np.array([0, 1, 1])
    a = np.arange(3.0)
    out = np.empty(2, dtype=object)
    ret = aggregate(group_idx, a, func="array", out=out)
    assert ret is out
    np.testing.assert_array_equal(out[1], [1.0, 2.0])
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="array", out=np.empty(3, dtype=object))


def test_collecting_errors():
    group_idx = np.array([0, 1, 1])
    a = np.arange(3.0)
    with pytest.raises(ValueError, match="too large"):
        aggregate(group_idx, a, func="array", size=1)
    with pytest.raises(ValueError, match="negative"):
        aggregate(group_idx - 1, a, func="array")
    # a scalar stands for one value, which the collecting function cannot take
    with pytest.raises(ValueError, match="scalar inputs"):
        aggregate(group_idx, 1.0, func="array")


# The on-disk cache is shared by the workers of a parallel test run: numba
# writes the index of a cache file before the data file it announces, so
# concurrent writers of one cache can leave an index naming the machine code of
# another signature.  ``_ProcessSafeCacheFile`` excludes those writers from each
# other, which is only observable under contention - so this is the one place
# the suite writes one cache from several flows of control at once.  Threads
# rather than processes: ``flock`` excludes separate open file descriptions,
# which is what a writer holds, and the interpreter of a pytest worker is not a
# safe thing to fork.

_workers = min(4, os.cpu_count())
_per_worker = 25


def _write(worker, count, cache_path):
    """Save ``count`` distinct signatures of one cache file, from one flow."""
    cache_file = _ProcessSafeCacheFile(
        cache_path=cache_path,
        filename_base="probe",
        source_stamp="stamp",
    )
    for index in range(count):
        cache_file.save(f"key-{worker}-{index}", (worker, index))


def test_every_key_of_a_contended_cache_names_its_own_machine(tmp_path):
    """No key of a raced cache may name another writer's machine code."""
    cache_path = str(tmp_path)
    threads = [Thread(target=_write, args=(worker, _per_worker, cache_path)) for worker in range(_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    cache_file = _ProcessSafeCacheFile(cache_path=cache_path, filename_base="probe", source_stamp="stamp")
    assert len(cache_file._load_index()) == _workers * _per_worker
    for worker in range(_workers):
        for index in range(_per_worker):
            assert cache_file.load(f"key-{worker}-{index}") == (worker, index)
