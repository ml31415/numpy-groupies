import timeit
import pytest
import numpy as np


from ..aggregate_weave import aggregate as aggregate_weave, unpack, step_indices, step_count


def test_step_indices_length():
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
    for _ in xrange(20):
        np.random.shuffle(group_idx)
        step_cnt_ref = np.count_nonzero(np.diff(group_idx))
        assert step_count(group_idx) == step_cnt_ref + 1
        assert len(step_indices(group_idx)) == step_cnt_ref + 2


def test_step_indices_fields():
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
    steps = step_indices(group_idx)
    np.testing.assert_array_equal(steps, np.array([ 0, 3, 5, 7, 8, 9, 11]))


def test_unpack_compare():
    group_idx = np.arange(10)
    np.random.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 3)
    a = np.random.randn(group_idx.size)
    np.testing.assert_array_equal(unpack(group_idx, a), a[group_idx])


def test_unpack_simple():
    group_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    vals = aggregate_weave(group_idx, np.arange(group_idx.size))
    unpacked = unpack(group_idx, vals)
    np.testing.assert_array_equal(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21, 30, 30, 30]))


def test_unpack_incontiguous_a():
    group_idx = np.array([5, 5, 3, 3, 1, 1, 4, 4])
    vals = aggregate_weave(group_idx, np.arange(group_idx.size))
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_unpack_incontiguous_b():
    group_idx = np.array([5, 5, 12, 5, 9, 12, 9])
    x = np.array([1, 2, 3, 24, 15, 6, 17])
    vals = aggregate_weave(group_idx, x)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_unpack_long():
    group_idx = np.repeat(np.arange(10000), 20)
    a = np.arange(group_idx.size, dtype=int)
    vals = aggregate_weave(group_idx, a)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_unpack_timing():
    # Unpacking should not be considerably slower than usual indexing
    group_idx = np.repeat(np.arange(10000), 20)
    a = np.arange(group_idx.size, dtype=int)
    vals = aggregate_weave(group_idx, a)

    t0 = timeit.Timer(lambda: vals[group_idx]).timeit(number=100)
    t1 = timeit.Timer(lambda: unpack(group_idx, vals)).timeit(number=100)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])
    # This was a speedup once, but using openblas speeds up numpy greatly
    # So let's just make sure it's not a too big drawback
    assert t0 / t1 > 0.5


@pytest.mark.skipif(True, reason="downscaled needs reimplementation")
def test_unpack_downscaled():
    group_idx = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
    vals = aggregate_weave(group_idx, np.arange(group_idx.size), mode='downscaled')
    unpacked = unpack(group_idx, vals, mode='downscaled')
    np.testing.assert_array_equal(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21]))
