import timeit
import pytest
import numpy as np


from ..aggregate_weave import aggregate as aggregate_weave, unpack, step_indices, step_count


def test_step_indices_length():
    accmap = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
    for _ in xrange(20):
        np.random.shuffle(accmap)
        step_cnt_ref = np.count_nonzero(np.diff(accmap))
        assert step_count(accmap) == step_cnt_ref + 1
        assert len(step_indices(accmap)) == step_cnt_ref + 2


def test_step_indices_fields():
    accmap = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
    steps = step_indices(accmap)
    np.testing.assert_array_equal(steps, np.array([ 0, 3, 5, 7, 8, 9, 11]))


def test_unpack_compare():
    accmap = np.arange(10)
    np.random.shuffle(accmap)
    accmap = np.repeat(accmap, 3)
    a = np.random.randn(accmap.size)
    np.testing.assert_array_equal(unpack(accmap, a), a[accmap])


def test_unpack_simple():
    accmap = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    vals = aggregate_weave(accmap, np.arange(accmap.size))
    unpacked = unpack(accmap, vals)
    np.testing.assert_array_equal(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21, 30, 30, 30]))


def test_unpack_incontiguous_a():
    accmap = np.array([5, 5, 3, 3, 1, 1, 4, 4])
    vals = aggregate_weave(accmap, np.arange(accmap.size))
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])


def test_unpack_incontiguous_b():
    accmap = np.array([5, 5, 12, 5, 9, 12, 9])
    x = np.array([1, 2, 3, 24, 15, 6, 17])
    vals = aggregate_weave(accmap, x)
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])


def test_unpack_long():
    accmap = np.repeat(np.arange(10000), 20)
    a = np.arange(accmap.size, dtype=int)
    vals = aggregate_weave(accmap, a)
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])


def test_unpack_timing():
    # Unpacking should not be considerably slower than usual indexing
    accmap = np.repeat(np.arange(10000), 20)
    a = np.arange(accmap.size, dtype=int)
    vals = aggregate_weave(accmap, a)

    t0 = timeit.Timer(lambda: vals[accmap]).timeit(number=100)
    t1 = timeit.Timer(lambda: unpack(accmap, vals)).timeit(number=100)
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])
    # This was a speedup once, but using openblas speeds up numpy greatly
    # So let's just make sure it's not a too big drawback
    assert t0 / t1 > 0.5


@pytest.skip("Needs reimplementation")
def test_unpack_downscaled():
    accmap = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
    vals = aggregate_weave(accmap, np.arange(accmap.size), mode='downscaled')
    unpacked = unpack(accmap, vals, mode='downscaled')
    np.testing.assert_array_equal(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21]))
