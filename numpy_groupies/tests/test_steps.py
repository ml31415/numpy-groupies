"""The run edges of ``group_idx``, reported by every backend.

``step_count`` and ``step_indices`` are the public face of the group
boundaries - the numpy and the numba backend answer them the same way.
The sort key of the numpy backend, which narrows the group indices for the
sort, is checked here as well.
"""

import numpy as np

from numpy_groupies.aggregate_numpy.steps import sort_key


def test_step_indices_length(impl, rng):
    """``step_count`` and ``step_indices`` length agree with a numpy reference."""
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
    for _ in range(20):
        rng.shuffle(group_idx)
        step_cnt_ref = np.count_nonzero(np.diff(group_idx))
        assert impl.step_count(group_idx) == step_cnt_ref + 1
        assert len(impl.step_indices(group_idx)) == step_cnt_ref + 2


def test_step_indices_fields(impl):
    """``step_indices`` returns the run edges of ``group_idx``."""
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
    steps = impl.step_indices(group_idx)
    np.testing.assert_array_equal(steps, np.array([0, 3, 5, 7, 8, 9, 11]))


def test_sort_key_narrows_the_dtype():
    """The sort key is as narrow as the group indices allow."""
    # the groups fit into an int16
    group_idx = np.array([0, 1, 1, 2], dtype=np.int64)
    assert sort_key(group_idx).dtype == np.int16
    # they fit into an int32, but not into an int16
    group_idx = np.array([0, 1, 1, 70000], dtype=np.int64)
    assert sort_key(group_idx).dtype == np.int32
    # they fit into neither of them
    group_idx = np.array([0, 1, 1, 2**40], dtype=np.int64)
    assert sort_key(group_idx).dtype == np.int64
    # negative groups keep the dtype they came in
    group_idx = np.array([-1, 0, 1], dtype=np.int64)
    assert sort_key(group_idx).dtype == np.int64
    # the order of the values does not change
    group_idx = np.array([7, 3, 7, 1], dtype=np.int64)
    np.testing.assert_array_equal(np.argsort(sort_key(group_idx)), np.argsort(group_idx))
