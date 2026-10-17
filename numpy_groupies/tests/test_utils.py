"""The functions the package exports besides the aggregations.

``step_count`` and ``step_indices`` report the runs of ``group_idx``, ``unpack``
inverts a group reduction - every backend provides them.
"""

import numpy as np

from numpy_groupies.tests.helpers import random_values
from numpy_groupies.utils import check_dtype, unpack


def test_check_dtype():
    dtype = check_dtype(None, "mean", np.arange(10, dtype=int), 10)
    assert np.issubdtype(dtype, np.floating)


def test_unpack(rng):
    """Keep this test, in case unpack might get reimplemented again at some point."""
    group_idx = np.arange(10)
    rng.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 3)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_unpack_long(rng):
    group_idx = np.repeat(np.arange(10000), 20)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


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
