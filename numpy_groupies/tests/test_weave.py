import numpy as np

from ..aggregate_weave import step_indices, step_count


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
