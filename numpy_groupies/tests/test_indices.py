import numpy as np


def test_step_indices_length(impl):
    """``step_count`` and ``step_indices`` length agree with a numpy reference."""
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
    for _ in range(20):
        np.random.shuffle(group_idx)
        step_cnt_ref = np.count_nonzero(np.diff(group_idx))
        assert impl.step_count(group_idx) == step_cnt_ref + 1
        assert len(impl.step_indices(group_idx)) == step_cnt_ref + 2


def test_step_indices_fields(impl):
    """``step_indices`` returns the run edges of ``group_idx``."""
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
    steps = impl.step_indices(group_idx)
    np.testing.assert_array_equal(steps, np.array([0, 3, 5, 7, 8, 9, 11]))
