import pytest
import numpy as np

from . import aggregate_weave, aggregate_numba, _impl_name
_implementations = [aggregate_weave, aggregate_numba]


@pytest.fixture(params=_implementations, ids=_impl_name)
def aggregate_nb_wv(request):
    if request.param is None:
        pytest.xfail("Implementation not available")
    return request.param


def test_step_indices_length(aggregate_nb_wv):
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
    for _ in range(20):
        np.random.shuffle(group_idx)
        step_cnt_ref = np.count_nonzero(np.diff(group_idx))
        assert aggregate_nb_wv.step_count(group_idx) == step_cnt_ref + 1
        assert len(aggregate_nb_wv.step_indices(group_idx)) == step_cnt_ref + 2


def test_step_indices_fields(aggregate_nb_wv):
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
    steps = aggregate_nb_wv.step_indices(group_idx)
    np.testing.assert_array_equal(steps, np.array([ 0, 3, 5, 7, 8, 9, 11]))
