import numpy as np
import pytest

from numpy_groupies import aggregate_np


@pytest.mark.parametrize("func", ["sum", "nansum"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("fill_value", [0, 7])
def test_wide_integer_grouped_sum(func, dtype, fill_value):
    groups = np.array([0, 2, 0, 2])
    values = np.array([2**60 + 1, 2**53 + 1, 2**60 + 2, 2**53 + 2], dtype=dtype)
    before = values.copy()
    result = aggregate_np(groups, values, func=func, size=4, fill_value=fill_value)
    expected = [sum(int(v) for v in values[groups == g]) if g in groups else fill_value for g in range(4)]
    np.testing.assert_array_equal(result, np.array(expected, dtype=result.dtype))
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("func", ["sum", "nansum"])
def test_wide_signed_integer_cancellation(func):
    groups = np.array([0, 1, 0, 1])
    values = np.array([2**60 + 1, np.iinfo(np.int64).min, -(2**60), np.iinfo(np.int64).max])
    result = aggregate_np(groups, values, func=func)
    np.testing.assert_array_equal(result, [1, -1])


def test_integer_grouped_sum_axis():
    groups = np.array([0, 0])
    values = np.array([[2**60 + 1, -(2**60)], [2**53 + 1, -(2**53)]])
    result = aggregate_np(groups, values, func="sum", axis=1)
    np.testing.assert_array_equal(result, [[1], [1]])


def test_integer_grouped_sum_bound_within_float_precision():
    values = np.array([-(2**52), 2**52], dtype=np.int64)
    np.testing.assert_array_equal(aggregate_np(np.zeros(2, dtype=int), values, func="sum"), [0])
