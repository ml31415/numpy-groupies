import numpy as np
import pytest

from .. import aggregate_np


@pytest.mark.parametrize("func", ["mean", "nanmean"])
@pytest.mark.parametrize("fill", [0, 7, np.nan])
def test_cancelling_integer_mean(func, fill):
    maximum = np.iinfo(np.int64).max
    values = np.array([maximum, -maximum + 1, maximum, maximum, maximum], dtype=np.int64)
    labels = np.array([0, 0, 1, 1, 1])
    original = values.copy()
    result = aggregate_np(labels, values, func=func, size=3, fill_value=fill)
    # Group1's sum exceeds int64, but its mean is representable in float64.
    np.testing.assert_array_equal(result, [0.5, float(maximum), fill])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("func", ["mean", "nanmean"])
def test_integer_mean_rounds_after_division(dtype, func):
    values = np.array([2**53 + 1, 0, 0], dtype=dtype)
    expected = (2**53 + 1) / 3
    result = aggregate_np(np.array([0, 0, 0]), values, func=func)
    np.testing.assert_array_equal(result, [expected])


@pytest.mark.parametrize("func", ["mean", "nanmean"])
def test_cancelling_integer_mean_axis(func):
    maximum = np.iinfo(np.int64).max
    values = np.array([[maximum, -maximum + 1], [maximum, -maximum + 3]], dtype=np.int64)
    result = aggregate_np(np.array([0, 0]), values, func=func, axis=1)
    np.testing.assert_array_equal(result, [[0.5], [1.5]])
