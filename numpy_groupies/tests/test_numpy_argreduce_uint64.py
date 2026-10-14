import numpy as np
import pytest

from numpy_groupies import aggregate_np


@pytest.mark.parametrize("base", [2**53, 2**63, 2**64 - 8])
@pytest.mark.parametrize("func", ["argmin", "argmax", "nanargmin", "nanargmax"])
@pytest.mark.parametrize("axis", [None, -1])
def test_uint64_argreductions_preserve_order(base, func, axis):
    group_idx = np.array([0, 0, 2, 2, 2, 2])
    a = np.array([base + 1, base, base + 2, base + 3, base + 2, base + 3], dtype=np.uint64)
    expected = np.array([1, -1, 2]) if func.endswith("min") else np.array([0, -1, 3])
    if axis is not None:
        a = np.stack([a, a])
        expected = np.stack([expected, expected])

    actual = aggregate_np(group_idx, a, func=func, axis=axis, fill_value=-1)

    np.testing.assert_array_equal(actual, expected)
