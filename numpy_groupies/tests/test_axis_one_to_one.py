import numpy as np
import pytest


@pytest.mark.parametrize(
    "func", ["cumsum", np.cumsum, "nancumsum", np.nancumsum, "sort", np.sort, "cumprod", "cummin", "cummax"]
)
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("order", ["C", "F"])
def test_one_to_one_axis_shape(impl, func, axis, order):
    rows = np.array([[3.0, 1.0, 5.0, 2.0], [20.0, 40.0, 10.0, 30.0]])
    data = rows if axis == -1 else rows.T
    labels = np.array([0, 1, 0, 1])
    name = func if isinstance(func, str) else func.__name__
    expected = np.empty_like(rows)
    for row in range(2):
        for group in range(2):
            positions = np.flatnonzero(labels == group)
            values = rows[row, positions]
            if name in ("cummin", "cummax"):
                operation = np.minimum if name == "cummin" else np.maximum
                values = operation.accumulate(values)
            else:
                values = getattr(np, name)(values)
            expected[row, positions] = values
    if axis == 0:
        expected = expected.T

    actual = impl.aggregate(labels, data, func=func, axis=axis, order=order)

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == data.shape
