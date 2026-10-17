import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("func", ["sum", "mean", "argmax"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_axis_output_order_preserves_groups(impl, shape, broadcast, func, order):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    labels = np.array([0, 1, 0, 1])
    if broadcast:
        labels = np.broadcast_to(labels, shape).copy()
        labels = (labels + np.arange(np.prod(shape[:-1])).reshape(shape[:-1] + (1,))) % 2
    expected = np.empty(shape[:-1] + (2,))
    for loc in np.ndindex(shape[:-1]):
        row_labels = labels[loc] if broadcast else labels
        for group in range(2):
            positions = np.flatnonzero(row_labels == group)
            values = data[loc][positions]
            if func == "argmax":
                value = positions[np.argmax(values)]
            else:
                value = getattr(np, func)(values)
            expected[loc + (group,)] = value

    actual = impl.aggregate(labels, data, func=func, axis=-1, order=order)

    np.testing.assert_allclose(actual, expected)
    if func != "argmax":
        assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
