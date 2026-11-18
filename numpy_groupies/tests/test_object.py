"""The object columns: values of no dtype of their own.

An object column holds whatever it is given - the numbers of a mixed column,
the strings of a text column, the nan of a float column wrapped in objects.
The numpy backend orders them the way python orders them, the pure python one
does the same, and the numba one has no dtype for them and says so in its own
words.

The tests below pin the answers of the order statistics, the nan of a group,
the fill of a group without values and the refusal of the backend that cannot
compile them.
"""

import numpy as np
import pytest

from numpy_groupies import npg_numba
from numpy_groupies.tests.helpers import impl_name

# the labels of the tests below: two groups with values, one without
GROUP_IDX = np.array([0, 0, 1])
SIZE = 3


@pytest.fixture
def skip_numba(impl):
    """The numba backend compiles with the dtype of the values, and an object
    column has none it can compile - it refuses the column in its own words.
    """
    if impl_name(impl) == "numba":
        pytest.skip("the numba implementation has no dtype for an object column")


def _object(values):
    return np.array(values, dtype=object)


def _assert_same(got, want):
    """Compare two answers of an object column: the nan of one of them is the
    nan of the other - the values of an object column are compared one by one,
    numpy's own comparison of two arrays knows no nan of an object dtype.
    """
    assert len(got) == len(want)
    for one, other in zip(got, want):
        if isinstance(one, float) and one != one:
            assert isinstance(other, float) and other != other
        else:
            assert one == other


def test_min_and_max_of_object_column(impl, skip_numba):
    # the values are ordered the way python orders them
    a = _object([3, 1, 2])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="min", size=SIZE), [1, 2, np.nan])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="max", size=SIZE), [3, 2, np.nan])


def test_min_and_max_of_text_column(impl, skip_numba):
    a = _object(["b", "a", "c"])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="min", size=SIZE), ["a", "c", np.nan])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="max", size=SIZE), ["b", "c", np.nan])


def test_nan_wins_in_object_group(impl, skip_numba):
    # a nan of a group wins over its other values, like numpy's min and max
    a = _object([1.0, np.nan, 3.0])
    res = impl.aggregate(GROUP_IDX, a, func="min", size=SIZE)
    assert np.isnan(res[0])
    assert res[1] == 3.0


def test_nan_wins_over_text(impl, skip_numba):
    # the nan is not compared with the other values: numpy's object loop
    # would raise for a text and a nan
    a = _object(["a", np.nan, "c"])
    res = impl.aggregate(GROUP_IDX, a, func="min", size=SIZE)
    assert np.isnan(res[0])
    assert res[1] == "c"


def test_nan_variants_skip_the_nan(impl, skip_numba):
    a = _object([1.0, np.nan, 3.0])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="nanmin", size=SIZE), [1.0, 3.0, np.nan])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="nanmax", size=SIZE), [1.0, 3.0, np.nan])


def test_arg_extremes_of_object_column(impl, skip_numba):
    a = _object([3, 1, 2])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="argmin", size=SIZE), [1, 2, -1])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="argmax", size=SIZE), [0, 2, -1])


def test_group_with_nan_has_no_position(impl, skip_numba):
    # a group holding a nan has no extreme to point at
    a = _object([1.0, np.nan, 3.0])
    _assert_same(impl.aggregate(GROUP_IDX, a, func="argmin", size=SIZE), [-1, 2, -1])
    # the nan variants point at the extreme of the other values
    _assert_same(impl.aggregate(GROUP_IDX, a, func="nanargmin", size=SIZE), [0, 2, -1])


def test_fill_of_group_without_values(impl, skip_numba):
    # an object column holds a nan, so the nan is the fill of the group
    # without values
    a = _object([1, 2, 3])
    res = impl.aggregate(GROUP_IDX, a, func="min", size=SIZE)
    assert np.isnan(res[2])


def test_numba_refuses_object_column(impl):
    if impl_name(impl) != "numba":
        pytest.skip("the numba implementation is the one that refuses an object column")
    a = _object([1, 2, 3])
    with pytest.raises(TypeError, match="object column"):
        impl.aggregate(GROUP_IDX, a, func="min", size=SIZE)


def test_numba_collects_object_column(impl):
    # the collecting function is the one call the numba backend hands to the
    # numpy implementation
    if impl_name(impl) != "numba":
        pytest.skip("the numba implementation is the one that collects an object column")
    a = _object([1, 2, 3])
    res = impl.aggregate(GROUP_IDX, a, func="array", size=SIZE)
    np.testing.assert_array_equal(res[0], np.array([1, 2], dtype=object))


def test_bare_kernel_refuses_object_column():
    a = _object([1, 2, 3])
    with pytest.raises(TypeError, match="object column"):
        npg_numba.min(GROUP_IDX, a, SIZE)
