"""The collecting function - the one whose result is a sequence per group.

``array`` (and its aliases ``list``, ``split``, ``splice``) hands out the
values of every group instead of reducing them to one number, so its result
has a shape the other functions do not have: a sequence per group, which the
two numpy-based backends hold in an object array and the pure python one in a
list of lists.  The tests below therefore look at the groups and not at the
container - and the comparison of two such results is the helper of this
module, which the other test modules use as well.
"""

import itertools

import numpy as np
import pytest

from numpy_groupies.tests.helpers import assert_same_result, is_purepy
from numpy_groupies.tests.reference import aggregate_ref


def test_collecting_gives_the_values_of_every_group(impl):
    group_idx = np.array([3, 1, 3, 0, 1, 3])
    a = np.arange(6.0)
    res = impl.aggregate(group_idx, a, func="array")
    # the values of a group keep the order they appear in
    np.testing.assert_array_equal(res[3], [0.0, 2.0, 5.0])
    np.testing.assert_array_equal(res[1], [1.0, 4.0])
    # a group without values holds an empty sequence
    assert len(res[2]) == 0


@pytest.mark.parametrize("func", ["array", list, "split", "splice", np.array, np.asarray])
def test_collecting_aliases(impl, func):
    group_idx = np.array([0, 1, 1])
    a = np.arange(3.0)
    res = impl.aggregate(group_idx, a, func=func)
    np.testing.assert_array_equal(res[1], [1.0, 2.0])


@pytest.mark.parametrize("fill_value", [None, -1, np.nan], ids=["none", "int", "nan"])
def test_collecting_fill_value(impl, fill_value):
    # a group without values takes the fill value
    group_idx = np.array([0, 2, 2])
    a = np.arange(3.0)
    res = impl.aggregate(group_idx, a, func="array", fill_value=fill_value)
    if fill_value is None:
        # the fill value is a value of its own, not a request for a default
        assert res[1] is None
    else:
        np.testing.assert_equal(res[1], fill_value)


def test_collecting_empty_fill_value(impl):
    # an empty sequence leaves the empty sequence of the group alone
    group_idx = np.array([0, 2, 2])
    a = np.arange(3.0)
    res = impl.aggregate(group_idx, a, func="array", fill_value=[])
    assert len(res[1]) == 0


def test_collecting_multidimensional_labels(impl):
    # the labels name a group by coordinate, and a hole in them is a group
    # without values (issue #32)
    group_idx = np.array(list(itertools.product(range(3), range(4)))).T[:, :-1]
    a = np.arange(11, dtype=float)
    res = impl.aggregate(group_idx, a, func="array", size=(3, 4))
    assert res.shape == (3, 4)
    assert len(res[2, 3]) == 0


def test_collecting_keeps_the_dtype(impl):
    # the values are handed out as they are - the float16 of the input is not
    # widened to float32 the way the kernels widen it
    group_idx = np.array([0, 0, 1])
    a = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    res = impl.aggregate(group_idx, a, func="array")
    np.testing.assert_array_equal(res[0], [1.5, 2.5])
    if not is_purepy(impl):
        # the pure python implementation carries no dtype at all
        assert res[0].dtype == np.float16


def test_collecting_keeps_the_widest_float(impl):
    # float128 is no dtype a compiled driver can place - numba refuses it
    # while typing - so the values are copied around by the numpy
    # implementation, which keeps them exact
    group_idx = np.array([0, 0, 1])
    a = np.array([1.5, 2.5, 3.5], dtype=np.longdouble)
    res = impl.aggregate(group_idx, a, func="array")
    np.testing.assert_array_equal(res[0], [1.5, 2.5])
    if not is_purepy(impl):
        assert res[0].dtype == np.longdouble


@pytest.mark.parametrize("values", ["a_nans", "a_complex"], ids=["nan", "complex"])
def test_collecting_keeps_the_values_of_the_block(impl, block, values):
    """the values of a group are handed out as they are.

    A nan in either part of a complex value is a nan of the value - and the
    collecting function does not look at the values at all: it hands them out,
    nan parts and all.
    """
    data = getattr(block, values)
    res = impl.aggregate(block.group_idx, data, func="array", size=block.size)
    want = aggregate_ref(block.group_idx, data, func="array", size=block.size)
    assert_same_result(res, want, "array")


def test_collecting_refuses_a_scalar(impl):
    # a scalar stands for one value, which the collecting function cannot take
    group_idx = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="scalar inputs"):
        impl.aggregate(group_idx, 1.0, func="array")


def test_collecting_refuses_negative_labels(impl):
    group_idx = np.array([0, 1, 1])
    a = np.arange(3.0)
    with pytest.raises(ValueError, match="negative"):
        impl.aggregate(group_idx - 1, a, func="array")
