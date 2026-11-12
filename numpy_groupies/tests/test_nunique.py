"""The number of distinct values of every group.

The distinct values are the ones that differ from their predecessor in the
ordered group: the values that are equal count once, a zero and a minus zero
among them, an integer and the float of the same number as well.  The nans
are one value among themselves - a nan in either part of a complex value is a
nan - and the plain function counts them, while its nan variant leaves them
out.

The tests compare against the group loop of :mod:`.reference`, which counts
the distinct values one group at a time.
"""

import numpy as np
import pytest

from numpy_groupies.tests.helpers import assert_matches_the_group_loop_values


def _compare(impl, group_idx, a, func, size=None):
    """The result of the backend against the reference, group by group."""
    assert_matches_the_group_loop_values(impl, group_idx, a, func, size)


def test_distinct_values_of_every_group(impl, block):
    """Every group answers with the number of the distinct values it holds."""
    _compare(impl, block.group_idx, block.a, "nunique", block.size)


def test_nans_of_group_are_one_value(impl, block):
    """The nans of a group count once among themselves, like pandas counts them."""
    _compare(impl, block.group_idx, block.a_nans, "nunique", block.size)


def test_nan_variant_leaves_nans_out(impl, block):
    """The nan variant counts the distinct values of the values that are left."""
    _compare(impl, block.group_idx, block.a_nans, "nannunique", block.size)


def test_distinct_values_of_complex_group(impl, block):
    """A complex group counts its distinct values, a nan in either part being one."""
    _compare(impl, block.group_idx, block.a_complex, "nunique", block.size)
    _compare(impl, block.group_idx, block.a_complex, "nannunique", block.size)


def test_equal_values_count_once(impl):
    """The values that are equal are one value, whatever their type is."""
    group_idx = np.zeros(6, dtype=int)
    a = np.array([1.0, 1, 1.0, 0.0, -0.0, np.nan])
    _compare(impl, group_idx, a, "nunique")


def test_distinct_values_of_integers_and_booleans(impl):
    """The integers and the booleans count as the numbers they are."""
    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array([1, 2, 2, True, False, True])
    _compare(impl, group_idx, a, "nunique")


def test_group_of_nans_is_one_value(impl):
    """The plain function counts the nans of a group as one value, the nan variant leaves them out."""
    group_idx = np.zeros(3, dtype=int)
    a = np.array([np.nan, np.nan, np.nan])
    _compare(impl, group_idx, a, "nunique")
    _compare(impl, group_idx, a, "nannunique")


def test_result_is_count(impl):
    """The number of the distinct values is a count, and so an integer."""
    group_idx = np.array([0, 0, 1])
    a = np.array([1.0, 1.0, 2.0])
    res = impl.aggregate(group_idx, a, func="nunique")
    assert np.issubdtype(np.asarray(res).dtype, np.integer)


def test_absent_group_takes_fill_value(impl):
    """A group without values has no distinct values, and takes the fill value."""
    group_idx = np.array([0, 0])
    a = np.array([1.0, 2.0])
    res = impl.aggregate(group_idx, a, func="nunique", size=3)
    np.testing.assert_array_equal(res, [2, 0, 0])


def test_unconvertible_fill_value_is_refused(impl):
    """A count cannot hold a nan, and the fill value says so."""
    group_idx = np.array([0, 0])
    a = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        impl.aggregate(group_idx, a, func="nunique", size=3, fill_value=np.nan)
