"""The tools of the groups: the ways to lay them out and to number them.

``multi_arange``, ``label_contiguous_1d``, ``relabel_groups_unique`` and
``relabel_groups_masked`` are public names of the package - they are
exported by ``numpy_groupies`` and documented with an example each.  The
examples are the specification here: they are run as the tests they are.
"""

import numpy as np
import pytest

import numpy_groupies as npg


def test_multi_arange_counts_the_values_of_every_group():
    """The values of every group, numbered from zero."""
    n = np.array([0, 0, 3, 0, 0, 2, 0, 2, 1])
    np.testing.assert_array_equal(npg.multi_arange(n), [0, 1, 2, 0, 1, 0, 1, 0])


def test_multi_arange_of_an_empty_array():
    """No group, no value."""
    np.testing.assert_array_equal(npg.multi_arange(np.array([], dtype=int)), [])


def test_multi_arange_refuses_an_array_of_more_than_one_dimension():
    with pytest.raises(ValueError, match="1d array"):
        npg.multi_arange(np.zeros((2, 2), dtype=int))


def test_label_contiguous_1d_numbers_the_blocks_of_a_mask():
    """Every block of true values gets a number of its own."""
    mask = np.array([False, True, True, False, False, True, False, False, False, True, True, True])
    np.testing.assert_array_equal(npg.label_contiguous_1d(mask), [0, 1, 1, 0, 0, 2, 0, 0, 0, 3, 3, 3])


def test_label_contiguous_1d_numbers_the_blocks_of_the_values():
    """Every block of equal values gets a number of its own."""
    values = np.array([0, 3, 3, 0, 0, 5, 5, 5, 1, 1, 0, 2])
    np.testing.assert_array_equal(npg.label_contiguous_1d(values), [0, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 4])


def test_label_contiguous_1d_refuses_an_array_of_more_than_one_dimension():
    with pytest.raises(ValueError, match="1d masks only"):
        npg.label_contiguous_1d(np.zeros((2, 2), dtype=bool))


def test_relabel_groups_unique_fills_the_gaps():
    """The groups that are named are numbered without a gap."""
    group_idx = np.array([0, 3, 3, 3, 0, 2, 5, 2, 0, 1, 1, 0, 3, 5, 5])
    np.testing.assert_array_equal(
        npg.relabel_groups_unique(group_idx),
        [0, 3, 3, 3, 0, 2, 4, 2, 0, 1, 1, 0, 3, 4, 4],
    )


def test_relabel_groups_masked_removes_the_groups_of_the_mask():
    """The groups the mask drops are taken out, and the rest close the gap."""
    group_idx = np.array([0, 3, 3, 3, 0, 2, 5, 2, 0, 1, 1, 0, 3, 5, 5])
    keep_group = np.array([0, 1, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(
        npg.relabel_groups_masked(group_idx, keep_group),
        [0, 2, 2, 2, 0, 0, 4, 0, 0, 1, 1, 0, 2, 4, 4],
    )


def test_relabel_groups_masked_keeps_the_first_group():
    """The first group cannot be removed - the mask drops it, the function keeps it.

    The mask of the caller asks for the group 0 to be gone, and the answer
    keeps it: the group 0 is the one every other group is counted from, so
    it stays - and the labels that name it stay the same.
    """
    group_idx = np.array([0, 3, 3, 3, 0, 2, 5, 2, 0, 1, 1, 0, 3, 5, 5])
    keep_group = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(
        npg.relabel_groups_masked(group_idx, keep_group),
        [0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0, 1, 3, 3],
    )
    # the mask of the caller is not changed by the call
    np.testing.assert_array_equal(keep_group, [0, 0, 0, 1, 1, 1])


def test_relabel_groups_masked_takes_an_integer_mask():
    """A mask of the caller that is no boolean one is read as one."""
    group_idx = np.array([0, 1, 2])
    np.testing.assert_array_equal(npg.relabel_groups_masked(group_idx, np.array([1, 0, 1])), [0, 0, 1])
