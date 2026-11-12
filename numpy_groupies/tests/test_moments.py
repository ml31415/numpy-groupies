"""The skewness and the kurtosis of every group.

Both are moments of the deviations of the values of a group from their mean:
the skewness is the third over the second one to the three-halves power, the
kurtosis the fourth over the square of the second one, minus the three of a
normal distribution.  The correction for the size of the group is the one
pandas applies, so that the estimators are unbiased for a normal
distribution.

The tests compare against the group loop of :mod:`.reference`, which computes
the moments one group at a time.
"""

import numpy as np
import pytest

from numpy_groupies.tests.helpers import assert_matches_the_group_loop_values
from numpy_groupies.tests.reference import aggregate_ref


def _compare(impl, group_idx, a, func, size=None):
    """The result of the backend against the reference, group by group."""
    assert_matches_the_group_loop_values(impl, group_idx, a, func, size)


@pytest.mark.parametrize("func", ["skew", "kurt", "nanskew", "nankurt"])
def test_every_group(impl, block, func):
    """Every group answers with the moment of the values it holds."""
    _compare(impl, block.group_idx, block.a, func, block.size)


@pytest.mark.parametrize("func", ["skew", "kurt", "nanskew", "nankurt"])
def test_group_with_nans(impl, block, func):
    """The plain moments are poisoned by a nan, the nan variants skip it."""
    _compare(impl, block.group_idx, block.a_nans, func, block.size)


@pytest.mark.parametrize("func", ["skew", "kurt", "nanskew", "nankurt"])
def test_integers(impl, func):
    """The moments of integer values are the moments of the numbers they are."""
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    a = np.array([1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80])
    _compare(impl, group_idx, a, func)


@pytest.mark.parametrize("func", ["skew", "kurt", "nanskew", "nankurt"])
def test_narrow_integers_do_not_wrap_their_deviations(impl, func):
    """The deviations of the moments are carried in float64, not in the
    dtype of the values.

    A subtraction in the dtype of narrow signed values wraps - a 0 against an
    origin of 100 would be a -156 there, and the moments of the wrapped
    deviations are the moments of different values entirely.  The values
    below span more than the int8 around their group's first value, so a
    deviation computed in the input dtype is a wrap for sure.
    """
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([100, -100, 0, 50, -128, 127, -1, 0], dtype=np.int8)
    _compare(impl, group_idx, a, func)


def test_group_of_two_values_has_no_skewness(impl):
    """Two values have no skewness - the estimator needs three of them."""
    group_idx = np.array([0, 0, 1, 1, 1])
    a = np.array([1.0, 2.0, 1.0, 2.0, 5.0])
    got = impl.aggregate(group_idx, a, func="skew")
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1], aggregate_ref(group_idx, a, func="skew")[1])


def test_group_of_three_values_has_no_kurtosis(impl):
    """Three values have no kurtosis - the estimator needs four of them."""
    group_idx = np.array([0, 0, 0, 1, 1, 1, 1])
    a = np.array([1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 9.0])
    got = impl.aggregate(group_idx, a, func="kurt")
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1], aggregate_ref(group_idx, a, func="kurt")[1])


def test_group_without_spread_has_no_moment(impl):
    """A group whose values are all the same has no spread - and no moment."""
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 4.0])
    got = impl.aggregate(group_idx, a, func="skew")
    assert got[0] == 0.0
    got = impl.aggregate(group_idx, a, func="kurt")
    assert got[0] == 0.0


def test_group_of_equal_values_has_no_moment_either(impl):
    """The value 0.1 is the one whose sum is inexact.

    The sum of python floats is only exact since python 3.12, where it
    compensates the rounding of every step: on an older python the mean of
    a group of equal values is off by an ulp - and the moments of a group
    without spread are the noise of that ulp instead of zero.  The group is
    centered on its first value, so its deviations are exactly zero, and the
    mean of those deviations is taken off as well.
    """
    group_idx = np.array([0, 0, 0])
    a = np.array([0.1, 0.1, 0.1])
    got = impl.aggregate(group_idx, a, func="skew")
    assert got[0] == 0.0
    group_idx = np.array([0, 0, 0, 0, 0, 0])
    a = np.array([1.1] * 6)
    got = impl.aggregate(group_idx, a, func="kurt")
    assert got[0] == 0.0


def test_absent_group_takes_fill_value(impl):
    """A group without values takes the fill value, like every other function."""
    group_idx = np.array([0, 0, 0, 2, 2, 2])
    a = np.array([1.0, 2.0, 5.0, 1.0, 2.0, 9.0])
    got = impl.aggregate(group_idx, a, func="skew", size=4)
    assert np.isnan(got[1])
    assert np.isnan(got[3])


def test_complex_values_are_refused(impl):
    """The moments of a complex value are not defined - the backends say so."""
    group_idx = np.array([0, 0, 0, 0])
    a = np.array([1 + 1j, 2 + 2j, 3 + 4j, 4 + 8j])
    for func in ("skew", "kurt"):
        with pytest.raises(TypeError, match="complex"):
            impl.aggregate(group_idx, a, func=func)


def test_kurtosis_is_excess_of_normal_distribution(impl, rng):
    """The kurtosis of a normal distribution is zero - the excess over three."""
    # the excess kurtosis of a normal sample is zero up to its own error,
    # which is sqrt(24/n) - a hundred thousand values put the tolerance of
    # the test six standard errors away, so the sample of the seed cannot
    # fail it
    group_idx = np.zeros(100_000, dtype=int)
    a = rng.normal(size=100_000)
    got = impl.aggregate(group_idx, a, func="kurt")
    assert abs(got[0]) < 0.1
