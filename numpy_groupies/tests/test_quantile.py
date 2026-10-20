"""The quantile and the percentile of every group.

The quantile is an order statistic like the median: it names a position
between the values of a group, and the two values it falls between are
interpolated the way numpy interpolates them.  The tests compare against
``np.quantile`` over the values of every group, which the reference module
computes one group at a time.
"""

import numpy as np
import pytest

from numpy_groupies.aggregate_purepy import aggregate as aggregate_py
from numpy_groupies.tests.reference import aggregate_ref

_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
_METHODS = ("linear", "lower", "higher", "midpoint", "nearest")


def _compare(impl, block, func, **kwargs):
    """The result of the backend against the reference, group by group."""
    got = impl.aggregate(block.group_idx, block.a, func=func, size=block.size, **kwargs)
    want = aggregate_ref(block.group_idx, block.a, func=func, size=block.size, **kwargs)
    np.testing.assert_allclose(got, want, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("q", _QUANTILES)
@pytest.mark.parametrize("method", _METHODS)
def test_the_quantile_of_every_group_is_the_one_numpy_gives(impl, block, q, method):
    """Every group answers with the quantile of its values, interpolated as numpy does."""
    _compare(impl, block, "quantile", q=q, method=method)


@pytest.mark.parametrize("q", _QUANTILES)
def test_the_percentile_of_every_group_is_the_one_numpy_gives(impl, block, q):
    """The percentile is the quantile in percent."""
    _compare(impl, block, "percentile", q=q)


def test_the_quantile_without_a_quantile_asked_for_is_the_median(impl, block):
    """Without a quantile of its own the function answers the median."""
    _compare(impl, block, "quantile")


def test_the_nanquantile_skips_the_nans(impl, block):
    """The nan variant leaves the nans out and answers for the values that are left."""
    got = impl.aggregate(block.group_idx, block.a_nans, func="nanquantile", q=0.5, size=block.size)
    want = aggregate_ref(block.group_idx, block.a_nans, func="nanquantile", q=0.5, size=block.size)
    np.testing.assert_allclose(got, want, rtol=1e-12, equal_nan=True)


def test_the_nanpercentile_skips_the_nans(impl, block):
    """The nan variant leaves the nans out and answers for the values that are left."""
    got = impl.aggregate(block.group_idx, block.a_nans, func="nanpercentile", q=25.0, size=block.size)
    want = aggregate_ref(block.group_idx, block.a_nans, func="nanpercentile", q=25.0, size=block.size)
    np.testing.assert_allclose(got, want, rtol=1e-12, equal_nan=True)


def test_a_quantile_out_of_the_range_is_refused(impl, block):
    """A quantile below 0 or above 1 is not a quantile."""
    with pytest.raises(ValueError, match="range"):
        impl.aggregate(block.group_idx, block.a, func="quantile", q=1.5, size=block.size)
    with pytest.raises(ValueError, match="range"):
        impl.aggregate(block.group_idx, block.a, func="percentile", q=-1.0, size=block.size)


def test_an_unknown_interpolation_method_is_refused(impl, block):
    """A method numpy does not know is not one this package knows either."""
    with pytest.raises(ValueError, match="interpolation"):
        impl.aggregate(block.group_idx, block.a, func="quantile", q=0.5, method="cubic", size=block.size)


def test_the_pure_python_backend_says_that_it_does_not_know_the_quantile(block):
    """The pure python implementation does not do the quantile, and says so."""
    with pytest.raises(NotImplementedError, match="quantile"):
        aggregate_py(block.group_idx, block.a, func="quantile", q=0.5, size=block.size)


def test_the_quantile_of_a_complex_group_keeps_the_complex_dtype(impl, block):
    """The complex groups are ordered lexicographically, like the median orders them.

    numpy's own quantile takes real values only, so the complex case is
    measured against the numpy implementation of this package.
    """
    from numpy_groupies.aggregate_numpy import aggregate as aggregate_np

    got = impl.aggregate(block.group_idx, block.a_complex, func="quantile", q=0.5, size=block.size)
    want = aggregate_np(block.group_idx, block.a_complex, func="quantile", q=0.5, size=block.size)
    np.testing.assert_allclose(got, want, rtol=1e-12, equal_nan=True)


def test_the_quantile_of_integer_values_is_a_float(impl):
    """The quantile of integers is a float, as it is in numpy."""
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    got = impl.aggregate(group_idx, a, func="quantile", q=0.5, size=2)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, [2.0, 4.5])
