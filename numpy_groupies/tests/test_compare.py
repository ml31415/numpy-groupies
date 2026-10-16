"""
Compare the results of all implementations against a reference backend:
aggregate_numpy for all backends, and aggregate_purepy for numpy itself.
Implementations signal missing functionality with NotImplementedError,
which the backend proxies translate into skips.
"""

from itertools import product

import numpy as np
import pytest

from numpy_groupies.tests.helpers import (
    _BackendProxy,
    aggregate_numpy,
    aggregate_purepy,
    func_list,
    is_numpy,
    is_purepy,
    random_group_idx,
    random_values,
)


@pytest.fixture
def reference(impl):
    """The reference backend as ``_BackendProxy``: numpy for all backends,
    purepy for numpy itself."""
    if is_numpy(impl):
        return _BackendProxy(aggregate_purepy)
    return _BackendProxy(aggregate_numpy)


@pytest.fixture(scope="module")
def compare_data(rng):
    """Shared random data block for the comparisons in this module."""
    group_idx = np.repeat(rng.permutation(1000), 20)
    a = random_values(rng, group_idx.size)
    nana = a.copy()
    nana[::3] = np.nan
    nana[: a.size // 2] = np.nan
    return group_idx, a, nana


@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.deselect_if(func=is_purepy)
@pytest.mark.parametrize("fill_value", [0, 1, np.nan])
@pytest.mark.parametrize("func", func_list, ids=lambda x: getattr(x, "__name__", x))
def test_cmp(impl, reference, compare_data, func, fill_value, decimal=10):
    is_nanfunc = "nan" in getattr(func, "__name__", func)
    if is_nanfunc and "arg" in func and is_purepy(reference):
        pytest.skip("purepy nanargmin/nanargmax deviate from numpy without raising")
    group_idx, a, nana = compare_data
    a = nana if is_nanfunc else a
    try:
        ref = reference.aggregate(group_idx, a, func=func, fill_value=fill_value)
    except ValueError:
        # fill_value does not fit the result dtype (e.g. nan into int) - the
        # subject must raise the same error. NotImplementedError becomes a
        # skip via the backend proxy, even inside pytest.raises.
        with pytest.raises(ValueError):
            impl.aggregate(group_idx, a, func=func, fill_value=fill_value)
    else:
        # NotImplementedError is translated into a skip by the backend proxy
        try:
            res = impl.aggregate(group_idx, a, func=func, fill_value=fill_value)
        except ValueError:
            if is_purepy(reference) and np.isnan(fill_value):
                pytest.skip(
                    "pure python version uses lists and does not raise ValueErrors when inserting nan into integers"
                )
            else:
                raise
        if isinstance(ref, np.ndarray):
            assert res.dtype == ref.dtype
        np.testing.assert_allclose(res, ref, rtol=10**-decimal)


@pytest.mark.deselect_if(func=is_purepy)
@pytest.mark.parametrize(["ndim", "order"], list(product([2, 3], ["C", "F"])))
def test_cmp_ndim(impl, reference, rng, ndim, order, outsize=100, decimal=14):
    nindices = int(outsize**ndim)
    outshape = tuple([outsize] * ndim)
    group_idx = random_group_idx(rng, (ndim, nindices), outsize)
    a = random_values(rng, group_idx.shape[1])

    res = impl.aggregate(group_idx, a, size=outshape, order=order)
    ref = reference.aggregate(group_idx, a, size=outshape, order=order)
    if ndim > 1 and order == "F":
        # 1d arrays always return False here
        assert np.isfortran(res)
    else:
        assert not np.isfortran(res)
    assert res.shape == outshape
    np.testing.assert_array_almost_equal(res, ref, decimal=decimal)
