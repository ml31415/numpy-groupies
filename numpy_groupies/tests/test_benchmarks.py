"""The participants of the benchmark suite.

The benchmark is a tool of its own and not part of the package, but the
participants are the implementations of this package - a mistake in one of
them would show up in the numbers of the table and nowhere else.
"""

import numpy as np
import pytest

from numpy_groupies.benchmarks import _numbagg, select_funcs


def test_the_function_names_are_the_ones_of_the_package():
    from numpy_groupies.benchmarks import func_list

    assert select_funcs(None) == tuple(func_list)
    namen = [getattr(func, "__name__", func) for func in func_list]
    assert "min" in namen and "max" in namen
    gewaehlt = select_funcs("min,max")
    assert [getattr(func, "__name__", func) for func in gewaehlt] == ["min", "max"]


@pytest.mark.skipif(_numbagg is None, reason="numbagg is not installed")
def test_the_two_numbagg_columns_give_the_same_result():
    # the columns differ only in the number of the threads numbagg may use
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 10_000)
    values = rng.random(10_000)
    for func in ("sum", "prod", "len", "all", "any", "max", "min", "mean"):
        eins = _numbagg.aggregate(group_idx, values, func=func)
        viele = _numbagg.aggregate_mt(group_idx, values, func=func)
        np.testing.assert_allclose(eins, viele, rtol=1e-12, equal_nan=True)


@pytest.mark.skipif(_numbagg is None, reason="numbagg is not installed")
def test_the_numbagg_column_is_the_one_threaded_one():
    # the functions of the one-threaded column switch the thread count of
    # numba, the ones of the other column do not
    for name, func in _numbagg.funcs.items():
        assert hasattr(func, "__wrapped__"), name
        assert func.__wrapped__ is _numbagg.funcs_mt[name], name


@pytest.mark.skipif(_numbagg is None, reason="numbagg is not installed")
def test_the_numbagg_column_reports_what_it_has_not():
    # an operation without a numbagg counterpart is reported as "----", not as
    # an error
    with pytest.raises(NotImplementedError):
        _numbagg.aggregate(np.array([0, 1]), np.array([1.0, 2.0]), func="median")
