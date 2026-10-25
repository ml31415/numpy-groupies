"""The participants of the benchmark suite.

The benchmark is a tool of its own and not part of the package, but the
participants are the implementations of this package - a mistake in one of
them would show up in the numbers of the table and nowhere else.
"""

import numpy as np
import pytest

from numpy_groupies.benchmarks import COLUMN_ORDER, _numbagg, select_funcs


def test_the_function_names_are_the_ones_of_the_package():
    from numpy_groupies.benchmarks import func_list

    assert select_funcs(None) == tuple(func_list)
    namen = [getattr(func, "__name__", func) for func in func_list]
    assert "min" in namen and "max" in namen
    gewaehlt = select_funcs("min,max")
    assert [getattr(func, "__name__", func) for func in gewaehlt] == ["min", "max"]


def test_a_participant_that_writes_into_the_data_is_undone():
    # the participants must not write into the data: one that does would
    # change the results of the runs that follow, silently.  The snapshot is
    # the bytes of the data as they were at the start of the run, and a change
    # is put back.
    from numpy_groupies.benchmarks import _byte_snapshot, _restore

    values = np.random.default_rng(100).random(1000)
    snapshot = _byte_snapshot(values)
    values[0] = 42.0
    assert _restore((values,), snapshot)
    np.testing.assert_array_equal(values, np.frombuffer(snapshot[0], dtype=values.dtype))


def test_the_pandas_column_does_not_copy_the_data():
    # the benchmark goes from the array to the array: a participant that
    # copies the data pays for the copy, and the pandas one does not have to
    # - its Series takes the buffer of the array.
    import pandas as pd

    values = np.random.default_rng(100).random(1000)
    series = pd.Series(values, copy=False)
    assert np.shares_memory(series.to_numpy(), values)


@pytest.mark.skipif(_numbagg is None, reason="numbagg is not installed")
def test_the_numbagg_column_is_the_one_of_numbagg():
    # the benchmark does not hold numbagg at one thread and does not measure
    # it twice: the threads of the machine are numbagg's own business, and a
    # second column would only show the same numbers twice - on a
    # one-dimensional input there is nothing for the parallel target of
    # numbagg to split
    import numbagg

    for name, func in _numbagg.funcs.items():
        assert isinstance(func, numbagg.decorators.groupndreduce), name
    # and the second column is gone
    assert not hasattr(_numbagg, "aggregate_mt")
    assert "numbagg-mt" not in COLUMN_ORDER


@pytest.mark.skipif(_numbagg is None, reason="numbagg is not installed")
def test_the_numbagg_column_reports_what_it_has_not():
    # an operation without a numbagg counterpart is reported as "----", not as
    # an error
    with pytest.raises(NotImplementedError):
        _numbagg.aggregate(np.array([0, 1]), np.array([1.0, 2.0]), func="median")
