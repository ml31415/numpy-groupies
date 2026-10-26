"""The participants of the benchmark suite, and the calibration tool.

The benchmark is a tool of its own and not part of the package, but the
participants are the implementations of this package - a mistake in one of
them would show up in the numbers of the table and nowhere else.  The
calibration tool of the multithreaded kernels is not a participant either -
it measures the kernels against each other and fits the cost model to the
measurements, and a mistake in it would show up in the constants of the
model and nowhere else.  The parts of it that decide are tested in the
second section: the measurement of a pair of calls, the crossover of the
paths and the report of the decisions.
"""

import time

import numpy as np
import pytest

from numpy_groupies.benchmarks import COLUMN_ORDER, _numbagg, mt, select_funcs


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


# ---------------------------------------------------------------------------
# the calibration tool of the multithreaded kernels


def test_a_pair_is_measured_in_the_same_round():
    """The two calls of a pair are measured one after the other.

    A hitch of the machine then hits both of them and not only one of them,
    and the fastest of the rounds is the one that counts.
    """
    calls = []

    def first():
        calls.append("first")

    def second():
        calls.append("second")

    (fast_first, median_first), (fast_second, median_second) = mt.measure_pair(first, second, 4)
    assert calls == ["first", "second"] * 4
    assert fast_first <= median_first
    assert fast_second <= median_second


def test_the_measurement_reports_the_spread():
    """The fastest call and the median of the calls.

    The spread between them is the load of the machine - a machine that is
    busy measures a median that is far above the fastest call, and the
    numbers of such a run are worth nothing.
    """
    calls = []

    def call():
        # every second call is a slow one
        calls.append(len(calls))
        time.sleep(0.05 if len(calls) % 2 == 0 else 0.001)

    fastest, median = mt.measure(call, 6)
    # the fastest of the calls and the median of them - the median is the
    # slow one, the fastest is the quick one
    assert fastest > 0
    assert median > fastest
    assert median > 10 * fastest


def test_the_crossover_is_where_the_parallel_path_wins():
    """The size from which the parallel path is faster, and the one before.

    The ratio is the median of the measurements of a size, and the crossover
    is interpolated between the two sizes that straddle it.
    """
    rows = [
        {"n_values": 1000, "seq_us": 100.0, "mt_us": 100.0},
        {"n_values": 2000, "seq_us": 200.0, "mt_us": 160.0},
        {"n_values": 4000, "seq_us": 400.0, "mt_us": 200.0},
        {"n_values": 8000, "seq_us": 800.0, "mt_us": 300.0},
    ]
    cross, below = mt.crossover(rows, "seq_us", "mt_us", 0.8)
    # the ratio is 1.0 at 1000 values and 0.8 at 2000 - the margin is met
    # exactly at 2000
    assert cross == pytest.approx(2000.0)
    assert below == 1000.0


def test_a_crossover_the_grid_does_not_reach_is_no_number():
    rows = [{"n_values": 1000, "seq_us": 100.0, "mt_us": 100.0}]
    cross, below = mt.crossover(rows, "seq_us", "mt_us", 0.8)
    assert cross == float("inf")
    assert below == 1000.0
    assert "beyond the grid" in mt._values(cross)


def test_the_decision_of_the_package_is_the_one_of_its_model():
    """What the package decides for a measurement, and what its model says.

    The tool does not change the constants of the package - it says what
    they do, so that a decision can be compared with the measurement.
    """
    from numpy_groupies.aggregate_numba import mt as package_mt

    row = {"path": "chunked", "n_values": 1_000_000, "n_groups": 1000, "n_threads": 6}
    decision, modelled = mt._package_decision(row)
    assert decision == bool(package_mt.should_use_mt(1000, 6, 1_000_000))
    assert modelled == pytest.approx(package_mt.estimate_mt_ratio(1000, 6, 1_000_000))

    row = {"path": "gather", "n_values": 1_000_000, "n_groups": 1, "n_threads": 6}
    decision, modelled = mt._package_decision(row)
    assert decision == bool(
        1_000_000 >= package_mt.GATHER_MT_MIN_VALUES
        and package_mt.estimate_mt_ratio(0, 6, 1_000_000, 0) <= package_mt.MT_MAX_RATIO
    )


def test_the_model_of_the_group_path_has_a_term_for_the_groups():
    """The cost of a value and the cost of a group are two different things.

    With many groups and few values the second is the larger of the two, and
    a model that leaves it out is wrong where it matters.
    """
    model = mt.GroupModel(18.0, 50.0, 0.25, 0.4, 10.0)
    assert model.kernel_seq_us(1000, 10) == pytest.approx((1000 * 18.0 + 10 * 50.0) / 1e3)
    assert model.kernel_seq_us(1000, 10_000) == pytest.approx((1000 * 18.0 + 10_000 * 50.0) / 1e3)
    # the parallel path is faster than the sequential one, and the fixed cost
    # of the path is on top of it
    assert model.kernel_mt_us(100_000, 100, 6, 5.0) < model.kernel_seq_us(100_000, 100)
    # with a single thread the parallel path is the sequential one and the
    # fixed cost
    assert model.kernel_mt_us(100_000, 100, 1, 5.0) == pytest.approx(model.kernel_seq_us(100_000, 100) + 15.0)


def test_the_fit_of_the_group_path_finds_the_constants():
    """The fit finds the constants of a model that made the measurements."""
    from numpy_groupies.aggregate_numba import mt as package_mt

    truth = mt.GroupModel(18.0, 40.0, 0.3, 0.45, 12.0)
    rows = []
    for n_values in (10_000, 100_000, 1_000_000):
        for n_groups in (100, 10_000):
            for n_threads in (2, 6):
                seq_us = truth.kernel_seq_us(n_values, n_groups)
                mt_us = truth.kernel_mt_us(n_values, n_groups, n_threads, 5.0)
                rows.append(
                    {
                        "n_values": n_values,
                        "n_groups": n_groups,
                        "n_threads": n_threads,
                        "seq_us": seq_us,
                        "mt_us": mt_us,
                    }
                )
    overhead = {2: 5.0, 6: 5.0}
    model, loss = mt.fit_group(rows, overhead)
    assert loss < 1e-6
    assert model.ns_per_value == pytest.approx(18.0, rel=0.05)
    assert model.ns_per_group == pytest.approx(40.0, rel=0.05)
    assert model.share == pytest.approx(0.3, rel=0.05)
    assert model.per_thread == pytest.approx(0.45, rel=0.05)
    assert model.fixed_us == pytest.approx(12.0, rel=0.05)
    assert package_mt is not None


def test_the_spread_of_the_measurements_is_reported():
    """The quality of the measurements - the fastest against the median."""
    rows = [
        {"seq_us": 100.0, "mt_us": 50.0, "seq_median_us": 110.0, "mt_median_us": 60.0},
        {"seq_us": 200.0, "mt_us": 100.0, "seq_median_us": 600.0, "mt_median_us": 300.0},
    ]
    median, worst = mt._spread(rows)
    assert median >= 1.0
    assert worst >= median
