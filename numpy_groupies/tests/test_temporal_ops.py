"""The temporal operations of the numpy based backends: how they are built.

The temporals are built like the other operations of a backend: one
operation per function name, composed of the policy of
``numpy_groupies.temporal`` and the kernels of the backend, and put into
the dispatch of the backend beside the numeric ones.  The tests here pin
that structure - the operations are in the dispatch, they take the
arguments of the other operations of their backend, and the operations on
the int64 ticks are operations of the backend as well.
"""

import numpy as np
import pytest

from numpy_groupies import npg_numba, npg_numpy
from numpy_groupies.registry import function_names
from numpy_groupies.temporal import build_temporal_op
from numpy_groupies.tests.helpers import impl_name

# the pure python backend knows no temporal types - the same reason the
# other temporal modules of the suite leave it out, and the same fixture
pytestmark = pytest.mark.usefixtures("skip_purepy")


def test_every_function_has_temporal_operation(impl):
    """The temporal operation of every function of the package is in the
    dispatch of the backend, beside the numeric one."""
    dispatch = impl._dispatch
    for name in function_names():
        temporal_op = dispatch[name].temporal_op
        assert temporal_op is not None, name
        assert callable(temporal_op), name


def test_temporal_operations_are_not_aliases(impl):
    """The temporal operations are no aliases of the public functions - they
    are the other half of the operation of a function, and they are reached
    by their name."""
    dispatch = impl._dispatch
    if dispatch is None:
        pytest.skip("the pure python backend knows no temporal types")
    for name in function_names():
        temporal_op = dispatch[name].temporal_op
        assert temporal_op is not dispatch[name].op, name
        assert temporal_op.__name__ == f"temporal_{name}"


def test_backend_carrying_nat_policy_hands_rows_over():
    """The engine drops no NaT row when the operation of the backend on the
    ticks carries the NaT policy of the function itself - the kernel finds
    the NaT of a group and leaves it out on its own, so a copy of the data
    buys nothing.  A backend whose tick operation does not carry the policy
    gets the rows without the NaT, as before."""

    def op(group_idx, a, size, fill_value=None, dtype=None, **kwargs):
        seen.append(np.asarray(a))
        return np.zeros(size, dtype=np.int64)

    def call(op, group_idx, values, size, fill_value, dtype, order, axis, **kwargs):
        return op(group_idx, values, size, fill_value=fill_value, dtype=dtype, **kwargs)

    a = np.array([1, "NaT", 3], dtype="m8[s]")
    group_idx = np.array([0, 0, 1])
    kwargs = {"q": 0.5, "method": "linear"}

    # the kernel carries the policy: every row reaches it
    seen = []
    carries = build_temporal_op("nanquantile", op, call, tick_op=op, tick_handles_nat=True)
    carries(group_idx, a, 2, None, **kwargs)
    assert len(seen[0]) == 3
    np.testing.assert_array_equal(seen[0], a.view(np.int64))

    # the engine carries the policy: the NaT rows are dropped on the way
    seen = []
    drops = build_temporal_op("nanquantile", op, call, tick_op=op)
    drops(group_idx, a, 2, None, **kwargs)
    np.testing.assert_array_equal(seen[0], np.array([1, 3], dtype=np.int64))


def test_operations_on_ticks_are_operations_of_backend():
    """The operations on the int64 ticks of the temporal values are built by
    the same factory as every other operation of the numba backend - they
    take the same arguments and run their compiled kernel the same way."""
    tick_ops = npg_numba.ops._tick_ops
    for name in ("mean", "median", "quantile", "percentile"):
        assert name in tick_ops, name
    # the tick mean of the backend is the compiled kernel of the ticks, and
    # it answers the ticks of a group
    group_idx = np.array([0, 0, 1])
    ticks = np.array([10, 20, 30], dtype=np.int64)
    res = tick_ops["mean"](group_idx, ticks, 2)
    assert res.dtype == np.int64
    np.testing.assert_array_equal(res, np.array([15, 30], dtype=np.int64))


def test_temporal_operation_built_from_function(impl):
    """The policy of a function is resolved when its operation is built, so
    that the call itself holds no questions about the function it answers."""

    def op(group_idx, a, size, fill_value=None, dtype=None, **kwargs):
        return np.zeros(size, dtype=np.int64)

    def call(op, group_idx, values, size, fill_value, dtype, order, axis, **kwargs):
        return op(group_idx, values, size, fill_value=fill_value, dtype=dtype, **kwargs)

    temporal_op = build_temporal_op("mean", op, call)
    assert temporal_op.__name__ == "temporal_mean"
    # the mean of the ticks needs an operation on the ticks - without one the
    # operation says so instead of answering something else
    with pytest.raises(NotImplementedError):
        temporal_op(np.array([0]), np.array([1], dtype="m8[s]"), 1)


def test_numpy_backend_answers_temporal_input_on_ticks(impl):
    """The numpy backend has no kernel that takes the temporal values as
    they are - every function of it is answered on the int64 ticks."""
    if impl_name(impl) != "numpy":
        pytest.skip("the numpy backend is the one without a native kernel")
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-01-03"], dtype="M8[s]")
    res = npg_numpy.aggregate(group_idx, a, func="mean", size=1)
    assert res.dtype == a.dtype
    assert res[0] == np.datetime64("2011-01-02", "s")
