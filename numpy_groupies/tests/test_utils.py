"""The functions the package exports besides the aggregations.

``step_count`` and ``step_indices`` report the runs of ``group_idx``, ``unpack``
inverts a group reduction - every backend provides them.
"""

import numpy as np
import pytest

from numpy_groupies.aggregate_common import build_dispatch, custom_callable, resolve_name
from numpy_groupies.tests.helpers import random_values
from numpy_groupies.utils import aliasing, check_dtype, unpack


def test_check_dtype():
    dtype = check_dtype(None, "mean", np.arange(10, dtype=int), 10)
    assert np.issubdtype(dtype, np.floating)


def test_build_dispatch_resolves_every_alias():
    """Every alias of every known function is a key of the fused table.

    Naming a function is then one lookup, and a function the backend does not
    implement is answered by that table too: the key is present, holding the name
    and no implementation, rather than missing and sending the caller off to a
    second table to find out which of the two it was.
    """
    implementation = {"sum": sum, "min": min}
    aliasing = {"sum": "sum", "add": "sum", "min": "min", "amin": "min", "nansum": "nansum"}

    dispatch = build_dispatch(implementation, aliasing)

    assert set(dispatch) == set(aliasing)
    assert dispatch["add"] == ("sum", sum)
    assert dispatch["amin"] == ("min", min)
    assert dispatch["nansum"] == ("nansum", None)


def test_a_name_is_resolved_without_asking_its_func_for_a_hash():
    """A func that cannot be a key names no function - and is not looked up to find

    that out.  ``resolve_name`` asks the type first, so the lookup that could not
    be made is never the thing that appears to have failed.
    """
    assert resolve_name([np.sum]) is None
    assert resolve_name({"func": "sum"}) is None
    assert resolve_name("amin") == "min"
    assert resolve_name(np.amin) is None  # the names-only table holds no objects
    assert resolve_name(np.amin, aliasing) == "min"


def test_a_func_that_is_neither_a_name_nor_a_callable_is_said_so():
    for func in (42, [1, 2], {"a": 1}, np.array([1.0])):
        with pytest.raises(ValueError, match="neither a valid function string nor a callable object"):
            custom_callable(func)


def test_unpack(rng):
    """Keep this test, in case unpack might get reimplemented again at some point."""
    group_idx = np.arange(10)
    rng.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 3)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_unpack_long(rng):
    group_idx = np.repeat(np.arange(10000), 20)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_step_indices_length(impl, rng):
    """``step_count`` and ``step_indices`` length agree with a numpy reference."""
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
    for _ in range(20):
        rng.shuffle(group_idx)
        step_cnt_ref = np.count_nonzero(np.diff(group_idx))
        assert impl.step_count(group_idx) == step_cnt_ref + 1
        assert len(impl.step_indices(group_idx)) == step_cnt_ref + 2


def test_step_indices_fields(impl):
    """``step_indices`` returns the run edges of ``group_idx``."""
    group_idx = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
    steps = impl.step_indices(group_idx)
    np.testing.assert_array_equal(steps, np.array([0, 3, 5, 7, 8, 9, 11]))
