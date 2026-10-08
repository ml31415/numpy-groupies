"""Tests for the out= parameter of the numba aggregate implementation."""

import numpy as np
import pytest

from .. import uaggregate
from ..aggregate_numba import aggregate


def test_aggregate_out():
    rng = np.random.default_rng(48)
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    ref = aggregate(group_idx, a, func="sum")
    out = np.empty(6)
    ret = aggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # wrong shape or dtype is rejected
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="sum", out=np.empty(5))
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="sum", out=np.empty(6, dtype=np.float32))
    # custom callables do not support out
    with pytest.raises(NotImplementedError, match="out="):
        aggregate(group_idx, a, func=lambda x: np.sum(x), out=out)


def test_uaggregate_out():
    rng = np.random.default_rng(49)
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    ref = uaggregate(group_idx, a, func="sum")
    out = np.empty(300)
    ret = uaggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # shape of ret[group_idx] is enforced
    with pytest.raises(ValueError, match="expected shape"):
        uaggregate(group_idx, a, func="sum", out=np.empty(299))
    with pytest.raises(TypeError, match="result dtype"):
        uaggregate(group_idx, a, func="sum", out=np.empty(300, dtype=np.int64))
    # multidimensional group_idx falls back to fancy indexing assignment
    group_idx_2d = rng.integers(0, 3, (2, 40))
    a2 = rng.random(40)
    ref2 = uaggregate(group_idx_2d, a2, func="sum")
    out2 = np.empty(ref2.shape)
    ret2 = uaggregate(group_idx_2d, a2, func="sum", out=out2)
    assert ret2 is out2
    np.testing.assert_array_equal(out2, ref2)


def test_aggregate_out_complex():
    rng = np.random.default_rng(50)
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300) + 1j * rng.random(300)
    ref = aggregate(group_idx, a, func="sum")
    out = np.empty(6, dtype=np.complex128)
    ret = aggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # a real out= would throw the imaginary part away, so it is refused
    with pytest.raises(TypeError, match="dtype complex128"):
        aggregate(group_idx, a, func="sum", out=np.empty(6))
    # var measures a real squared magnitude, so a real out= is the right shape
    ref_var = aggregate(group_idx, a, func="var")
    out_var = np.empty(6)
    aggregate(group_idx, a, func="var", out=out_var)
    np.testing.assert_allclose(out_var, ref_var)


def test_uaggregate_out_complex():
    rng = np.random.default_rng(51)
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300) + 1j * rng.random(300)
    ref = uaggregate(group_idx, a, func="sum")
    out = np.empty(300, dtype=np.complex128)
    ret = uaggregate(group_idx, a, out=out, func="sum")
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    with pytest.raises(TypeError, match="does not match the result dtype"):
        uaggregate(group_idx, a, out=np.empty(300), func="sum")
