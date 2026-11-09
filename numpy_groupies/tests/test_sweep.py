"""The sweep of the suite: every function of the package against the group loop.

One test per (backend, values, function): the values are the three sets the
block fixture carries - the plain ones, the ones with nans in them and the
complex ones - so a function that the sweep does not know about is a hole in
the sweep and not a hole in some test of its own.

The data comes in two sizes: the small one is below the floor of the parallel
path of the numba backend, the big one above it.  The sweep therefore
measures the sequential path and the parallel one with the very same test -
the backend that has threads takes the parallel path of the big data, and
the ones that have none answer the same at both sizes.

The yardstick is the group loop of :mod:`.reference`, which shares no code
with the backends: no bincount, no ufunc trick, no compiled kernel, one
computation per group.
"""

import numba as nb
import numpy as np
import pytest

from numpy_groupies import aggregate_numba
from numpy_groupies.tests.helpers import (
    SWEEP_SIZES,
    assert_matches_the_group_loop,
    complex_funcs,
    func_list,
)

# the values of the block and the functions that answer for them - the
# complex values are only asked of the functions whose result is defined for
# a complex value (see the sets of the helpers)
_SWEEPS = (
    ("a", func_list),
    ("a_nans", func_list),
    ("a_complex", complex_funcs),
)

_CASES = [(values, func) for values, funcs in _SWEEPS for func in funcs]


@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:overflow encountered")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.parametrize(("values", "func"), _CASES, ids=[f"{values}-{func}" for values, func in _CASES])
def test_matches_the_group_loop(impl, func, scaled_block, n_values, values):
    """every function over the values of the block, against the group loop.

    The plain functions let the nan reach the result, their nan counterparts
    skip the value, and a group of nothing but nans is an empty group for
    them - the rule the loop spells out one group at a time.

    The answers that are zero - the variance of a group whose values are all
    the same, the product of a group whose values underflow - are compared
    with an absolute tolerance: the two ways of computing them leave a
    rounding residue behind, and the size of that residue is the one of the
    sum of the values of the group.
    """
    assert_matches_the_group_loop(
        impl,
        func,
        scaled_block,
        getattr(scaled_block, values),
        atol=n_values * 1e-15,
    )


def test_the_big_data_reaches_the_parallel_path(block):
    """The big data of the sweep is the data of the parallel path.

    The cost model is the one the drivers ask before they take the parallel
    path - and the big data is above the floor of both of them, with the
    threads of the pool: the operations that reduce the values in chunks and
    the ones that cut the groups into ranges both take their parallel path.
    """
    from numpy_groupies.aggregate_numba import mt

    threads = nb.get_num_threads()
    if threads < 4:
        pytest.skip("the machine has fewer than four threads to compare against")
    n_values = SWEEP_SIZES["large"]
    # the groups of the block - the scaled block of the sweep has the same
    # ones, the two of them empty
    groups = block.size
    # the chunked path: the operations that reduce the values in chunks, the
    # state of the cheapest and of the dearest one of them
    for state_size in (1, 3):
        assert mt.should_use_mt(groups, threads, n_values, state_size)
    # the path over the groups: the order statistics, the cheapest and the
    # dearest of them
    for ns_per_value in (18.0, 68.0):
        assert mt.should_use_group_mt(groups, threads, n_values, ns_per_value)


def test_the_parallel_path_is_the_one_that_runs(scaled_block, n_values):
    """The sums of the parallel path add up in the order of the chunks.

    The last bits of a float sum notice it - so a result that is identical
    to the one of the single thread is a result of the single thread, and
    the big data of the sweep is the data of the parallel path.
    """
    if n_values <= SWEEP_SIZES["small"]:
        pytest.skip("the small data is below the floor of the parallel path")
    group_idx, values = scaled_block.group_idx, scaled_block.a
    before = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        one = aggregate_numba.aggregate(group_idx, values, func="sum", size=scaled_block.size)
        nb.set_num_threads(before)
        many = aggregate_numba.aggregate(group_idx, values, func="sum", size=scaled_block.size)
    finally:
        nb.set_num_threads(before)
    assert not np.array_equal(one, many)
