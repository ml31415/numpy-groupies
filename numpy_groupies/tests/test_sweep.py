"""The sweep of the suite: every function of the package against the group loop.

One test per (backend, values, function): the values are the six sets the
block fixture carries - the plain floats, the ones with nans in them, the
complex ones, and the integer ones in the three dtypes the package meets:
the wide signed one, the narrow signed one, whose dtype the reductions that
answer a value of the group preserve, and the narrow unsigned one, whose
sums and products the dtype ladder widens.  A function that the sweep does
not know about is a hole in the sweep and not a hole in some test of its
own, whatever kind of values it is asked for.

The data comes in two sizes: the small one is below the floor of the parallel
path of the numba backend, the big one above it.  The sweep therefore
measures the sequential path and the parallel one with the very same test -
the backend that has threads takes the parallel path of the big data, and
the ones that have none answer the same at both sizes.

The yardstick is the group loop of :mod:`.reference`, which shares no code
with the backends: no bincount, no ufunc trick, no compiled kernel, one
computation per group.
"""

import pytest

from numpy_groupies.npg_numba import mt
from numpy_groupies.tests.helpers import (
    SWEEP_SIZES,
    assert_matches_the_group_loop,
    func_list,
)
from numpy_groupies.tests.test_complex import complex_funcs

# the values of the block and the functions that answer for them - the
# complex values are only asked of the functions whose result is defined for
# a complex value (see the lists of the complex test); the integer values get every
# function - a nan variant over integers is the function itself, and the
# ladder of the output dtypes is the business of the comparison, not of the
# selection
_SWEEPS = (
    ("a", func_list),
    ("a_nans", func_list),
    ("a_complex", complex_funcs),
    ("a_ints", func_list),
    ("a_ints8", func_list),
    ("a_uints", func_list),
)

_CASES = [(values, func) for values, funcs in _SWEEPS for func in funcs]


@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:overflow encountered")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.parametrize(("values", "func"), _CASES, ids=[f"{values}-{func}" for values, func in _CASES])
def test_matches_group_loop(impl, func, scaled_block, n_values, values):
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


def test_big_data_reaches_parallel_path(block):
    """The big data of the sweep is the data of the parallel path.

    The cost model is the one the drivers ask before they take the parallel
    path - and the big data is above the floor of both of them.  The cheapest
    and the dearest operation of each path is asked, at the fewest threads
    that can take the path at all, so the answer covers the whole span of the
    operations the sweep runs: if the big data dropped below the floor of one
    of them, the sweep would measure the sequential path twice and call it
    the parallel one.
    """

    n_values = SWEEP_SIZES["large"]
    # the groups of the block - the scaled block of the sweep has the same
    # ones, the two of them empty
    groups = block.size
    # the chunked path: the operations that reduce the values in chunks, the
    # state of the cheapest and of the dearest one of them - two threads are
    # the fewest the path takes
    for state_size in (1, 5):
        assert mt.should_use_mt(groups, 2, n_values, state_size)
    # the path over the groups: the order statistics, the cheapest and the
    # dearest of them - four threads are the fewest theirs takes
    for ns_per_value in (13.0, 68.0):
        assert mt.should_use_group_mt(groups, 4, n_values, ns_per_value)


# test_the_parallel_path_is_the_one_that_runs lives in test_mt_kernels.py
# (``test_the_number_of_the_threads_of_a_call_reaches_the_kernel``): the same
# tripwire over the same data, over every number of the threads above one.
