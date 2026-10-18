"""Test suite for numpy_groupies.

The modules are sorted by what a test checks, not by backend: anything a
backend can be expected to pass takes the ``impl`` fixture and runs against
each of them.  So are :mod:`.test_api` (the calling contract),
:mod:`.test_dtypes` (the dtype and precision policy), :mod:`.test_nan` (what
nans do to their group), :mod:`.test_funcs` (the value of each function),
:mod:`.test_axis` (axis aggregation and multi-dimensional labels),
:mod:`.test_complex` (complex input) and :mod:`.test_utils` (the exported
helper functions).  :mod:`.test_purepy`, :mod:`.test_numba` and
:mod:`.test_numba_kernels` cover what only one implementation has.

Their yardstick is :mod:`.reference`, ``aggregate`` spelled out as a group loop:
a backend is measured against it, never against itself.  Where one deviates
from it without raising, the predicate that names the deviation lives in
:mod:`.helpers`.  What the comparison against the group loop cannot tell apart
is not dropped from it, but gets a test of its own that asserts the floor of it.
"""
