"""Test suite for numpy_groupies.

The modules are sorted by what a test checks, not by backend: anything a
backend can be expected to pass takes the ``impl`` fixture and runs against
each of them.  So are :mod:`.test_api` (the calling contract),
:mod:`.test_dtypes` (the dtype and precision policy), :mod:`.test_nan` (what
nans do to their group), :mod:`.test_funcs` (the value of each function),
:mod:`.test_axis` (axis aggregation and multi-dimensional labels),
:mod:`.test_complex` (complex input), :mod:`.test_utils` (the exported helper
functions) and :mod:`.test_compare` (the backends against each other).
:mod:`.test_purepy`, :mod:`.test_numba` and :mod:`.test_numba_kernels` cover
what only one implementation has.

Shared helpers live in :mod:`.helpers`, the group loop that serves as the
reference of them all in :mod:`.reference`, fixtures and hooks in the
``conftest`` modules.
"""
