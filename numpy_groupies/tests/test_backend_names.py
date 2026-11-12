"""Tests for the names of the modules of the implementations.

The modules of the implementations are called ``npg_numpy``, ``npg_numba``
and ``npg_purepy``; the names they had before are still there and reach the
same implementations.
"""

import numpy_groupies


def test_modules_of_implementations_carry_their_new_names():
    """The modules of the implementations are the ones of the new names."""
    assert numpy_groupies.npg_numpy.__name__ == "numpy_groupies.npg_numpy"
    assert numpy_groupies.npg_numba.__name__ == "numpy_groupies.npg_numba"
    assert numpy_groupies.npg_purepy.__name__ == "numpy_groupies.npg_purepy"


def test_old_paths_of_modules_reach_implementations():
    """The old paths of the modules reach the implementations.

    The modules are in sys.modules under their old names, so that the path of
    the import stays.
    """
    from numpy_groupies.aggregate_numpy import mean as mean_of_numpy
    from numpy_groupies.aggregate_purepy import aggregate as aggregate_of_purepy

    assert mean_of_numpy is numpy_groupies.npg_numpy.mean
    assert aggregate_of_purepy is numpy_groupies.npg_purepy.aggregate


def test_old_names_of_modules_reach_implementations():
    """The old names of the modules are the modules of the implementations."""
    from numpy_groupies import (
        aggregate_numba,
        aggregate_numpy,
        aggregate_purepy,
    )

    assert aggregate_numpy.mean is numpy_groupies.npg_numpy.mean
    assert aggregate_purepy.aggregate is numpy_groupies.npg_purepy.aggregate
    assert aggregate_numba.mean is numpy_groupies.npg_numba.mean
