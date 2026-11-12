"""The numpy level: the dtype policy and the shape check.

The two helpers of the numpy level that the tests of the package reach for
directly: ``dtypes.check_dtype``, which guesses the dtype of an answer, and
``inputs.check_nton_shape``, which refuses the shape of an answer a function
that gives one value per input item cannot fill.
"""

import re

import numpy as np
import pytest

from numpy_groupies.dtypes import check_dtype
from numpy_groupies.inputs import check_nton_shape


def test_check_dtype():
    dtype = check_dtype(None, "mean", np.arange(10, dtype=int), 10)
    assert np.issubdtype(dtype, np.floating)


def test_check_nton_shape_complains_about_missing_entries():
    """A one-out-per-in function that cannot fill a hole says how many entries are missing."""
    ret = np.arange(2)
    with pytest.raises(ValueError, match=r"'sort' gives one value per input item and cannot fill the 1 absent group"):
        check_nton_shape(ret, 3, "sort")


@pytest.mark.parametrize(
    ("size", "expected"),
    [(3, "(3,)"), ((3,), "(3,)"), (np.array([3]), "(3,)")],
    ids=["int", "tuple", "array"],
)
def test_check_nton_shape_names_size_in_all_forms(size, expected):
    """The size is a tuple of its dimensions, no matter how the caller wrote it."""
    ret = np.arange(2)
    with pytest.raises(ValueError, match=re.escape(f"absent group entries of size {expected}")):
        check_nton_shape(ret, size, "sort")


def test_check_nton_shape_stays_quiet_for_agreeing_sizes():
    """No complaint when the output holds as many entries as the size asks for."""
    check_nton_shape(np.arange(3), 3, "sort")
    check_nton_shape(np.arange(6), (2, 3), "sort")
