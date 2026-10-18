"""fixtures and collection hooks shared by the test modules"""

import typing

import numpy as np
import pytest

from numpy_groupies.tests.helpers import _BackendProxy, _implementations, impl_name, is_supported


class Block(typing.NamedTuple):
    """Labels and values for the comparisons against the group loop.

    ``group_idx`` labels ``a``; ``a_nans`` and ``a_complex`` are the same
    positions with nans in them, the latter of a nan in one, in the other or in
    both of its parts.  ``size`` names two groups more than the labels use, so
    that an absent group is part of every comparison.
    """

    group_idx: np.ndarray
    size: int
    a: np.ndarray
    a_nans: np.ndarray
    a_complex: np.ndarray


@pytest.fixture(params=_implementations, ids=impl_name)
def impl(request):
    """The current backend module under test, as a ``_BackendProxy``.

    Tests use it like a module: ``impl.aggregate(...)``, ``impl.step_count(...)``
    etc. ``NotImplementedError`` (missing functionality) and missing attributes
    are translated into ``pytest.skip``. ``impl_name(impl)`` is the short backend
    name ("purepy", "numpy", ...).
    """
    return _BackendProxy(request.param)


@pytest.fixture(scope="module")
def rng(request):
    """Module-wide seeded random number generator.

    Pass this to any test or fixture that generates random data, e.g. via
    the ``random_group_idx``/``random_values`` helpers. Data is generated
    once per module and reproducible for a given seed, independent of the
    execution order of modules. The seed is configurable via ``--seed``.
    """
    return np.random.default_rng(request.config.getoption("--seed"))


@pytest.fixture(scope="module")
def block(request):
    """A small block of values, the awkward groups included - the comparisons
    against the group loop of :mod:`.reference` need cases, not statistics.

    Twelve groups of uneven length carry them, and two further labels name no
    value at all:

    - 0: a group of nothing but nans, an empty group for the nan functions
    - 1: a single value, no nan with it
    - 2: a single value, and that one a nan
    - 3: two values, exactly one of them a nan
    - 4 to 11: from five to thirty two values, a third of them nans

    The labels are shuffled, so the values of a group are no neighbours of each
    other.  ``a_complex`` carries the nan in a different part of the value per case
    - a nan in either part makes the whole value incomparable, so which one
    carries it must not change the answer - one group of nothing but such values,
    and one in which all three kinds sit beside plain ones.

    No infinity among them: numpy's own functions disagree with each other there
    (``np.median`` selects with ``np.partition``, which does not order the values
    the way ``np.sort`` does once infinities are mixed in), so no answer to offer
    as the reference of it.

    The arrays are write protected, so a test that writes to one of them fails at
    that place rather than in whatever test reads the block next.  It draws on a
    generator of its own rather than on ``rng``, whose state depends on how much
    randomness the tests of the module asked for before the fixture was made.
    """
    rng = np.random.default_rng(request.config.getoption("--seed"))
    sizes = (3, 1, 1, 2, 5, 8, 13, 20, 32, 24, 16, 9)
    group_idx = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    a = rng.standard_normal(group_idx.size)

    a_nans = a.copy()
    a_nans[::3] = np.nan  # a nan in about every third group
    a_nans[group_idx == 0] = np.nan  # a group of nothing but nans
    a_nans[group_idx == 1] = a[group_idx == 1]  # a single value, no nan with it
    a_nans[group_idx == 2] = np.nan  # a single value, and that one a nan
    two = np.flatnonzero(group_idx == 3)  # two values, exactly one of them a nan
    a_nans[two[:1]] = np.nan
    a_nans[two[1:]] = a[two[1:]]

    # the parts poisoned apart from each other, on every twelfth position, so that
    # a nan in the one, in the other and in both of them appears, and a group holds
    # values of all three kinds at once - and a plain one beside them
    real, imag = a.copy(), a.copy()
    real[0::12] = np.nan  # the nan sits in the real part ...
    imag[1::12] = np.nan  # ... in the imaginary one ...
    real[2::12] = np.nan  # ... and in both of them
    imag[2::12] = np.nan
    plain = group_idx == 1  # the one value of this group stays a plain number
    real[plain] = imag[plain] = a[plain]
    a_complex = np.empty(a.size, dtype=np.complex128)
    a_complex.real, a_complex.imag = real, imag
    a_complex[group_idx == 0] = np.nan + 1j * np.nan  # nothing but nans, alike
    kinds = [complex(np.nan, 1.0), complex(1.0, np.nan), complex(np.nan, np.nan)]
    eight = np.flatnonzero(group_idx == 5)  # the three kinds side by side ...
    a_complex[eight[:6]] = kinds * 2
    a_complex[eight[6:]] = (1 + 1j) * a[eight[6:]]  # ... and plain values among them

    block = Block(
        group_idx=group_idx,
        size=14,  # labels 12 and 13 name no value at all
        a=a,
        a_nans=a_nans,
        a_complex=a_complex,
    )
    for array in (group_idx, a, a_nans, a_complex):
        array.setflags(write=False)
    return block


def pytest_collection_modifyitems(config, items):
    """Deselect the combos that cannot run at all.

    A parametrization of ``impl`` and ``func`` is dropped when the dispatch of
    that backend has no such function - a missing function is not a test waiting
    to fail.  Everything else a backend cannot do signals with
    ``NotImplementedError``, which the ``impl`` proxy turns into a skip at
    runtime, and a backend that deviates without raising is measured against the
    group loop like any other and called out where it differs.
    """
    removed, kept = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}
        if "impl" in params and "func" in params and not is_supported(params["impl"], params["func"]):
            removed.append(item)
            continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
