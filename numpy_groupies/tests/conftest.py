"""fixtures and collection hooks shared by the test modules"""

import typing

import numpy as np
import numpy.typing as npt
import pytest

from numpy_groupies import npg_numba, npg_numpy, npg_purepy
from numpy_groupies.tests.helpers import (
    SWEEP_SIZES,
    _BackendProxy,
    impl_name,
    is_purepy,
    is_supported,
)

# the backends the suite runs against - the module of every implementation
_implementations = [npg_purepy, npg_numpy, npg_numba]


class Block(typing.NamedTuple):
    """Labels and values for the comparisons against the group loop.

    ``group_idx`` labels ``a``; ``a_nans`` and ``a_complex`` are the same
    positions with nans (the complex one in one part, the other, or both).
    ``a_ints``, ``a_ints8`` and ``a_uints`` carry three integer dtypes: the wide
    signed one, the narrow signed one (kept by the reductions that preserve the
    values of the group) and the narrow unsigned one (widened by the dtype
    ladder).  ``size`` names two groups more than the labels use, so an absent
    group is part of every comparison.
    """

    group_idx: npt.NDArray[np.integer]
    size: int
    a: npt.NDArray[np.number]
    a_nans: npt.NDArray[np.floating]
    a_complex: npt.NDArray[np.complexfloating]
    a_ints: npt.NDArray[np.integer]
    a_ints8: npt.NDArray[np.integer]
    a_uints: npt.NDArray[np.integer]


@pytest.fixture(params=_implementations, ids=impl_name)
def impl(request):
    """The current backend module under test, as a ``_BackendProxy``.

    Tests use it like a module: ``impl.aggregate(...)`` etc.
    ``NotImplementedError`` and missing attributes become ``pytest.skip``.
    """
    return _BackendProxy(request.param)


@pytest.fixture
def skip_purepy(impl):
    """The pure python backend knows no datetime64 or timedelta64 values:
    its arithmetic runs on python numbers, not on the ticks of temporal
    values.  The temporal tests leave it out.  Not autouse: it would ask
    for the backend of every test of the suite, and the tests that ask for
    no backend at all would run three times over for nothing.
    """
    if impl_name(impl) == "purepy":
        pytest.skip("the pure python backend knows no datetime64 or timedelta64 values")


@pytest.fixture(scope="module")
def rng(request):
    """Module-wide seeded random number generator.

    Data is generated once per module, reproducible for the ``--seed``
    and independent of the execution order of modules.
    """
    return np.random.default_rng(request.config.getoption("--seed"))


@pytest.fixture(scope="session")
def block(request):
    """A small block of values, the awkward groups included - the comparisons
    against the group loop of :mod:`.reference` need cases, not statistics.

    Twelve groups of uneven length carry them; labels 12 and 13 name no value
    at all.  Groups 0 to 3 are the nan cases: nothing but nans, a single clean
    value, a single nan, two values with exactly one nan; groups 4 to 11 hold
    five to thirty two values, a third of them nans.  The labels are shuffled,
    so the values of a group are no neighbours of each other.  ``a_complex``
    carries the nan in a different part of the value per case - a nan in either
    part makes the whole value incomparable, so which one carries it must not
    change the answer - and one group holds nothing but such values.

    No infinity among them: numpy's own functions disagree there (``np.median``
    selects with ``np.partition``, which does not order the values the way
    ``np.sort`` does once infinities are mixed in), so no reference to offer.

    The arrays are write protected, so a test that writes to one fails at that
    place.  It draws on a generator of its own rather than on ``rng``, whose
    state depends on how much randomness the tests asked for before.  The block
    is the same in every module - the seed is the same and the generator is
    drawn anew - so it is made once for the session.
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

    # the integer values: small ones, so the products of the biggest group fit
    # the wide integer whatever the dtype they are carried in - the sums and
    # the products of the narrow dtypes are exact all the same, and no wrap of
    # a narrow dtype can make the backends disagree about the values
    a_ints = rng.integers(-2, 3, group_idx.size)  # the wide signed dtype
    a_ints8 = rng.integers(-2, 3, group_idx.size).astype(np.int8)  # the narrow signed one
    a_uints = rng.integers(0, 5, group_idx.size).astype(np.uint8)  # the narrow unsigned one

    block = Block(
        group_idx=group_idx,
        size=14,  # labels 12 and 13 name no value at all
        a=a,
        a_nans=a_nans,
        a_complex=a_complex,
        a_ints=a_ints,
        a_ints8=a_ints8,
        a_uints=a_uints,
    )
    for array in (group_idx, a, a_nans, a_complex, a_ints, a_ints8, a_uints):
        array.setflags(write=False)
    return block


@pytest.fixture(params=SWEEP_SIZES.values(), ids=list(SWEEP_SIZES), scope="module")
def n_values(request):
    """The number of values the data of the generic sweep holds.

    The two sizes of :data:`.helpers.SWEEP_SIZES`: the small one below the
    floor of the parallel path of the numba backend, the big one above it -
    so the sweep measures the sequential path and the parallel one with the
    very same test.
    """
    return request.param


@pytest.fixture(scope="module")
def scaled_block(block, n_values):
    """The block of the suite, at the size the sweep asks for.

    The values are repeated until the data is as big as the scale asks for:
    the groups keep their uneven sizes, and the nans and the complex values
    with them.  The scope is the module, not the test: the big scale is a few
    hundred megabytes of arrays - built once per module, not once per test.
    """
    repeats = -(-n_values // block.group_idx.size)
    group_idx = np.tile(block.group_idx, repeats)[:n_values]
    a, a_nans, a_complex, a_ints, a_ints8, a_uints = (
        np.tile(values, repeats)[:n_values]
        for values in (block.a, block.a_nans, block.a_complex, block.a_ints, block.a_ints8, block.a_uints)
    )
    for array in (group_idx, a, a_nans, a_complex, a_ints, a_ints8, a_uints):
        array.setflags(write=False)
    return Block(
        group_idx=group_idx,
        size=block.size,
        a=a,
        a_nans=a_nans,
        a_complex=a_complex,
        a_ints=a_ints,
        a_ints8=a_ints8,
        a_uints=a_uints,
    )


def pytest_collection_modifyitems(config, items):
    """Deselect the combos that cannot run at all.

    A parametrization of ``impl`` and ``func`` is dropped when the dispatch of
    that backend has no such function - a missing function is not a test waiting
    to fail.  Everything else a backend cannot do signals with
    ``NotImplementedError``, which the ``impl`` proxy turns into a skip at
    runtime.  The big sweep size is deselected for pure python: it walks the
    data in a python loop and has no threads anyway, so the big size would
    measure the very same path twice.  Tests that ask for the same function
    are put into one group, so the workers of a parallel run share the compiled
    kernel of that function (see :func:`_group_of`).
    """
    removed, kept = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}
        if "impl" in params and "func" in params and not is_supported(params["impl"], params["func"]):
            removed.append(item)
            continue
        if (
            "impl" in params
            and "n_values" in params
            and is_purepy(params["impl"])
            and params["n_values"] > SWEEP_SIZES["small"]
        ):
            removed.append(item)
            continue
        group = _group_of(item, params)
        if group is not None:
            item.add_marker(pytest.mark.xdist_group(name=group))
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


# the parameters that name the function a test asks for
_FUNCTION_PARAMS = ("func", "kernel_name", "name")


def _group_of(item, params):
    """The group a test belongs to: the function it asks for.

    The workers of a parallel run share one disk cache of the compiled
    kernels and wait for each other, so the kernel of a function is compiled
    once in the whole suite and not once per worker.  A test that asks for
    no function at all is grouped by its module.
    """
    module = item.module.__name__.rsplit(".", 1)[-1]
    for name in _FUNCTION_PARAMS:
        value = params.get(name)
        if isinstance(value, str):
            return f"fn-{value}"
    return module
