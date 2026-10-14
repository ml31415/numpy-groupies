"""fixtures shared by the test modules"""

import numpy as np
import pytest

from .helpers import _BackendProxy, _implementations, impl_name


@pytest.fixture(params=_implementations, ids=impl_name)
def impl(request):
    """The current backend module under test, as a ``_BackendProxy``.

    Tests use it like a module: ``impl.aggregate(...)``, ``impl.step_count(...)``
    etc. ``NotImplementedError`` (missing functionality) and missing attributes
    are translated into ``pytest.skip``. ``impl_name(impl)`` is the short backend
    name ("purepy", "pandas", ...).
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
