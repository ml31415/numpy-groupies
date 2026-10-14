"""fixtures and collection hooks shared by the test modules"""

import inspect

import numpy as np
import pytest

from .helpers import _BackendProxy, _implementations, impl_name, is_supported


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "deselect_if(func): deselect parametrizations for which func(**item.params) is true",
    )


def pytest_collection_modifyitems(config, items):
    """Deselect parametrizations that cannot run:

    - combos where the backend dispatch does not support the parametrized
      ``func`` - applies automatically to every test parametrized over both
      ``impl`` and ``func``, no marker needed
    - combos flagged with a ``deselect_if`` marker whose predicate returns
      true for the item's params; predicates only declare the params they
      actually need, the rest is filtered by signature
    """
    removed, kept = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}
        if "impl" in params and "func" in params and not is_supported(params["impl"], params["func"]):
            removed.append(item)
            continue
        if any(_marker_deselects(marker, params) for marker in item.iter_markers("deselect_if")):
            removed.append(item)
            continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


def _marker_deselects(marker, params):
    func = marker.kwargs["func"]
    accepted = inspect.signature(func).parameters
    kwargs = {name: params[name] for name in accepted if name in params}
    return func(**kwargs)
