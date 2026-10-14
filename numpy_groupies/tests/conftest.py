"""fixtures shared by the test modules"""

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
