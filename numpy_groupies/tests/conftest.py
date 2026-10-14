"""fixtures shared by the test modules"""

import pytest

from . import _BackendProxy, _impl_name, _implementations


@pytest.fixture(params=_implementations, ids=_impl_name)
def impl(request):
    """The current backend module under test, as a ``_BackendProxy``.

    Tests use it like a module: ``impl.aggregate(...)``,
    ``impl.step_count(...)`` etc. The ``aggregate`` attribute is wrapped
    so that ``NotImplementedError`` from a backend that does not support
    the requested ``func`` is converted to a ``pytest.skip``. All other
    attributes are transparently forwarded to the backend module.
    """
    return _BackendProxy(request.param, _impl_name(request.param))
