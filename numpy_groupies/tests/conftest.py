"""fixtures shared by the test modules"""

import pytest

from . import _impl_name, _implementations, _wrap_notimplemented_skip


@pytest.fixture(params=_implementations, ids=_impl_name)
def aggregate_all(request):
    impl = request.param
    if impl is None:
        pytest.skip("Implementation not available")
    name = _impl_name(impl)
    return _wrap_notimplemented_skip(impl.aggregate, "aggregate_" + name)
