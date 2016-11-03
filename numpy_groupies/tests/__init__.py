import pytest

from .. import aggregate_purepy, aggregate_numpy_ufunc, aggregate_numpy
try:
    from .. import aggregate_numba
except ImportError:
    aggregate_numby = None
try:
    from .. import aggregate_weave
except ImportError:
    aggregate_weave = None
try:
    from .. import aggregate_pandas
except ImportError:
    aggregate_pandas = None

_implementations = [aggregate_purepy, aggregate_numpy_ufunc, aggregate_numpy,
                    aggregate_numba, aggregate_weave, aggregate_pandas]
_implementations = [i for i in _implementations if i is not None]


def _impl_name(impl):
    if not impl:
        return
    return impl.__name__.rsplit('aggregate_', 1)[1].rsplit('_', 1)[-1]


def _wrap_notimplemented_xfail(impl, name=None):
    def _try_xfail(*args, **kwargs):
        """ Some implementations lack some functionality. That's ok, let's xfail that instead of raising errors. """
        try:
            return impl(*args, **kwargs)
        except NotImplementedError:
            raise pytest.xfail("Functionality not implemented")
    if name:
        _try_xfail.__name__ = name
    else:
        _try_xfail.__name__ = impl.__name__
    return _try_xfail