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

_not_implemented_by_impl = {
    'numpy': [],
    'numba': ('array', 'list', '<lambda>', 'func_preserve_order', 'func_arbitrary'),
    'weave':  ('argmin', 'argmax', 'array', 'list', '<lambda>', 'func_preserve_order', 'func_arbitrary')}

def _wrap_notimplemented_xfail(impl, name=None):
    def _try_xfail(*args, **kwargs):
        """ Some implementations lack some functionality. That's ok, let's xfail that instead of raising errors. """
        try:
            return impl(*args, **kwargs)
        except NotImplementedError as e:
            func = kwargs.pop('func', None)
            if callable(func):
                func = func.__name__
            only = _not_implemented_by_impl.get(func, None)
            if only is None or func in only:
                pytest.xfail("Functionality not implemented")
            else:
                raise e
    if name:
        _try_xfail.__name__ = name
    else:
        _try_xfail.__name__ = impl.__name__
    return _try_xfail