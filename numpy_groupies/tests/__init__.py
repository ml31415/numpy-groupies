import pytest

from .. import aggregate_purepy, aggregate_numpy_ufunc, aggregate_numpy
try:
    from .. import aggregate_numba
except ImportError:
    aggregate_numba = None
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


_not_implemented_by_impl_name = {
    'numpy':  ('cumprod','cummax', 'cummin'),
    'purepy': ('cumsum', 'cumprod','cummax', 'cummin', 'sumofsquares'),
    'numba':  ('array', 'list', 'sort'),
    'pandas': ('array', 'list', 'sort', 'sumofsquares', 'nansumofsquares'),
    'weave':  ('argmin', 'argmax', 'array', 'list', 'sort', 'cumsum', 'cummax', 'cummin',
               'nanargmin', 'nanargmax', 'sumofsquares', 'nansumofsquares',
               '<lambda>', 'custom_callable'),
    'ufunc':  'NO_CHECK'}


def _wrap_notimplemented_xfail(impl, name=None):
    """Some implementations lack some functionality. That's ok, let's xfail that instead of raising errors."""

    def try_xfail(*args, **kwargs):
        try:
            return impl(*args, **kwargs)
        except NotImplementedError as e:
            impl_name = impl.__module__.split('_')[-1]
            func = kwargs.pop('func', None)
            if callable(func):
                func = func.__name__
            not_implemented_ok = _not_implemented_by_impl_name.get(impl_name, [])
            if not_implemented_ok == 'NO_CHECK' or func in not_implemented_ok:
                pytest.xfail("Functionality not implemented")
            else:
                raise e
    if name:
        try_xfail.__name__ = name
    else:
        try_xfail.__name__ = impl.__name__
    return try_xfail


func_list = ('sum', 'prod', 'min', 'max', 'all', 'any', 'mean', 'std', 'var', 'len',
             'argmin', 'argmax', 'anynan', 'allnan', 'cumsum', 'sumofsquares',
             'nansum', 'nanprod', 'nanmin', 'nanmax', 'nanmean', 'nanstd', 'nanvar','nanlen',
             'nanargmin', 'nanargmax', 'nansumofsquares')
