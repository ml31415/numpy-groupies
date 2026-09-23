from functools import partial

import numpy as np
import pandas as pd

from .aggregate_numpy import _aggregate_base
from .utils import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    allnan,
    anynan,
    check_dtype,
    funcs_no_separate_nan,
)


def _wrapper(group_idx, a, size, fill_value, func="sum", dtype=None, ddof=0, **kwargs):
    if len(group_idx) == 0:
        raise ValueError("group_idx must not be empty")
    # scalar input needs broadcasting before anything group-based can run
    a = a if np.ndim(a) else np.broadcast_to(a, len(group_idx))
    if func is anynan or func is allnan:
        # route through the cython any/all kernels on a precomputed isnan
        # mask - pushing a python callable through groupby.aggregate is
        # ~10x slower
        a = np.isnan(a)
        func = "any" if func is anynan else "all"
    funcname = func.__name__ if callable(func) else func
    # pandas skips NaN in median; the plain median has to poison its group
    # afterwards (GroupBy.median() accepts no skipna argument)
    poison_nan = kwargs.pop("_nan_poison", False)
    if funcname == "nancumsum":
        # pandas skipna-cumsum keeps NaN in place, whereas nancumsum
        # semantics expect NaN treated as 0 (issues #79 and #91)
        a = np.where(np.isnan(a), 0, a)
        func = "cumsum"
        funcname = "cumsum"
    if funcname in ("var", "std"):
        kwargs["ddof"] = ddof
    # kwargs starting with "_" are internal flags (e.g. _nansqueeze injected by
    # _aggregate_base) that pandas does not understand - the rest is forwarded.
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    # grouping a Series over the raw arrays skips per-call DataFrame
    # construction (~2ms) and direct method calls skip the generic
    # aggregate-dispatch (~1ms)
    grouped = pd.Series(a).groupby(group_idx, sort=False)
    if callable(func):
        result = grouped.aggregate(func, **kwargs)
    else:
        result = getattr(grouped, funcname)(**kwargs)

    dtype = check_dtype(dtype, funcname, a, size)
    if funcname.startswith("cum"):
        ret = result.to_numpy()
    else:
        ret = np.full(size, fill_value, dtype=dtype)
        with np.errstate(invalid="ignore"):
            ret[np.asarray(result.index)] = result.to_numpy()
        if poison_nan:
            poisoned = np.bincount(group_idx, weights=np.isnan(a), minlength=size) > 0
            ret[poisoned] = np.nan
    return ret


_supported_funcs = [
    "sum",
    "prod",
    "all",
    "any",
    "min",
    "max",
    "mean",
    "median",
    "var",
    "std",
    "first",
    "last",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
]
_impl_dict = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}
_impl_dict.update(
    ("nan" + fn, partial(_wrapper, func=fn)) for fn in _supported_funcs if fn not in funcs_no_separate_nan
)
# plain cumsum must propagate NaNs within their group (issue #91), which is
# pandas skipna=False - nancumsum is handled inside _wrapper instead
_impl_dict["cumsum"] = partial(_wrapper, func="cumsum", skipna=False)
_impl_dict["nancumsum"] = partial(_wrapper, func="nancumsum")
# plain median propagates NaNs like np.median - pandas skips them by default
# and GroupBy.median() takes no skipna argument
_impl_dict["median"] = partial(_wrapper, func="median", _nan_poison=True)
_impl_dict.update(
    allnan=partial(_wrapper, func=allnan),
    anynan=partial(_wrapper, func=anynan),
    len=partial(_wrapper, func="count"),
    nanlen=partial(_wrapper, func="count"),
    argmax=partial(_wrapper, func="idxmax"),
    argmin=partial(_wrapper, func="idxmin"),
    nanargmax=partial(_wrapper, func="idxmax"),
    nanargmin=partial(_wrapper, func="idxmin"),
    generic=_wrapper,
)


def aggregate(
    group_idx,
    a,
    func="sum",
    size=None,
    fill_value=DEFAULT_FILL_VALUE,
    order="C",
    dtype=None,
    axis=None,
    **kwargs,
):
    return _aggregate_base(
        group_idx,
        a,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        func=func,
        axis=axis,
        _impl_dict=_impl_dict,
        is_pandas=True,
        **kwargs,
    )


aggregate.__doc__ = (
    """
    This is the pandas implementation of aggregate. It makes use of
    `pandas`'s groupby machinery and is mainly used for reference
    and benchmarking.
    """
    + aggregate_common_doc
)
