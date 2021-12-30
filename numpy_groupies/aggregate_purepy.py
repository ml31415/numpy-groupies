from __future__ import division

import itertools
import math
import operator

from .utils import aggregate_common_doc, aliasing, funcs_no_separate_nan, get_func, isstr

# min - builtin
# max - builtin
# sum - builtin
# all - builtin
# any - builtin


def _last(x):
    return x[-1]


def _first(x):
    return x[0]


def _array(x):
    return x


def _mean(x):
    return sum(x) / len(x)


def _var(x, ddof=0):
    mean = _mean(x)
    return sum((xx - mean) ** 2 for xx in x) / (len(x) - ddof)


def _std(x, ddof=0):
    return math.sqrt(_var(x, ddof=ddof))


def _prod(x):
    r = x[0]
    for xx in x[1:]:
        r *= xx
    return r


def _anynan(x):
    return any(math.isnan(xx) for xx in x)


def _allnan(x):
    return all(math.isnan(xx) for xx in x)


def _argmax(x_and_idx):
    return max(x_and_idx, key=operator.itemgetter(1))[0]


_argmax.x_and_idx = True  # tell aggregate what to use as first arg


def _argmin(x_and_idx):
    return min(x_and_idx, key=operator.itemgetter(1))[0]


_argmin.x_and_idx = True  # tell aggregate what to use as first arg


def _sort(group_idx, a, reverse=False):
    def _argsort(unordered):
        return sorted(range(len(unordered)), key=lambda k: unordered[k])

    sortidx = _argsort(list((gi, aj) for gi, aj in zip(group_idx, -a if reverse else a)))
    revidx = _argsort(_argsort(group_idx))
    a_srt = [a[si] for si in sortidx]
    return [a_srt[ri] for ri in revidx]


_impl_dict = dict(min=min, max=max, sum=sum, prod=_prod, last=_last,
                  first=_first, all=all, any=any, mean=_mean, std=_std,
                  var=_var, anynan=_anynan, allnan=_allnan, sort=_sort,
                  array=_array, argmax=_argmax, argmin=_argmin, len=len)
_impl_dict.update(('nan' + k, v) for k, v in list(_impl_dict.items())
                  if k not in funcs_no_separate_nan)


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order=None,
              dtype=None, axis=None, **kwargs):
    if axis is not None:
        raise NotImplementedError("axis arg not supported in purepy implementation.")

    # Check for 2d group_idx
    if size is None:
        try:
            size = 1 + int(max(group_idx))
        except (TypeError, ValueError):
            raise NotImplementedError("pure python implementation doesn't"
                                      " accept ndim idx input.")

    for i in group_idx:
        try:
            i = int(i)
        except (TypeError, ValueError):
            if isinstance(i, (list, tuple)):
                raise NotImplementedError("pure python implementation doesn't"
                                          " accept ndim idx input.")
            else:
                try:
                    len(i)
                except TypeError:
                    raise ValueError("invalid value found in group_idx: %s" % i)
                else:
                    raise NotImplementedError("pure python implementation doesn't "
                                              "accept ndim indexed input.")
        else:
            if i < 0:
                raise ValueError("group_idx contains negative value")

    func = get_func(func, aliasing, _impl_dict)
    if isinstance(a, (int, float)):
        if func not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', "
                             "'prod' and 'len'")
        a = [a] * len(group_idx)
    elif len(group_idx) != len(a):
        raise ValueError("group_idx and a must be of the same length")

    if isstr(func):
        if func.startswith('nan'):
            func = func[3:]
            # remove nans
            group_idx, a = zip(*((ix, val) for ix, val in zip(group_idx, a)
                                 if not math.isnan(val)))

        func = _impl_dict[func]
    if func is _sort:
        return _sort(group_idx, a, reverse=kwargs.get('reverse', False))

    # sort data and evaluate function on groups
    ret = [fill_value] * size
    if not getattr(func, 'x_and_idx', False):
        data = sorted(zip(group_idx, a), key=operator.itemgetter(0))
        for ix, group in itertools.groupby(data, key=operator.itemgetter(0)):
            ret[ix] = func(list(val for _, val in group), **kwargs)
    else:
        data = sorted(zip(range(len(a)), group_idx, a), key=operator.itemgetter(1))
        for ix, group in itertools.groupby(data, key=operator.itemgetter(1)):
            ret[ix] = func(list((val_idx, val) for val_idx, _, val in group), **kwargs)

    return ret


aggregate.__doc__ = """
    This is the pure python implementation of aggregate. It is terribly slow.
    Using the numpy version is highly recommended.
    """ + aggregate_common_doc
