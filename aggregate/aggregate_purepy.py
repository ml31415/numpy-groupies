import math
import itertools

from .utils import get_func_str, aliasing


# min - builtin
# max - builtin
# sum - builtin
# all - builtin
# any - builtin

_last = lambda x: x[-1]

_first = lambda x: x[0]

_mean = lambda x: sum(x) / len(x)

_array = lambda x: x

_sort = lambda x: sorted(x)

_rsort = lambda x: sorted(x, reverse=True)

def _anynan(x):
    return any(math.isnan(xx) for xx in x)

def _allnan(x):
    return all(math.isnan(xx) for xx in x)

def _var(x):
    mean = _mean(x)
    return sum((xx - mean) ** 2 for xx in x) / len(x)

def _std(x):
    return math.sqrt(_var(x))

def _prod(x):
    r = x[0]
    for xx in x[1:]:
        r *= xx
    return r

_func_dict = dict(min=min, max=max, sum=sum, prod=_prod, last=_last, first=_first,
                all=all, any=any, mean=_mean, std=_std, var=_var,
                anynan=_anynan, allnan=_allnan, sort=_sort, rsort=_rsort,
                array=_array)


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order=None, dtype=None):
    """ Accumulation function similar to Matlab's `accumarray` function.
    
        See readme file at https://github.com/ml31415/accumarray for 
        full description.
    """
    # Check for 2d group_idx
    for i in group_idx:
        if isinstance(i, int):
            if i < 0:
                raise ValueError("group_idx contains negative value")
        elif isinstance(i, (list, tuple)):
            raise NotImplementedError("pure python implementation doesn't accept 2d idx input.")
        else:
            raise ValueError("Invalid value found in group_idx: %s" % i)

    if len(group_idx) != len(a):
            raise ValueError("group_idx and a must be of the same length")

    func_str = get_func_str(aliasing, func)
    if func_str.startswith('nan'):
        func_str = func_str[3:]
        # remove nans
        group_idx, a = zip(*((ix, val) for ix, val in zip(group_idx, a) if not math.isnan(val)))

    func = _func_dict.get(func_str, func)
    if size is None:
        size = 1 + max(group_idx)

    # sort data and evaluate function on groups
    data = sorted(zip(group_idx, a), key=lambda tp: tp[0])
    ret = [fill_value] * size
    for ix, group in itertools.groupby(data, key=lambda tp: tp[0]):
        ret[ix] = func(tuple(val for _, val in group))

    return ret

