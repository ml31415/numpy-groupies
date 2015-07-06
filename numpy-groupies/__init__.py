import numpy as np

from .aggregate_purepy import aggregate as aggregate_py
from .aggregate_numpy import aggregate as aggregate_np
from .aggregate_numpy_ufunc import aggregate as aggregate_ufunc
from .misc_tools import multi_arange, multi_cumsum

try:
    import pandas
except ImportError:
    aggregate_pd = None
else:
    from .aggregate_pandas import aggregate as aggregate_pd

try:
    from scipy import weave
except ImportError:
    aggregate_weave = None
else:
    from .aggregate_weave import aggregate as aggregate_weave, step_indices, step_count


# Use the fastest implementation available
aggregate = aggregate_weave or aggregate_np


def unpack(group_idx, ret, mode='normal'):
    """ Take an aggregate packed array and uncompress it to the size of group_idx. 
        This is equivalent to ret[group_idx] for the common case.
    """
    if mode == 'downscaled':
        group_idx = np.unique(group_idx, return_inverse=True)[1]
    return ret[group_idx]


def uaggregate(group_idx, a, **kwargs):
    return unpack(group_idx, aggregate(group_idx, a, **kwargs))
