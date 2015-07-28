
def dummy_no_impl(*args,**kwargs):
    raise NotImplementedError("You may need to install another package (numpy, "
                              "weave, or numba) to access a working implementation.")
    
from .aggregate_purepy import aggregate as aggregate_py

aggregate = aggregate_py

try:
    import numpy as np
except ImportError:
    aggregate_np = aggregate_ufunc = dummy_no_impl
    multi_arange = multi_cumsum = label_contiguous_1d = dummy_no_impl
else:
    from .aggregate_numpy import aggregate as aggregate_np
    from .aggregate_numpy_ufunc import aggregate as aggregate_ufunc
    aggregate = aggregate_np
    from .misc_tools_numpy import multi_arange, multi_cumsum, label_contiguous_1d

    
# TODO: unless we are benchmarking/testing there is probably no need to import pandas as this stage
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
    aggregate = aggregate_weave 



def unpack(group_idx, ret, mode='normal'):
    """ Take an aggregate packed array and uncompress it to the size of group_idx. 
        This is equivalent to ret[group_idx] for the common case.
    """
    if mode == 'downscaled':
        group_idx = np.unique(group_idx, return_inverse=True)[1]
    return ret[group_idx]


def uaggregate(group_idx, a, **kwargs):
    return unpack(group_idx, aggregate(group_idx, a, **kwargs))
