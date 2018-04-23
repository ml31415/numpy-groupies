def dummy_no_impl(*args, **kwargs):
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
    from .aggregate_numpy import aggregate
    aggregate_np = aggregate
    from .aggregate_numpy_ufunc import aggregate as aggregate_ufunc
    from .utils_numpy import (multi_arange,
                              label_contiguous_1d,
                              relabel_groups_masked,
                              relabel_groups_unique,
                              unpack)


try:
    try:
        import weave
    except ImportError:
        from scipy import weave
except ImportError:
    aggregate_wv = None
else:
    from .aggregate_weave import aggregate as aggregate_wv, step_indices, step_count
    aggregate = aggregate_wv


try:
    import numba
except ImportError:
    aggregate_nb = None
else:
    from .aggregate_numba import aggregate as aggregate_nb, step_indices, step_count
    aggregate = aggregate_nb


def uaggregate(group_idx, a, **kwargs):
    return unpack(group_idx, aggregate(group_idx, a, **kwargs))

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
