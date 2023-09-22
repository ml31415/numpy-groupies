from .aggregate_purepy import aggregate as aggregate_py


def dummy_no_impl(*args, **kwargs):
    raise NotImplementedError(
        "You may need to install another package (numpy or numba) to access a working implementation."
    )


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
    from .utils import (
        label_contiguous_1d,
        multi_arange,
        relabel_groups_masked,
        relabel_groups_unique,
        unpack,
    )


try:
    import numba
except ImportError:
    aggregate_nb = None
else:
    from .aggregate_numba import aggregate as aggregate_nb
    from .aggregate_numba import step_count, step_indices

    aggregate = aggregate_nb


def uaggregate(group_idx, a, **kwargs):
    return unpack(group_idx, aggregate(group_idx, a, **kwargs))


try:
    # Version is added only when packaged
    from ._version import __version__
except ImportError:
    try:
        from setuptools_scm import get_version
    except ImportError:
        __version__ = "0.0.0"
    else:
        __version__ = get_version(root="..", relative_to=__file__)
        del get_version
