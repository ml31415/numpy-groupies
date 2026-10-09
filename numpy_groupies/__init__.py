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
    multi_arange = label_contiguous_1d = dummy_no_impl
else:
    from .aggregate_numpy import aggregate

    aggregate_np = aggregate
    from .aggregate_numpy_ufunc import aggregate as aggregate_ufunc
    from .utils import (
        default_fill_value,
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
    from .aggregate_numba import step_count, step_indices, unpack_into

    aggregate = aggregate_nb


def uaggregate(group_idx, a, out=None, **kwargs):
    """
    Aggregate the values of ``a`` by the groups in ``group_idx`` and broadcast the result
    back to the size of ``a``, so that every element carries the result of its own group.
    This is equivalent to ``unpack(group_idx, aggregate(group_idx, a, ...))``, i.e.
    ``aggregate(...)[group_idx]``, and takes the same arguments as ``aggregate``.

    By example:

        group_idx = [3, 0, 0, 1, 0, 3, 5, 5, 0, 4]
        a         = [13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7]
        ret       = [13.3, 2.5, 2.5, -8.2, 2.5, 13.3, 46.05, 46.05, 2.5, 53.7]

    with ``func='mean'``. A typical use is demeaning within each group, as in
    ``a - uaggregate(group_idx, a, func='mean')``.

    Pass ``out`` to gather into an array provided by the caller, which has to match the
    shape of ``ret[group_idx]`` and the dtype of ``ret``. Only the numba implementation
    supports this, the others raise ``NotImplementedError``.

    See also ``aggregate``, ``unpack``, ``unpack_into``.
    """
    ret = aggregate(group_idx, a, **kwargs)
    if out is None:
        return unpack(group_idx, ret)
    if aggregate is not aggregate_nb:
        raise NotImplementedError("out= is only supported by the numba implementation")
    return unpack_into(group_idx, ret, out)


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
