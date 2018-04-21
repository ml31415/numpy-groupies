import numpy as np


def unpack(group_idx, ret):
    """ Take an aggregate packed array and uncompress it to the size of group_idx. 
        This is equivalent to ret[group_idx].
    """
    return ret[group_idx]


def allnan(x):
    return np.all(np.isnan(x))


def anynan(x):
    return np.any(np.isnan(x))


def nanfirst(x):
    return x[~np.isnan(x)][0]


def nanlast(x):
    return x[~np.isnan(x)][-1]


def multi_arange(n):
    """By example:
    
        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]

    That is it is equivalent to something like this :
    
        hstack((arange(n_i) for n_i in n))
        
    This version seems quite a bit faster, at least for some
    possible inputs, and at any rate it encapsulates a task 
    in a function.
    """
    if n.ndim != 1:
        raise ValueError("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1] + 1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return np.cumsum(ret)[:-1]


def label_contiguous_1d(X):
    """ 
    WARNING: API for this function is not liable to change!!!    
    
    By example:

        X =      [F T T F F T F F F T T T]
        result = [0 1 1 0 0 2 0 0 0 3 3 3]
    
    Or:
        X =      [0 3 3 0 0 5 5 5 1 1 0 2]
        result = [0 1 1 0 0 2 2 2 3 3 0 4]
    
    The ``0`` or ``False`` elements of ``X`` are labeled as ``0`` in the output. If ``X``
    is a boolean array, each contiguous block of ``True`` is given an integer
    label, if ``X`` is not boolean, then each contiguous block of identical values
    is given an integer label. Integer labels are 1, 2, 3,..... (i.e. start a 1
    and increase by 1 for each block with no skipped numbers.)
    
    """

    if X.ndim != 1:
        raise ValueError("this is for 1d masks only.")

    is_start = np.empty(len(X), dtype=bool)
    is_start[0] = X[0]  # True if X[0] is True or non-zero

    if X.dtype.kind == 'b':
        is_start[1:] = ~X[:-1] & X[1:]
        M = X
    else:
        M = X.astype(bool)
        is_start[1:] = X[:-1] != X[1:]
        is_start[~M] = False

    L = np.cumsum(is_start)
    L[~M] = 0
    return L


def relabel_groups_unique(group_idx):
    """
    See also ``relabel_groups_masked``.
    
    keep_group:  [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]
    ret:         [0 3 3 3 0 2 4 2 0 1 1 0 3 4 4]
    
    Description of above: unique groups in input was ``1,2,3,5``, i.e.
    ``4`` was missing, so group 5 was relabled to be ``4``.
    Relabeling maintains order, just "compressing" the higher numbers
    to fill gaps.
    """

    keep_group = np.zeros(np.max(group_idx) + 1, dtype=bool)
    keep_group[0] = True
    keep_group[group_idx] = True
    return relabel_groups_masked(group_idx, keep_group)


def relabel_groups_masked(group_idx, keep_group):
    """
    group_idx: [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]
   
                 0 1 2 3 4 5
    keep_group: [0 1 0 1 1 1]
    
    ret:       [0 2 2 2 0 0 4 0 0 1 1 0 2 4 4]
    
    Description of above in words: remove group 2, and relabel group 3,4, and 5
    to be 2, 3 and 4 respecitvely, in order to fill the gap.  Note that group 4 was never used
    in the input group_idx, but the user supplied mask said to keep group 4, so group
    5 is only moved up by one place to fill the gap created by removing group 2.
    
    That is, the mask describes which groups to remove,
    the remaining groups are relabled to remove the gaps created by the falsy
    elements in ``keep_group``.  Note that ``keep_group[0]`` has no particular meaning because it refers
    to the zero group which cannot be "removed".

    ``keep_group`` should be bool and ``group_idx`` int.
    Values in ``group_idx`` can be any order, and 
    """

    keep_group = keep_group.astype(bool, copy=not keep_group[0])
    if not keep_group[0]:  # ensuring keep_group[0] is True makes life easier
        keep_group[0] = True

    relabel = np.zeros(keep_group.size, dtype=group_idx.dtype)
    relabel[keep_group] = np.arange(np.count_nonzero(keep_group))
    return relabel[group_idx]
