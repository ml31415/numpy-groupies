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


def multi_cumsum(X, L, invalid=np.nan):
    """    
    WARNING: API for this function is not liable to change!!!
    
    By example:

        X=     [3   5 8  9 1  2    5    8 5  4  9   2]
        L=     [0   1 1  2 2  0    0    1 1  1  0   5]
        result=[NaN 5 13 9 10 NaN  NaN  8 13 17 NaN 2]

    That is we calculate the cumsum for each section of `X`
    where the sections are defined by contiguous blocks of
    labels in `L`. Where `L==0`, the output is set to `invalid`     
    """

    L, X = L.ravel(), X.ravel()  # TODO: be consistent with other functions in this module

    if len(L) != len(X):
        raise ValueError('The two inputs should be vectors of the same length.')

    # Do the full cumulative sum
    X[np.isnan(X)] = 0
    S = np.cumsum(X)

    mask = L.astype(bool)

    # Lookup the cumulative value just before the start of each segment
    is_start = mask.copy()
    is_start[1:] &= (L[:-1] != L[1:])
    start_inds, = is_start.nonzero()
    S_starts = S[start_inds - 1] if start_inds[0] != 0 else  np.insert(S[start_inds[1:] - 1], 0, 0)

    # Subtract off the excess values (i.e. the ones obtained above)
    L_safe = np.cumsum(is_start)  # we do this in case the labels in L were not sorted integers
    S[mask] = S[mask] - S_starts[L_safe[mask] - 1]

    # Put NaNs in the false blocks
    S[L == 0] = invalid

    return S


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


def find_contiguous_boundaries(X):
    """
            0 1 2 3 4 5 6 7 8 9 10 11
        X: [4 0 1 1 1 0 3 3 4 4  4  0]
        starts: [0 2 6 8]
        ends: [0 4 7 10]
        
    """
    change_idx, = (X[:-1] != X[1:]).nonzero()
    M = X.astype(bool, copy=False)
    end_idx = change_idx[M[change_idx]]
    if X[-1]:
        end_idx = np.append(end_idx, len(X) - 1)
    change_idx += 1
    start_idx = change_idx[M[change_idx]]
    if X[0]:
        start_idx = np.insert(start_idx, 0, 0)

    return start_idx, end_idx
