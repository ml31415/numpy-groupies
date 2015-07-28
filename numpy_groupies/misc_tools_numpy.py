import numpy as np
    
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
        raise Exception("n is supposed to be 1d array.")
        
    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1]+1,dtype=int)
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
    
    L, X = L.ravel(), X.ravel() # TODO: be consistent with other functions in this module
    
    if len(L) != len(X):
        raise Exception('The two inputs should be vectors of the same length.')
    
    # Do the full cumulative sum
    X[np.isnan(X)] = 0
    S = np.cumsum(X)
    
    mask = L.astype(bool)
    
    # Lookup the cumulative value just before the start of each segment
    isStart = mask.copy()
    isStart[1:] &= (L[:-1] != L[1:])
    startInds, = isStart.nonzero()
    S_starts = S[startInds-1] if startInds[0] != 0 else  np.insert(S[startInds[1:]-1],0,0)
    
    # Subtract off the excess values (i.e. the ones obtained above)
    L_safe = np.cumsum(isStart) # we do this in case the labels in L were not sorted integers
    S[mask] = S[mask] - S_starts[L_safe[mask]-1]  
    
    # Put NaNs in the false blocks
    S[L==0] = invalid
    
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
        raise Exception("this is for 1d masks only.")

    is_start = np.empty(len(X),dtype=bool)
    is_start[0] = X[0] # True if X[0] is True or non-zero
        
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