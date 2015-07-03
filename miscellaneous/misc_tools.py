import numpy as np

def multi_repeat(n):
    """By example:
    
        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [2, 2, 2, 5, 5, 7, 7, 8]
        
    That is the input specifies how many times to repeat the given index.

    It is equivalent to something like this :
    
        hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))
        
    But this version seems to be faster, and probably scales better, at
    any rate it encapsulates a task in a function.
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")
        
    n_mask = n.astype(bool)
    n_inds = np.nonzero(n_mask)[0]
    n_inds[1:] = n_inds[1:]-n_inds[:-1] # take diff and leave 0th value in place
    n_cumsum = np.empty(len(n)+1,dtype=int)
    n_cumsum[0] = 0 
    n_cumsum[1:] = np.cumsum(n)
    ret = np.zeros(n_cumsum[-1],dtype=int)
    ret[n_cumsum[n_mask]] = n_inds # note that n_mask is 1 element shorter than n_cumsum
    return np.cumsum(ret)

    
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


def multi_cumsum(X, L, invalid=nan):
    """    
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
    S_starts = S[startInds-1] if startInds[0] != 0 else  insert(S[startInds[1:]-1],0,0)
    
    # Subtract off the excess values (i.e. the ones obtained above)
    L_safe = np.cumsum(isStart) # we do this in case the labels in L were not sorted integers
    S[mask] = S[mask] - S_starts[L_safe[mask]-1]  
    
    # Put NaNs in the false blocks
    S[L==0] = invalid
    
    return S

    
def label_mask_1D(M):
    """ By example:
    M =      [F T T F F T F F F T T T]
    result = [0 1 1 0 0 2 0 0 0 3 3 3]
    
    M is boolean array, result is integer labels of contiguous True sections."""
    
    if M.ndim != 1:
        raise Exception("this is for 1d masks only.")
        
    is_start = np.empty(len(M),dtype=bool)
    is_start[0] = M[0]
    is_start[1:] = ~M[:-1] & M[1:]
    L = np.cumsum(is_start)
    L[~M] = 0
    return L