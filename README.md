accumarray
==========

Replacement for matlabs accumarray in numpy. The package actually contains 
three independent implementations of accum. One _really_ slow pure python
implementation, `accum_py`, one pure numpy implmentation, `accum_np`, and 
for most common functions optimized implementations written in C, `accum`.
If accum is called with an aggregation function, which doesn't have an
optimized implementation, it falls back to the numpy implementation.

The optimized aggregation functions are: 
sum, min, max, mean, std, prod, all, any, allnan, anynan,
nansum, nanmin, nanmax, nanmean, nanstd, nanprod


Usage
=====

    from accumarray import accum
    a = np.arange(10)
    # accmap must be of type integer and have the same length as a
    accmap = np.array([0,0,1,1,0,0,2,2,4,4])
    
    accum(accmap, a)
    >>> array([10,  5, 13,  0, 17])
    
Output dtype is generally taken from the input arguments and the
aggregation function, but it can be overwritten:

    accum(accmap, a, dtype=float)
    >>> array([ 10.,   5.,  13.,   0.,  17.])

    accum(accmap, a, func='std')
    >>> array([ 2.0616,  0.5   ,  0.5   ,  0.    ,  0.5   ])

    accum(accmap, a, func='std', fillvalue=np.nan)
    array([ 2.0616,  0.5   ,  0.5   ,     nan,  0.5   ])

The next call actually falls back to `accum_np`, but acts as expected:

    accum(accmap, a, func=list)
    >>> array([[0, 1, 4, 5], [2, 3], [6, 7], 0, [8, 9]], dtype=object)
    
    
    