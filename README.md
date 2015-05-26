accumarray
==========

Replacement for matlabs accumarray in numpy. The package actually contains 
three independent implementations of accum. One _really_ slow pure python
implementation, `accum_py`, one pure numpy implmentation, `accum_np`, and 
for most common functions optimized implementations written in C, `accum`.
If `accum` is called with an aggregation function, which doesn't have an
optimized implementation, it falls back to the numpy implementation.

The optimized aggregation functions are:

sum, min, max, mean, std, prod, all, any, allnan, anynan,
nansum, nanmin, nanmax, nanmean, nanstd, nanprod


Requirements
------------

* python 2.7.x
* numpy
* scipy
* gcc for using scipy.weave
* pytest for testing


Usage
-----

```python
from accumarray import accum, accum_np, accum_py, unpack
a = np.arange(10)
# accmap must be of type integer and have the same length as a
accmap = np.array([0,0,1,1,0,0,2,2,4,4])
    
accum(accmap, a)
>>> array([10,  5, 13,  0, 17])
```

The C functions are compiled on the fly by `scipy.weave` depending
on the selected aggregation function and the data types of the inputs.
The compiled functions are cached and reused, whenever possible. So 
for the first runs, expect some delays for the compilation.

The output dtype is generally taken from the input arguments and the
aggregation function, but it can be overwritten:

```python
accum(accmap, a, dtype=float)
>>> array([ 10.,   5.,  13.,   0.,  17.])

accum(accmap, a, func='std')
>>> array([ 2.0616,  0.5   ,  0.5   ,  0.    ,  0.5   ])

accum(accmap, a, func='std', fillvalue=np.nan)
>>> array([ 2.0616,  0.5   ,  0.5   ,     nan,  0.5   ])
```

The next call actually falls back to `accum_np`, but acts as expected:

```python
accum(accmap, a, func=list)
>>> array([[0, 1, 4, 5], [2, 3], [6, 7], 0, [8, 9]], dtype=object)
```    

There is an alternate mode of operation, contiguous, which creates
a new output entry for every value change in accmap. The parameter
fillvalue has no effect here, as all output fields are filled. For
inflating these kind of outputs to the full array size, the function
`unpack` is provided.

```python
accum(accmap, a, mode='contiguous')
>>> array([ 1,  5,  9, 13, 17])

unpack(accmap, accum(accmap, a, mode='contiguous'), mode='contiguous')
>>> array([ 1,  1,  5,  5,  9,  9, 13, 13, 17, 17])
```

Speed comparison
----------------

```python
accmap = np.repeat(np.arange(10000), 10)
a = np.arange(len(accmap))

timeit accum_py(accmap, a)
>>> 1 loops, best of 3: 1.35 s per loop

timeit accum_np(accmap, a)
>>> 1 loops, best of 3: 196 ms per loop

timeit accum(accmap, a)
>>> 1000 loops, best of 3: 1.09 ms per loop

timeit accum(accmap, a, func='mean')
>>> 1000 loops, best of 3: 1.7 ms per loop
```

*Octave*
```matlab
accmap = repmat(1:10000, 10, 1)(:);
a = 1:numel(accmap);
tic; accumarray(accmap, a); toc
>>> Elapsed time is 0.0015161 seconds.

tic; accumarray(accmap, a, [1, numel(accmap)], @mean); toc
>>>Elapsed time is 1.733 seconds.
```
When running these commands with octave the first time, they run notably
slower. The values above are the best of several consequent tries. I'm
not sure what goes wrong with octaves mean function, but for me it's
painfully slow. If I'm doing something wrong here, please let me know.
