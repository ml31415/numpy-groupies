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

*numpy ufuncs*
Numpy offers bincount and ufunc.at for broadcasting. I recently added
a proof of concept implementation of accumarray based on these functions.
While bincount offers decent speed, the ufunc based functions are not
really competitive. Here is a recent benchmark print. accum_np marks
the naive python for-loop approach for reference:

```
sum -------------------------------
         accum_np       362.167
         accum_ufunc      5.969
         accum            8.683
amin ------------------------------
         accum_np       326.713
         accum_ufunc    158.091
         accum            9.777
amax ------------------------------
         accum_np       335.089
         accum_ufunc    157.735
         accum            9.673
prod ------------------------------
         accum_np       337.212
         accum_ufunc    149.197
         accum            7.047
all -------------------------------
         accum_np       415.426
         accum_ufunc    174.221
         accum            7.240
any -------------------------------
         accum_np       412.707
         accum_ufunc    169.546
         accum            6.654
mean ------------------------------
         accum_np       674.523
         accum_ufunc     10.295
         accum            7.995
std -------------------------------
         accum_np      1723.885
         accum_ufunc     19.235
         accum           11.468
nansum ----------------------------
         accum_np       728.556
         accum_ufunc     12.245
         accum            8.035
nanmin ----------------------------
         accum_np       599.000
         accum_ufunc    160.834
         accum           10.952
nanmax ----------------------------
         accum_np       605.791
         accum_ufunc    161.090
         accum            9.983
nanmean ---------------------------
         accum_np      2510.927
         accum_ufunc     23.382
         accum            7.986
nanstd ----------------------------
         accum_np      5096.134
         accum_ufunc     32.621
         accum           10.594
anynan ----------------------------
         accum_np       465.890
         accum_ufunc    145.399
         accum            7.000
allnan ----------------------------
         accum_np       467.948
         accum_ufunc    150.159
         accum            6.500
```