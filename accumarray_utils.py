

def get_alias_info(with_numpy=False):
    """This should be called only once by an accumarray_implementation.py file,
        i.e. it should be called at the point when the given implementation is imported.

        It returns two things. The first is a dict mapping strings and functions
        to the list of supported funciton names:     
            e.g. alias['add'] = 'sum'  and alias[sorted] = 'sort'   
        The second output is a list of functions names which should not support
        nan- prefixing.
    """

    alias_str = {
    'or': 'any', 
    'and': 'all',
    'add': 'sum',
    'plus': 'sum', 
    'multiply': 'prod',
    'product': 'prod',
    'times': 'prod',
    'amax': 'max',
    'maximum': 'max',
    'amin': 'min',
    'minimum': 'min',
    'split': 'array',
    'splice': 'array',
    'sorted': 'sort',
    'asort': 'sort',
    'asorted': 'sort',
    'rsorted': 'rsort',
    'dsort': 'rsort',
    'dsorted': 'rsort',
    }

    alias_builtin = {
     all: 'all', 
     any: 'any',
     max: 'max',
     min: 'min',
     sum: 'sum',
     sorted: 'sort',
     slice: 'array',
     list: 'array',
    }

    alias_numpy = {}    
    if with_numpy:
        import numpy as np
        alias_numpy = {
            np.add: 'sum',
            np.sum: 'sum',
            np.any: 'any',
            np.all: 'all',
            np.multiply: 'prod',
            np.prod: 'prod',
            np.amin: 'min',
            np.min: 'min',
            np.minimum: 'min',
            np.amax: 'max',
            np.max: 'max',
            np.maximum: 'max',
            np.mean: 'mean',
            np.std: 'std',
            np.var: 'var',
            np.array: 'array',
            np.asarray: 'array',
            np.sort: 'sort',
            np.nansum: 'nansum',
            np.nanmean: 'nanmean',
            np.nanvar: 'nanvar',
            np.nanmax: 'nanmax',
            np.nanmin: 'nanmin',
            np.nanstd: 'nanstd',
        }
    
    alias = alias_str.copy()
    alias.update(alias_builtin)
    alias.update(alias_numpy)

    no_separate_nan_version = ('sort','rsort','array','allnan','anynan')

    return alias, no_separate_nan_version


