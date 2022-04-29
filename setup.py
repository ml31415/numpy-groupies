#!/usr/bin/env python

import os
import versioneer
from setuptools import setup
from distutils import log
from distutils.command.clean import clean
from distutils.dir_util import remove_tree

base_path = os.path.dirname(os.path.abspath(__file__))


long_description = """
This package consists of a couple of optimised tools for doing things that can roughly be 
considered "group-indexing operations". The most prominent tool is `aggregate`.

`aggregate` takes an array of values, and an array giving the group number for each of those 
values. It then returns the sum (or mean, or std, or any, ...etc.) of the values in each group. 
You have probably come across this idea before, using `matlab` accumarray, `pandas` groupby, 
or generally MapReduce algorithms and histograms.

There are different implementations of `aggregate` provided, based on plain `numpy`, `numba`
and `weave`. Performance is a main concern, and so far we comfortably beat similar 
implementations in other packages (check the benchmarks).
"""


class NumpyGroupiesClean(clean):
    """Custom clean command to tidy up the project root."""
    def run(self):
        clean.run(self)
        for folder in ('build', 'numpy_groupies.egg-info'):
            path = os.path.join(base_path, folder)
            if os.path.isdir(path):
                remove_tree(path, dry_run=self.dry_run)
        if not self.dry_run:
            self._rm_walk()

    def _rm_walk(self):
        for path, dirs, files in os.walk(base_path):
            if any(p.startswith('.') for p in path.split(os.path.sep)):
                # Skip hidden directories like the git folder right away
                continue
            if path.endswith('__pycache__'):
                remove_tree(path, dry_run=self.dry_run)
            else:
                for fname in files:
                    if fname.endswith('.pyc') or fname.endswith('.so'):
                        fpath = os.path.join(path, fname)
                        os.remove(fpath)
                        log.info("removing '%s'", fpath)


setup(name='numpy_groupies',
      version=versioneer.get_version(),
      author="@ml31415 and @d1manson",
      author_email="npgroupies@occam.com.ua",
      license='BSD',
      description="Optimised tools for group-indexing operations: aggregated sum and more.",
      long_description=long_description,
      url="https://github.com/ml31415/numpy-groupies",
      download_url="https://github.com/ml31415/numpy-groupies/archive/master.zip",
      keywords=[ "accumarray", "aggregate", "groupby", "grouping", "indexing"],
      packages=['numpy_groupies'],
      install_requires=[],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'numpy', 'numba'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   ],
      cmdclass=dict(clean=NumpyGroupiesClean, **versioneer.get_cmdclass()),
)
