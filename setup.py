#!/usr/bin/env python

import os
import versioneer
from setuptools import setup, Command
from shutil import rmtree

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


class Clean(Command):
    description = "clean up temporary files from 'build' command"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for folder in ('build', 'dist', 'intnan.egg-info'):
            path = os.path.join(base_path, folder)
            if os.path.isdir(path):
                print("removing '{}' (and everything under it)".format(path))
                if not self.dry_run:
                    rmtree(path)
        self._rm_walk()

    def _rm_walk(self):
        for path, dirs, files in os.walk(base_path):
            if any(p.startswith('.') for p in path.split(os.path.sep)):
                # Skip hidden directories like the git folder right away
                continue
            if path.endswith('__pycache__'):
                print("removing '{}' (and everything under it)".format(path))
                if not self.dry_run:
                    rmtree(path)
            else:
                for fname in files:
                    if fname.endswith('.pyc') or fname.endswith('.so'):
                        fpath = os.path.join(path, fname)
                        print("removing '{}'".format(fpath))
                        if not self.dry_run:
                            os.remove(fpath)


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
      install_requires=['numpy', 'numba'],
      extras_require={'tests': ['pytest']},
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10'],
      cmdclass=dict(clean=Clean, **versioneer.get_cmdclass()),
)
