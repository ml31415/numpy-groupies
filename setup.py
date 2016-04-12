#!/usr/bin/env python

from setuptools import setup

setup(name='numpy_groupies',
      version='0.9.1',
      author="@ml31415 and @d1manson",
      license='Public Domain',
      description="Optimised tools for group-indexing operations: aggregated sum and more.",
      url="https://github.com/ml31415/numpy-groupies",
      download_url="https://github.com/ml31415/numpy-groupies/archive/master.zip",
      keywords=[ "accumarray", "aggregate", "groupby", "grouping", "indexing"],
      packages=['numpy_groupies'],
      tests_require=[''],
      requires=[],
      classifiers=['Development Status :: 4 - Beta', 'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   ]
      )
