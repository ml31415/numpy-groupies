#!/usr/bin/env python

import os
from setuptools import setup
from distutils import log
from distutils.command.clean import clean
from distutils.dir_util import remove_tree

base_path = os.path.dirname(os.path.abspath(__file__))

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
      version='0.9.4',
      author="@ml31415 and @d1manson",
      license='BSD',
      description="Optimised tools for group-indexing operations: aggregated sum and more.",
      url="https://github.com/ml31415/numpy-groupies",
      download_url="https://github.com/ml31415/numpy-groupies/archive/master.zip",
      keywords=[ "accumarray", "aggregate", "groupby", "grouping", "indexing"],
      packages=['numpy_groupies'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      requires=[],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   ],
      cmdclass=dict(clean=NumpyGroupiesClean)
      )
