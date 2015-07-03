#!/usr/bin/env python

"""
setup.py file 
"""

#%% Load packages
from setuptools import setup, find_packages  # Always prefer setuptools over distutils

#from codecs import open  # To use a consistent encoding
from os import path
import os,sys
import numpy as np
import platform


#%% Test suite


packages=['aggregate']
    
setup (name = 'aggregate',
       version = '0.2',
       author      = "@ml31415 and @d1manson",
       description = "Replacement for Matlab's accumarray function.",
       author_email='xxx@gmail.com',
	license="Not set",
	keywords=[ "accumarray aggregate"],
        packages=packages,
	tests_require=[''],
        zip_safe=False,
	requires=['numpy'],
	classifiers=['Development Status :: 4 - Beta', 'Intended Audience :: Science/Research', 
	      'Programming Language :: Python :: 2',
	      'Programming Language :: Python :: 2.7',
	      'Programming Language :: Python :: 3',
	      'Programming Language :: Python :: 3.3',
	      'Programming Language :: Python :: 3.4', 
	      ]
       )


