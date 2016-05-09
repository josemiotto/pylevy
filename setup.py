#!/usr/bin/env python

from distutils.core import setup
import levy

setup(name='PyLevy',
      version=levy.__version__,
      author='Paul Harrison',
      author_email='pfh@logarithmic.net',
      url='http://www.logarithmic.net/pfh/pylevy',
      license='GPL',
      description='A package for calculating and fitting Levy stable distributions.',
      long_description=levy.__doc__,
      py_modules=['levy', 'levy_data', 'levy_approx_data'],
      options={'sdist': {'force_manifest': True}}
      )
