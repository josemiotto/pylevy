#!/usr/bin/env python

from distutils.core import setup
import levy

setup(name='PyLevy',
      version=levy.__version__,
      author='Paul Harrison, Jose Miotto',
      url='https://github.com/josemiotto/pylevy',
      license='GPL',
      description='A package for calculating and fitting Levy stable distributions.',
      long_description=levy.__doc__,
      packages=['levy'],
      package_data={'levy': ['cdf.npz', 'pdf.npz', 'lower_limit.npz', 'upper_limit.npz']},
      options={'sdist': {'force_manifest': True}},
     )
