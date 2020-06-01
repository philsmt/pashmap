#!/usr/bin/env python

# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from pathlib import Path

from setuptools import setup, find_packages


with open(Path(__file__).parent / 'README.md', 'r') as f:
    long_description = f.read()


setup(
    name='pashmap',
    version='1.0.0a0',
    description='Tools to perform PArallelized SHared memory MAP '
                'operations on large data sets using zero copies.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Philipp Schmidt',
    author_email='philipp.schmidt@xfel.eu',
    license='BSD-3-Clause',

    packages=find_packages(),

    python_requires='>=3.6',
    install_requires=['numpy'],

    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Physics',
    ]
)
