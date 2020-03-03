# -*- coding: utf-8 -*-
"""Setup script for pompy package."""

from setuptools import setup
from os import path
from io import open


here = path.abspath(path.dirname(__file__))


setup(
    name='pompy',
    version='0.1.0',
    description='Puff-based odour plume model',
    author='Matt Graham, Annie Rak',
    license='MIT',
    url='https://github.com/annierak/fly_pompy',
    packages=['pompy'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
    ],
    include_package_data=True
)
