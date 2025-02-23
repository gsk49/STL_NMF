#!/usr/bin/env python3

from setuptools import setup
import os
import sys

setup(name='stereoscope',
            version='0.3.1',
            description='Integration of ST and SC data',
            author='Alma Andersson',
            author_email='alma.andersson@scilifelab.se',
            url='http://github.com/almaan/stereoscope',
            download_url='https://github.com/almaan/stereoscope/archive/v_03.tar.gz',
            license='MIT',
            packages=['stsc'],
            python_requires='>3.7.0', # 3.9.19 (3.9.12?)
            # install_requires=[
            #                 'torch>=1.1.0', ##2.4.0
            #                 'numba==0.54.1', ## 0.55.2
            #                 'numpy==1.20.0',
            #                 'pandas>=0.25.0', #1.4.0
            #                 'matplotlib>=3.1.0', #3.4.0
            #                 'scikit-learn>=0.20.0', #1.5.1
            #                 'umap-learn>=0.4.1', #0.5.6
            #                 'anndata', # .8 THIS BREAKS NUMPY!!!!!
            #                 'scipy', #1.7.3
            #                 'Pillow', # 8.0.0
            #           ],
            entry_points={'console_scripts': ['stereoscope = stsc.__main__:main',
                                             ]
                         },
            zip_safe=False)

