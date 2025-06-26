#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '../..')

from distutils.core import setup, Extension
import distutils.util
import numpy as np

# Include Abseil dependency
abseil_dir = os.path.join('.', 'abseil-cpp')

# Get the include path for the environment binding
env_binding_path = os.path.join('..', 'env_binding.h')

ext = Extension(
    'binding',
    sources=['binding.c'],
    include_dirs=['.', '..', abseil_dir, np.get_include()],
    language='c++',
    extra_compile_args=['-std=c++17', '-fPIC', '-fpermissive'],
    extra_link_args=['-std=c++17'],
)

# Build in-place using binding.cpp directly

if __name__ == '__main__':
    # Ensure we compile the C++ source file
    if os.path.exists('binding.cpp'):
        ext.sources = ['binding.cpp']
    setup(
        name='chess_binding',
        ext_modules=[ext]
    ) 