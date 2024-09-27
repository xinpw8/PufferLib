# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "pufferlib.environments.ocean.racing.cy_racing",
        ["pufferlib/environments/ocean/racing/cy_racing.pyx"],
        include_dirs=[numpy.get_include(), 'raylib-5.0_linux_amd64/include'],
        library_dirs=['raylib-5.0_linux_amd64/lib'],
        libraries=["raylib"],
        extra_compile_args=['-DPLATFORM_DESKTOP'],
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
