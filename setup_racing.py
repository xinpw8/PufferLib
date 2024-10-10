from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "pufferlib.environments.ocean.enduro_clone.cy_enduro_clone",
        ["pufferlib/environments/ocean/enduro_clone/cy_enduro_clone.pyx"],
        include_dirs=[numpy.get_include(), 'raylib-5.0_linux_amd64/include'],
        library_dirs=['raylib-5.0_linux_amd64/lib'],
        libraries=["raylib"],
        extra_compile_args=['-DPLATFORM_DESKTOP'],
        language="c",
    ),
]

setup(
    name="Enduro Clone",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
