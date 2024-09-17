from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "ocean_cy_so",  # Name of the output .so file
        sources=["cy_ocean_cy.pyx"],  # Path to the Cython file
        include_dirs=[np.get_include(), "/puffertank/pufferlib/pufferlib/environments/ocean/ocean_cy"],
        extra_compile_args=["-O3"],  # Optional: additional compiler options
    )
]

setup(
    name="ocean_cy",
    ext_modules=cythonize(ext_modules),
)
