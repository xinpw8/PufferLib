from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension for the memory environment
extensions = [
    Extension(
        name="cy_memory_cy",  # Name of the generated Cython module
        sources=["cy_memory_cy.pyx"],  # Source Cython file
        include_dirs=[np.get_include()],  # Include directory for NumPy
        extra_compile_args=["-O3"],  # Optional: Optimization flags
    )
]

# Setup configuration
setup(
    name="cy_memory",
    ext_modules=cythonize(extensions, annotate=True),  # Annotate True to see line coverage
    zip_safe=False,
)
