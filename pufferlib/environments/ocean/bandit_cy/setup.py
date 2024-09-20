from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module
extensions = [
    Extension(
        name="cy_bandit_cy",                # Name of the generated .so file
        sources=["cy_bandit_cy.pyx"],        # Cython source file
        include_dirs=[np.get_include()],     # Include numpy headers
        extra_compile_args=["-O3"],          # Optional: Add optimization flag for performance
    )
]

# Setup the extension
setup(
    name="cy_bandit_cy",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)
