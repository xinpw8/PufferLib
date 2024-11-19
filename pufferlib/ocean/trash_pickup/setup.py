from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="cy_trash_pickup",
        sources=["cy_trash_pickup.pyx", "trash_pickup.c"],
        include_dirs=[np.get_include(), "."],  # Include NumPy headers
        language="c",
    ),
]

setup(
    name="cy_trash_pickup",
    ext_modules=cythonize(extensions)
)
