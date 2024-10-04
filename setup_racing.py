from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "pufferlib.environments.ocean.racing.cy_racing",
        sources=["pufferlib/environments/ocean/racing/cy_racing.c",
                 "pufferlib/environments/ocean/racing/racing.c"],
        include_dirs=[np.get_include(), "pufferlib/environments/ocean/racing"],
    )
]


setup(
    name="pufferlib_racing",
    ext_modules=cythonize(ext_modules, language_level=3),
    include_dirs=[np.get_include()]
)

