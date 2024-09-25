from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    # enduro_cy
    Extension(
        name="pufferlib.environments.ocean.enduro_cy.cy_enduro_cy",
        sources=["pufferlib/environments/ocean/enduro_cy/cy_enduro_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    
    # enduro_game
    Extension(
        name="pufferlib.environments.ocean.enduro_cy.enduro_game",
        sources=["pufferlib/environments/ocean/enduro_cy/enduro_game.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    
]

setup(
    name="cython_extensions",
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
