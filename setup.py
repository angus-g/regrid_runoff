from setuptools import setup, find_packages
from Cython.Build import cythonize

import numpy as np

setup(
    name="regrid_runoff",
    ext_modules=cythonize(
        "regrid_runoff/regrid_tools.pyx",
        language_level="3",
        aliases={"NP_INCLUDE": np.get_include()},
    ),
    packages=find_packages(),
)
