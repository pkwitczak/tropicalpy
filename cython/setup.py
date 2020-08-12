from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name = "tropicalpy",
    version='0.0.1b',
    description='Python Cython-based tropical geometry',
    author='Jeremy Lane, Piotr Witczak',
    author_email='pkwitczak@gmail.com',
    ext_modules = cythonize("tropicalpy.pyx"),
    include_dirs=[numpy.get_include()]
)