from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'fastloop.pyx app',
  ext_modules = cythonize("caculate.pyx"),
)