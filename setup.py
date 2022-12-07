from setuptools   import setup, find_packages
from Cython.Build import cythonize

setup(name        = 'pet_code'     ,
      packages    = find_packages(),
      ext_modules = cythonize('pet_code/src/io_util.pyx', compiler_directives={'language_level' : "3"})
     )
