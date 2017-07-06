from distutils.core import setup
from distutils.extension import Extension

setup(
    name = 'SymmetryFunctionLib',
    ext_modules = [
        Extension("SymmetryFunctions", ["SymmetryFunctionsC.c"]),
        Extension("SymmetryFunctionSet", ["SymmetryFunctionSetC.c"]),
    ]
)
