from distutils.core import setup
from Cython.Build import cythonize

# compiler_directives={'boundscheck': False, 'wraparound': False}
#deprecated setup file
setup(
    name = 'SymmetryFuns',
    ext_modules = cythonize(["NeuralNetworks/SymmetryFunctionsC.pyx", "NeuralNetworks/SymmetryFunctionSetC.pyx"]), 
)
