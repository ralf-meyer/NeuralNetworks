from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import sys,shutil,os

if sys.argv[1]=="rebuild":
    try:
    	shutil.rmtree('build')
        os.remove("SymmetryFunctionsC.c")
        os.remove("SymmetryFunctionsC.so")
        os.remove("SymmetryFunctionSetC.c")
        os.remove("SymmetryFunctionSetC.so")
    except:
	print("build not found")
    sys.argv[1]="build"

setup(
    name = 'SymmetryFunctionLib',
    ext_modules = cythonize([
        Extension("SymmetryFunctionsC", ["SymmetryFunctionsC.pyx"]),
        Extension("SymmetryFunctionSetC", ["SymmetryFunctionSetC.pyx"]),
    ])
)

shutil.move("build/lib.linux-x86_64-2.7/SymmetryFunctionSetC.so","SymmetryFunctionSetC.so")
shutil.move("build/lib.linux-x86_64-2.7/SymmetryFunctionsC.so","SymmetryFunctionsC.so")
