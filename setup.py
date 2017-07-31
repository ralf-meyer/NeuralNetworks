from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from sympy import *
from NeuralNetworks import SympyToCython as converter
import numpy as np
import sys,shutil,os




def write_and_compile_symmfuns(symname_list, symfunc_list):
    
    if sys.argv[1]=="rebuild" or sys.argv[1]=="clean":
        try:
            os.remove("SymmetryFunctionsC.c")
            os.remove("SymmetryFunctionsC.so")
            os.remove("SymmetryFunctionSetC.c")
            os.remove("SymmetryFunctionSetC.so")
            os.remove("FermiBoxes.c")
            os.remove("FermiBoxes.so")
            os.remove("MySymmetryFunctions.pyx")
            os.remove("MySymmetryFunctions.pxd")
            shutil.rmtree('build')
        except:
            print("build not found")
       
    if sys.argv[1]=="rebuild":   
        sys.argv[1]="build"
    if sys.argv[1]=="build":
        
        if len(symname_list)==len(symfunc_list):
            #write files
            cy_code,pxd_code=converter.sympyToCython(symname_list, symfunc_list)
            with open("MySymmetryFunctions.pyx","w") as file:
                file.write(cy_code)
                file.close()
            with open("MySymmetryFunctions.pxd","w") as file:
                file.write(pxd_code)
                file.close()
    
            modules = cythonize([
                    Extension("SymmetryFunctionsC", ["SymmetryFunctionsC.pyx"]),
                    Extension("SymmetryFunctionSetC", ["SymmetryFunctionSetC.pyx"]),
                    Extension("MySymmetryFunctions", ["MySymmetryFunctions.pyx"]),
                    Extension("FermiBoxes", ["FermiBoxes.pyx"])])

            setup(
            name = 'SymmetryFunctionLib',
            ext_modules = modules)
            shutil.move("build/lib.linux-x86_64-2.7/MySymmetryFunctions.so","MySymmetryFunctions.so")
            shutil.move("build/lib.linux-x86_64-2.7/SymmetryFunctionSetC.so","SymmetryFunctionSetC.so")
            shutil.move("build/lib.linux-x86_64-2.7/SymmetryFunctionsC.so","SymmetryFunctionsC.so")
        else:
            print("Symmetry functions length "+str(len(symfunc_list))+" and function name length "+str(len(symname_list))+" does not match!")
    
#####################################################################

symname_list=[]
symfunc_list=[]
#specify your symmetry functions here (use predifined symbols)

rij,rik,costheta,rs,eta,lamb,zeta,cut= symbols('rij rik costheta rs eta lamb zeta cut')

#radial
radial_fun = exp(-eta*(rij-rs)**2)
symname_list.append("eval_radial")
symfunc_list.append(radial_fun)

der_radial_fun=diff(radial_fun,rij)
symname_list.append("eval_derivative_radial")
symfunc_list.append(der_radial_fun)

#
#angular
angular_fun = 2**(1-zeta)* ((1 + lamb*costheta)**zeta)* exp(-eta*(rij**2+rik**2))
symname_list.append("eval_angular")
symfunc_list.append(angular_fun)

der_angular_fun= diff(angular_fun,costheta)
symname_list.append("eval_derivative_angular")
symfunc_list.append(der_angular_fun)
#
#cutoffs
epsilon=1e-16
border=Heaviside(-rij+cut)
#
#cos-cutoff
cos_cutoff=0.5*(cos(pi*rij/cut)+1)
der_cos_cutoff=diff(cos_cutoff,rij)
#
#tanh-cutoff
tanh_cutoff=tanh(1.-rij/cut)**3
der_tanh_cutoff=diff(tanh_cutoff,rij)
#
#no cutoff
#rij and cut always have to be in the formula
no_cutoff=1+epsilon*rij+epsilon*cut
der_no_cutoff=0+epsilon*rij+epsilon*cut
#set cutoff
cutoff_fun=no_cutoff#*border (border has to be included for not constant cutoffs)
symname_list.append("eval_cutoff")
symfunc_list.append(cutoff_fun)

der_cutoff_fun=der_no_cutoff#*border
symname_list.append("eval_derivative_cutoff")
symfunc_list.append(der_cutoff_fun)
#
#Write c-files and compile 
write_and_compile_symmfuns(symname_list, symfunc_list)
