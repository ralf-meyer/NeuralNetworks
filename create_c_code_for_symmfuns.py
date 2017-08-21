from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from sympy import *
from NeuralNetworks import SympyToC as converter
import numpy as np
import sys,shutil,os


def write_symmfuns(symname_list, symfunc_list,file_name="MySymmetryFunctions"):
    if len(symname_list)==len(symfunc_list):
        #write files
        cy_code,pxd_code=converter.sympyToCython(symname_list, symfunc_list)
        with open(file_name+".pyx","w") as file:
            file.write(cy_code)
            file.close()
        with open(file_name+".pxd","w") as file:
            file.write(pxd_code)
            file.close()

def compile_symmfuns(name_list=["MySymmetryFunctions","CartesianDerivatives"]):
    
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
        
        ext_list=[Extension("SymmetryFunctionsC", ["SymmetryFunctionsC.pyx"]),
                    Extension("SymmetryFunctionSetC", ["SymmetryFunctionSetC.pyx"]),
                    Extension("FermiBoxes", ["FermiBoxes.pyx"])]

        for file_name in name_list:
            ext_list.append(Extension(file_name,[file_name+".pyx"]))

        modules = cythonize(ext_list)

        setup(
        name = 'SymmetryFunctionLib',
        ext_modules = modules)
        shutil.move("build/lib.linux-x86_64-2.7/SymmetryFunctionSetC.so","SymmetryFunctionSetC.so")
        shutil.move("build/lib.linux-x86_64-2.7/SymmetryFunctionsC.so","SymmetryFunctionsC.so")
        for file_name in name_list:
            shutil.move("build/lib.linux-x86_64-2.7/"+file_name+".so",file_name+".so")
    else:
        print("Symmetry functions length "+str(len(symfunc_list))+" and function name length "+str(len(symname_list))+" does not match!")
    
#####################################################################

symname_list=[]
symfunc_list=[]
#specify your symmetry functions here (use predifined symbols)

rij,rik,costheta= symbols('rij rik costheta')
variables_list=['rij','rik','costheta','xi','yi','zi','xj','yj','zj','xk','yk','zk']
rs,eta,lamb,zeta,cut=symbols('rs eta lamb zeta cut',constant = True)

#radial
radial_fun = exp(-eta*(rij-rs)**2)*cos(sqrt(3)*eta*(rij-rs)*0.5)
symname_list.append("radial")

#angular
angular_fun = 2**(1-zeta)* ((1 + lamb*costheta)**zeta)* exp(-eta*(rij**2+rik**2))
symname_list.append("angular")


#cutoffs
epsilon=1e-16
border=Heaviside(-rij+cut)
#
#cos-cutoff
cos_cutoff=0.5*(cos(np.pi*rij/cut)+1)
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
symname_list.append("cutoff")


#
#Write c-files for symmetry functions
#write_symmfuns(symname_list, symfunc_list,file_name="MySymmetryFunctions")

#Create cartesian derivatives for force calculations

xi,yi,zi,xj,yj,zj,xk,yk,zk= symbols('xi yi zi xj yj zj xk yk zk')
pi,pj,pk=Point(xi,yi,zi),Point(xj,yj,zj),Point(xk,yk,zk)

rij = pi.distance(pj)
rik = pi.distance(pk)
rjk = pj.distance(pk)
costheta =(rij**2+rik**2-rjk**2)/(2*rij*rik)


#redefine functions
#radial
radial_fun_xyz = exp(-eta*(rij-rs)**2)*cos(sqrt(3)*eta*(rij-rs)*0.5)
symfunc_list.append([radial_fun,radial_fun_xyz])
#angular
angular_fun_xyz = 2**(1-zeta)* ((1 + lamb*costheta)**zeta)* exp(-eta*(rij**2+rik**2))
symfunc_list.append([angular_fun,angular_fun_xyz])
#cos-cutoff
cos_cutoff=0.5*(cos(np.pi*rij/cut)+1)
#tanh-cutoff
tanh_cutoff=tanh(1.-rij/cut)**3
#no cutoff
no_cutoff=1+epsilon*rij+epsilon*cut
#cutoff
cutoff_fun_xyz=no_cutoff
symfunc_list.append([cutoff_fun,cutoff_fun_xyz])

converter.sympyToCpp(symname_list,symfunc_list,variables_list,True)
#write_symmfuns(symname_list, symfunc_list,file_name="CartesianDerivatives")

#compile files
#compile_symmfuns(name_list=["MySymmetryFunctions","CartesianDerivatives"])