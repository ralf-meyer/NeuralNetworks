#!python
#cython: boundscheck=False, wraparound=False, cdivision=True 
from SymmetryFunctionsC cimport RadialSymmetryFunction, AngularSymmetryFunction
import numpy as _np 
from scipy.misc import comb
import time


cdef extern from "math.h":
    double cos(double m)
    double sin(double m)
    double exp(double m)
    double tanh(double m)
    double cosh(double m)
    double sqrt(double m)
    double M_PI

#gaussian and its derivative
cdef double gaussian(double r,double rs,double eta):
    return exp(-eta*(r-rs)**2)

cdef double derivative_gaussian(double r,double rs,double eta):
    return (-2.0*eta*(r-rs)*gaussian(r,rs,eta))
            
#angular symmetry function and its derivative
cdef double angular(double rij,double rik,double costheta,double eta,double lamb,double zeta):
    return 2**(1-zeta)* ((1 + lamb*costheta)**zeta * 
            exp(-eta*(rij**2+rik**2)))
            
cdef double derivative_angular(double rij,double rik,double costheta,double eta,double lamb,double zeta):
    sintheta = sqrt(abs(1.-costheta**2))
    return 2**(1-zeta)*(-lamb * zeta * sintheta *
            (1 + lamb*costheta)**(zeta-1) * 
            exp(-eta*(rij**2+rik**2))) 
            
#cos cutoff function and its derivative
cdef double cos_cutoff(double r,double cut):
    if r>cut:
        return 0
    else:
        return 0.5*(cos(M_PI*r/cut)+1)

cdef double derivative_cos_cutoff(double r,double cut):
    if r>cut:
        return 0
    else:
        return (-0.5*M_PI*sin(M_PI*r/cut)/cut)

#tanh cutoff function and its derivative
cdef double tanh_cutoff(double r,double cut):
    if r>cut:
        return 0
    else:
        return tanh(1.-r/cut)**3    

cdef double derivative_tanh_cutoff(double r,double cut):
    if r>cut:
        return 0
    else:
        return (-3.0*tanh(1-r/cut)**2 /
                (cosh(1-r/cut)**2*cut))

#evaluation of the radial symmetry function and its derivative (tanh_cutoff disabled)
cdef double eval_radial(double r,double rs,double eta,double cut,double cut_type):
#    if cut_type==1:
     return gaussian(r,rs,eta)*cos_cutoff(r,cut)
#    else:
#        return gaussian(r,rs,eta)*tanh_cutoff(r,cut)

cdef double eval_derivative_radial(double r,double rs,double eta,double cut,int cut_type):

#    if cut_type==1:
     return derivative_gaussian(r,rs,eta)*cos_cutoff(r,cut)+gaussian(r,rs,eta)*derivative_cos_cutoff(r,cut)
#    else:
#        return derivative_gaussian(r,rs,eta)*tanh_cutoff(r,cut)+gaussian(r,rs,eta)*derivative_tanh_cutoff(r,cut)

#evaluation of the angular symmetry function and its derivative (tanh_cutoff disabled)
cdef double eval_angular(double rij,double rik,double costheta,double eta,double lamb,double zeta,double cut,int cut_type):
    #if cut_type==1:
    return angular(rij,rik,costheta,eta,lamb,zeta)*cos_cutoff(rij,cut)*cos_cutoff(rik,cut)
    #else:
    #    return angular(rij,rik,eta,lamb,zeta)*tanh_cutoff(rij,cut)*tanh_cutoff(rik,cut)
    
cdef double eval_derivative_angular(double rij,double rik,double costheta,double eta,double lamb,double zeta,double cut,int cut_type):
    #if cut_type==1:
    return derivative_angular(rij,rik,costheta,eta,lamb,zeta)*cos_cutoff(rij,cut)*cos_cutoff(rik,cut)
    #else:
    #    return derivative_angular(rij,rik,costheta,eta,lamb,zeta)*tanh_cutoff(rij,cut)*tanh_cutoff(rik,cut)

class SymmetryFunctionSet(object):
    
    
    def __init__(self, atomtypes, cutoff = 7.):
        self.atomtypes = atomtypes
        self.cutoff = cutoff
        self.radial_sym_funs = []
        self.angular_sym_funs = []
        
    def add_radial_functions(self, rss, etas):
        for rs in rss:
            for eta in etas:
                self.radial_sym_funs.append(
                        RadialSymmetryFunction(rs, eta, self.cutoff))
                                                              
    def add_angular_functions(self, etas, zetas, lambs):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    self.angular_sym_funs.append(
                            AngularSymmetryFunction(eta, zeta, lamb, self.cutoff))
                    
    def add_angular_functions_new(self, etas, zetas, lambs, rss):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    for rs in rss:
                        self.angular_sym_funs.append(
                            AngularSymmetryFunction(eta, zeta, lamb, rs, self.cutoff))
    
    def add_radial_functions_evenly(self, N):
        rss = _np.linspace(0.,self.cutoff,N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        for rs, eta in zip(rss, etas):
            self.radial_sym_funs.append(
                        RadialSymmetryFunction(rs, eta, self.cutoff))
            
    def get_symmfun_properties(self):
        
        #get radial symmetry function parameters
        self.rs=_np.zeros(len(self.radial_sym_funs),dtype=_np.float)
        self.eta_r=_np.zeros(len(self.radial_sym_funs),dtype=_np.float)
        self.cut_r=_np.zeros(len(self.radial_sym_funs),dtype=_np.float)
        self.cut_type_r=_np.zeros(len(self.radial_sym_funs),dtype=_np.int32)
        for i,rad_fun in enumerate(self.radial_sym_funs):
            self.rs[i] = (<RadialSymmetryFunction>rad_fun).rs
            self.eta_r[i] = (<RadialSymmetryFunction>rad_fun).eta
            self.cut_r[i] = (<RadialSymmetryFunction>rad_fun).cut
            self.cut_type_r[i] = (<RadialSymmetryFunction>rad_fun).cut_type
            
        #get angular symmetry function parameters
        self.eta_a=_np.zeros(len(self.angular_sym_funs),dtype=_np.float)
        self.lamb=_np.zeros(len(self.angular_sym_funs),dtype=_np.float)
        self.zeta=_np.zeros(len(self.angular_sym_funs),dtype=_np.float)
        self.cut_a=_np.zeros(len(self.angular_sym_funs),dtype=_np.float)
        self.cut_type_a=_np.zeros(len(self.angular_sym_funs),dtype=_np.int32)
        for j,ang_fun in enumerate(self.angular_sym_funs):
            self.eta_a[j] = (<AngularSymmetryFunction>ang_fun).eta
            self.lamb[j] = (<AngularSymmetryFunction>ang_fun).lamb
            self.zeta[j] = (<AngularSymmetryFunction>ang_fun).zeta
            self.cut_a[j] = (<AngularSymmetryFunction>ang_fun).cut
            self.cut_type_a[j] = (<AngularSymmetryFunction>ang_fun).cut_type
            
                        
        
    def eval_geometry(self, geometry, derivative = False):
        # Returns a (Number of atoms) x (Size of G vector) matrix
        # The G vector doubles in size if derivatives are also requested
        start=time.time()
        cdef int N = len(geometry) # Number of atoms
        cdef int Nt = len(self.atomtypes) # Number of atomtypes
        cdef int Nr = len(self.radial_sym_funs) # Number of radial symmetry functions
        cdef int Na = len(self.angular_sym_funs) # Number of angular symmetry functions
        cdef int tj, tk
        cdef int i, j, k # indices for the atomic loops
        cdef int ir, ia, itj, itk, ind
        
        cdef int [:] types = _np.array([self.atomtypes.index(atom[0]) for atom in geometry], dtype=_np.int32)
        cdef double [:,:] xyzs = _np.array([atom[1] for atom in geometry], dtype=_np.float)
        
        cdef double rij, rik, costheta
        
        cdef double [:,:] out
        
        #get parameters of symmetry functions
        SymmetryFunctionSet.get_symmfun_properties(self)
        
        #make memory view out of python lists
        cdef:
            double [:] mem_rs  = self.rs
            double [:] mem_eta_r = self.eta_r
            double [:] mem_cut_r = self.cut_r
            int [:] mem_cut_type_r = self.cut_type_r
        
            double [:] mem_eta_a = self.eta_a
            double [:] mem_lamb = self.lamb
            double [:] mem_zeta = self.zeta
            double [:] mem_cut_a = self.cut_a
            int [:] mem_cut_type_a = self.cut_type_a
        
        #create pointers to memory views
        cdef:
            double* rs_ptr = &mem_rs[0]
            double* eta_r_ptr = &mem_eta_r[0]
            double* cut_r_ptr = &mem_cut_r[0]
            int* cut_type_r_ptr = &mem_cut_type_r[0]
        
            double* eta_a_ptr = &mem_eta_a[0]
            double* lamb_ptr = &mem_lamb[0]
            double* zeta_ptr = &mem_zeta[0]
            double* cut_a_ptr = &mem_cut_a[0]
            int* cut_type_a_ptr = &mem_cut_type_a[0]
        
                        
        if derivative == False:
            out = _np.zeros((N, Nr*Nt + comb(Nt, 2, exact = True, repetition = True)*Na), dtype=_np.float)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    tj = types[j]
                    # Radial function
                    rij = sqrt((xyzs[i,0]-xyzs[j,0])**2 + (xyzs[i,1]-xyzs[j,1])**2 +
                                   (xyzs[i,2]-xyzs[j,2])**2)
                    for itj in range(Nt):
                        for ir in range(Nr):                        
                            if itj == tj:
                                out[i,itj*Nr+ir] += eval_radial(rij,rs_ptr[ir],eta_r_ptr[ir],cut_r_ptr[ir],cut_type_r_ptr[ir])
                                #out[i,itj*Nr+ir] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).evaluate(rij) 
                    # Angular functions
                    for k in range(N):
                        if i == k or j == k:
                            continue                        
                        tk = types[k]
                        rik = sqrt((xyzs[i,0]-xyzs[k,0])**2 + (xyzs[i,1]-xyzs[k,1])**2 +
                                   (xyzs[i,2]-xyzs[k,2])**2)
                        costheta = ((xyzs[j,0]-xyzs[i,0])*(xyzs[k,0]-xyzs[i,0])+
                                    (xyzs[j,1]-xyzs[i,1])*(xyzs[k,1]-xyzs[i,1])+
                                    (xyzs[j,2]-xyzs[i,2])*(xyzs[k,2]-xyzs[i,2]))/(rij*rik)
                        ind = 0
                        for itj in range(Nt):
                            for itk in range(itj, Nt):
                                for ia in range(Na):  
                                    if itj == tj and itk == tk:
                                        out[i,Nt*Nr+ ind*Na+ ia] += eval_angular(rij,rik,costheta,eta_a_ptr[ia],lamb_ptr[ia],zeta_ptr[ia],cut_a_ptr[ia],cut_type_a_ptr[ia])
                                        #out[i,Nt*Nr+ ind*Na+ ia] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).evaluate(rij,rik,costheta)
                                ind += 1  
        else:
            out = _np.zeros((N, 2*(Nr*Nt + comb(Nt, 2, exact = True, repetition = True)*Na)), dtype=_np.float)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    tj = types[j]
                    # Radial function
                    rij = sqrt((xyzs[i,0]-xyzs[j,0])**2 + (xyzs[i,1]-xyzs[j,1])**2 +
                                   (xyzs[i,2]-xyzs[j,2])**2)
                    for itj in range(Nt):
                        for ir in range(Nr):                        
                            if itj == tj:
                                out[i,itj*2*Nr+2*ir] += eval_radial(rij,rs_ptr[ir],eta_r_ptr[ir],cut_r_ptr[ir],cut_type_r_ptr[ir])
                                out[i,itj*2*Nr+2*ir+1] += eval_derivative_radial(rij,rs_ptr[ir],eta_r_ptr[ir],cut_r_ptr[ir],cut_type_r_ptr[ir])
                                #out[i,itj*2*Nr+2*ir] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).evaluate(rij) 
                                #out[i,itj*2*Nr+2*ir+1] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).derivative(rij) 
                    # Angular functions
                    for k in range(N):
                        if i == k or j == k:
                            continue                        
                        tk = types[k]
                        rik = sqrt((xyzs[i,0]-xyzs[k,0])**2 + (xyzs[i,1]-xyzs[k,1])**2 +
                                   (xyzs[i,2]-xyzs[k,2])**2)
                        costheta = ((xyzs[j,0]-xyzs[i,0])*(xyzs[k,0]-xyzs[i,0])+
                                    (xyzs[j,1]-xyzs[i,1])*(xyzs[k,1]-xyzs[i,1])+
                                    (xyzs[j,2]-xyzs[i,2])*(xyzs[k,2]-xyzs[i,2]))/(rij*rik)
                        ind = 0
                        for itj in range(Nt):
                            for itk in range(itj, Nt):
                                for ia in range(Na):  
                                    if itj == tj and itk == tk:
                                        out[i,Nt*2*Nr+ ind*2*Na+ 2*ia] += eval_angular(rij,rik,costheta,eta_a_ptr[ia],lamb_ptr[ia],zeta_ptr[ia],cut_a_ptr[ia],cut_type_a_ptr[ia])
                                        out[i,Nt*2*Nr+ ind*2*Na+ 2*ia + 1] += eval_derivative_angular(rij,rik,costheta,eta_a_ptr[ia],lamb_ptr[ia],zeta_ptr[ia],cut_a_ptr[ia],cut_type_a_ptr[ia])
                                        #out[i,Nt*2*Nr+ ind*2*Na+ 2*ia] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).evaluate(rij,rik,costheta)
                                        #out[i,Nt*2*Nr+ ind*2*Na+ 2*ia + 1] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).derivative(rij,rik,costheta)
                                ind += 1  
        return out
