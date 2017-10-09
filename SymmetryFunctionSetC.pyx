#!python
#cython: boundscheck=False, wraparound=False, cdivision=True 
from SymmetryFunctionsC cimport RadialSymmetryFunction, AngularSymmetryFunction
from MySymmetryFunctions cimport eval_radial,eval_derivative_radial,eval_angular,\
eval_derivative_angular,eval_cutoff,eval_derivative_cutoff

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
#            #input length of functions 
#            int cut_len=0
#            int radial_len=0
#            int angular_len=0
#            
            double [:] mem_rs  = self.rs   
            double [:] mem_eta_r = self.eta_r   
            double [:] mem_cut_r = self.cut_r
            double [:] mem_eta_a = self.eta_a
            double [:] mem_lamb = self.lamb
            double [:] mem_zeta = self.zeta
            double [:] mem_cut_a = self.cut_a
            
#        if len(self.rs) > 0:
#            radial_len+=1
#        if len(self.eta_r)>0:
#            radial_len+=1
#        if len(self.cut_r)>0:
#            cut_len+=2
#            
#        if len(self.eta_a)>0:
#            angular_len+=1
#        if len(self.lamb)>0:
#            angular_len+=1
#        if len(self.zeta)>0:
#            angular_len+=1         
#        if len(self.cut_a)>0:
#            if cut_len==0:
#                cut_len+=2
#            
#        if radial_len>0:
#            radial_len+=1 #if radial not constant add rij as parameter
#        
#        if angular_len>0:
#            angular_len+=3 #if angular not constant add rij,rik and costheta as parameters
        
#        #create pointers to memory views
        cdef:
            double* rs_ptr = &mem_rs[0]
            double* eta_r_ptr = &mem_eta_r[0]
            double* cut_r_ptr = &mem_cut_r[0]
        
            double* eta_a_ptr = &mem_eta_a[0]
            double* lamb_ptr = &mem_lamb[0]
            double* zeta_ptr = &mem_zeta[0]
            double* cut_a_ptr = &mem_cut_a[0]


#        cdef:
#
#            double [] radial_values = double [radial_len]
#            double* radial_ptr = &radial_values[0]
#
#            double [] angular_values = double [angular_len]
#            double* angular_ptr = &angular_values[0]
#
#            double [] cut_values = double [cut_r]
#            double* cut_ptr = &cut_values[0]


        
                        
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
                    #radial_values[1]= rij
                    #angular_values[3] = rij
                    
                    for itj in range(Nt):
                        for ir in range(Nr):                        
                            if itj == tj:
                                #radial_values[0] = eta[ia]
                                #radial_values[2] = rs[ia]
                                
                                #out[i,itj*Nr+ir] += eval_radial(radial_ptr)*eval_cutoff(cut_ptr)
                                out[i,itj*Nr+ir] += eval_radial(eta_r_ptr[ir],rij,rs_ptr[ir])*eval_cutoff(cut_r_ptr[ir],rij)
                                #out[i,itj*Nr+ir] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).evaluate(rij) 
                    # Angular functions
                    for k in range(N):
                        if i == k or j == k:
                            continue                        
                        tk = types[k]
                        rik = sqrt((xyzs[i,0]-xyzs[k,0])**2 + (xyzs[i,1]-xyzs[k,1])**2 +
                                   (xyzs[i,2]-xyzs[k,2])**2)
                        #angular_values[4] = rik
                        
                        costheta = ((xyzs[j,0]-xyzs[i,0])*(xyzs[k,0]-xyzs[i,0])+
                                    (xyzs[j,1]-xyzs[i,1])*(xyzs[k,1]-xyzs[i,1])+
                                    (xyzs[j,2]-xyzs[i,2])*(xyzs[k,2]-xyzs[i,2]))/(rij*rik)
                        #angular_values[0]= costheta

                        
                        ind = 0
                        for itj in range(Nt):
                            for itk in range(itj, Nt):
                                for ia in range(Na):  
                                    if itj == tj and itk == tk:
                                        #angular_values[1] = eta[ia]
                                        #angular_values[2] = lamb[ia]
                                        #angular_values[5] = zeta[ia]
                                        #out[i,Nt*Nr+ ind*Na+ ia] += eval_angular(angular_ptr)*eval_cutoff(cut_ptr)
                                        out[i,Nt*Nr+ ind*Na+ ia] += eval_angular(costheta,eta_a_ptr[ia],lamb_ptr[ia],rij,rik,zeta_ptr[ia])*eval_cutoff(cut_a_ptr[ia],rij)
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
                    
                    #radial_values[1]= rij
                    #angular_values[3] = rij
                    
                    for itj in range(Nt):
                        for ir in range(Nr):                        
                            if itj == tj:
                                #radial_values[0] = eta[ia]
                                #radial_values[2] = rs[ia]
                                
                                #out[i,itj*Nr+ir] += eval_radial(radial_ptr)*eval_cutoff(cut_ptr)
                                #out[i,itj*2*Nr+2*ir+1] += eval_derivative_radial(radial_ptr)*eval_cutoff(cut_ptr)\
                                #                          +eval_radial(radial_ptr)*eval_derivative_cutoff(cut_ptr)
                                out[i,itj*Nr+ir] += eval_radial(eta_r_ptr[ir],rij,rs_ptr[ir])*eval_cutoff(cut_r_ptr[ir],rij)
                                out[i,itj*2*Nr+2*ir+1] += eval_derivative_radial(eta_r_ptr[ir],rij,rs_ptr[ir])*eval_cutoff(cut_r_ptr[ir],rij)\
                                                          +eval_radial(eta_r_ptr[ir],rij,rs_ptr[ir])*eval_derivative_cutoff(cut_r_ptr[ir],rij)
                                #out[i,itj*2*Nr+2*ir] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).evaluate(rij) 
                                #out[i,itj*2*Nr+2*ir+1] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).derivative(rij) 
                    # Angular functions
                    for k in range(N):
                        if i == k or j == k:
                            continue                        
                        tk = types[k]
                        rik = sqrt((xyzs[i,0]-xyzs[k,0])**2 + (xyzs[i,1]-xyzs[k,1])**2 +
                                   (xyzs[i,2]-xyzs[k,2])**2)
                        #angular_values[4] = rik
                     
                        costheta = ((xyzs[j,0]-xyzs[i,0])*(xyzs[k,0]-xyzs[i,0])+
                                    (xyzs[j,1]-xyzs[i,1])*(xyzs[k,1]-xyzs[i,1])+
                                    (xyzs[j,2]-xyzs[i,2])*(xyzs[k,2]-xyzs[i,2]))/(rij*rik)
                        #angular_values[0]= costheta
                        
                        ind = 0
                        for itj in range(Nt):
                            for itk in range(itj, Nt):
                                for ia in range(Na):  
                                    if itj == tj and itk == tk:
                                        #angular_values[1] = eta[ia]
                                        #angular_values[2] = lamb[ia]
                                        #angular_values[5] = zeta[ia]
                                        
                                        #out[i,Nt*Nr+ ind*Na+ ia] += eval_angular(angular_ptr)*eval_cutoff(cut_ptr)
                                        #out[i,Nt*2*Nr+ ind*2*Na+ 2*ia + 1] += eval_derivative_angular(angular_ptr)*eval_cutoff(cut_ptr)
                                        out[i,Nt*Nr+ ind*Na+ ia] += eval_angular(costheta,eta_a_ptr[ia],lamb_ptr[ia],rij,rik,zeta_ptr[ia])*eval_cutoff(cut_a_ptr[ia],rij)
                                        out[i,Nt*2*Nr+ ind*2*Na+ 2*ia + 1] += eval_derivative_angular(costheta,eta_a_ptr[ia],lamb_ptr[ia],rij,rik,zeta_ptr[ia])*eval_cutoff(cut_a_ptr[ia],rij)
                                        #out[i,Nt*2*Nr+ ind*2*Na+ 2*ia] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).evaluate(rij,rik,costheta)
                                        #out[i,Nt*2*Nr+ ind*2*Na+ 2*ia + 1] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).derivative(rij,rik,costheta)
                                ind += 1  
        #print(time.time()-start)
        return out

