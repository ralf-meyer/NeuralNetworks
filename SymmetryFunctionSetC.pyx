#!python
#cython: boundscheck=False, wraparound=False, cdivision=True 
from SymmetryFunctionsC cimport RadialSymmetryFunction, AngularSymmetryFunction
import numpy as _np 
from scipy.misc import comb

cdef extern from "math.h":
    double sqrt(double m)

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
        
    def eval_geometry(self, geometry, derivative = False):
        # Returns a (Number of atoms) x (Size of G vector) matrix
        # The G vector doubles in size if derivatives are also requested
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
                                out[i,itj*Nr+ir] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).evaluate(rij) 
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
                                        out[i,Nt*Nr+ ind*Na+ ia] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).evaluate(rij,rik,costheta)
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
                                out[i,itj*2*Nr+2*ir] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).evaluate(rij) 
                                out[i,itj*2*Nr+2*ir+1] += (<RadialSymmetryFunction>self.radial_sym_funs[ir]).derivative(rij) 
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
                                        out[i,Nt*2*Nr+ ind*2*Na+ 2*ia] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).evaluate(rij,rik,costheta)
                                        out[i,Nt*2*Nr+ ind*2*Na+ 2*ia + 1] += (<AngularSymmetryFunction>self.angular_sym_funs[ia]).derivative(rij,rik,costheta)
                                ind += 1  
        return out
