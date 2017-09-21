import SymmetryFunctions as SFs
import numpy as _np
import ctypes as _ct
from scipy.spatial.distance import pdist, squareform
from scipy.misc import comb
lib = _ct.cdll.LoadLibrary("./symmetryFunctions/symmetryFunctionSet.so")

class SymmetryFunctionSet_new(object):
     def __init__(self):
         self.obj = lib.create_SymmetryFunctionSet()

     def test(self):
         lib.SymmetryFunctionSet_foo(self.obj)

     def add(self, a, b):
         lib.SymmetryFunctionSet_add.restype = _ct.c_double
         return lib.SymmetryFunctionSet_add(self.obj, _ct.c_double(a), _ct.c_double(b))

     def add_TwoBodySymmetryFunction(self, type1, type2, prms, funtype = "BehlerG2"):
         ptr = (_ct.c_double*len(prms))(*prms)
         lib.SymmetryFunctionSet_add_TwoBodySymmetryFunction(self.obj, _ct.c_int(type1), _ct.c_int(type2), funtype, ptr);

     def add_radial_function(self, rs, eta, cutoff):
         lib.SymmetryFunctionSet_add_radial_function(self.obj, _ct.c_double(rs), _ct.c_double(eta), _ct.c_double(cutoff))

     def add_radial_functions_evenly(self, N):
        rss = _np.linspace(0.,self.cutoff,N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        for rs, eta in zip(rss, etas):
            add_radial_function(self, rs, eta, self.cutoff)

     def add_angular_function(self, eta, zeta, lamb, cutoff):
         lib.SymmetryFunctionSet_add_angular_function(self.obj, _ct.c_double(eta), _ct.c_double(zeta), _ct.c_double(lamb), _ct.c_double(cutoff))

     def eval_geometry(self, geometry, derivative = False):
        N = len(geometry) # Number of atoms
        Nt = len(self.atomtypes) # Number of atomtypes
        Nr = len(self.radial_sym_funs) # Number of radial symmetry functions
        Na = len(self.angular_sym_funs) # Number of angular symmetry functions

        if derivative == False:
            out = _np.zeros((N, Nr*Nt + comb(Nt, 2, exact = True, repetition = True)*Na), dtype=_np.float)
            lib.SymmetryFunctionSet_eval_geometry(self.obj, out.ctypes)
        else:
            out = _np.zeros((N, 2*(Nr*Nt + comb(Nt, 2, exact = True, repetition = True)*Na)), dtype=_np.float)
            lib.SymmetryFunctionSet_eval_geometry(self.obj, out.ctypes)
        return out

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
                        SFs.RadialSymmetryFunction(rs, eta, self.cutoff))

    def add_angular_functions(self, etas, zetas, lambs):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    self.angular_sym_funs.append(
                            SFs.AngularSymmetryFunction(eta, zeta, lamb, self.cutoff))

    def add_angular_functions_new(self, etas, zetas, lambs, rss):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    for rs in rss:
                        self.angular_sym_funs.append(
                            SFs.AngularSymmetryFunction(eta, zeta, lamb, rs, self.cutoff))

    def add_radial_functions_evenly(self, N):
        rss = _np.linspace(0.,self.cutoff,N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        for rs, eta in zip(rss, etas):
            self.radial_sym_funs.append(
                        SFs.RadialSymmetryFunction(rs, eta, self.cutoff))

    def eval_geometry(self, geometry, derivative = False):
        # Returns a (Number of atoms) x (Size of G vector) matrix
        # The G vector doubles in size if derivatives are also requested
        # Calculate distance matrix. Should be solvable without using
        # squareform!
        # TODO: rewrite even more efficient
        # TODO: Implement derivative
        N = len(geometry) # Number of atoms
        Nt = len(self.atomtypes) # Number of atomtypes
        Nr = len(self.radial_sym_funs) # Number of radial symmetry functions
        Na = len(self.angular_sym_funs) # Number of angular symmetry functions

        dist_mat = squareform(pdist([g[1] for g in geometry]))
        # Needed for angular symmetry functions
        # maybe more elegant solution possible using transposition?
        rij = _np.tile(dist_mat.reshape((N,N,1)),(1,1,N))
        rik = _np.tile(dist_mat.reshape((N,1,N)),(1,N,1))
        rjk = _np.tile(dist_mat.reshape((1,N,N)),(N,1,1))
        costheta = (rij**2+rik**2-rjk**2)
        costheta[rij*rik > 0] = costheta[rij*rik > 0] / ((2*rij*rik)[rij*rik > 0])
	costheta[rij*rik == 0] = 0.0
        # (1-eye) to satify the j != i condition of the sum
        kron_ij = (1.-_np.eye(N))
        # Similar for the condition j != i, k != j in the angular sum
        dij = _np.tile(_np.eye(N).reshape((N,N,1)),(1,1,N))
        dik = _np.tile(_np.eye(N).reshape((N,1,N)),(1,N,1))
        djk = _np.tile(_np.eye(N).reshape((1,N,N)),(N,1,1))
        kron_ijk = 1. - _np.sign(dij+dik+djk)

        if derivative == False:
            out = _np.zeros((N, Nr*Nt + comb(Nt, 2, exact = True,
                                             repetition = True)*Na))

            ind = 0 # Counter for the combinations of angle types
            for t, atype in enumerate(self.atomtypes):
                # Mask for the different atom types
                mask = [a[0] == atype for a in geometry]
                for i, rad_fun in enumerate(self.radial_sym_funs):
                    out[:,t*Nr+i] = (kron_ij * rad_fun(dist_mat)).dot(mask)
                for atype2 in self.atomtypes[t:]:
                    # Second mask because two atom types are involved
                    mask2 = [a[0] == atype2 for a in geometry]
                    for j, ang_fun in enumerate(self.angular_sym_funs):
                        out[:,Nt*Nr+ind*Na+j] = (kron_ijk *
                            ang_fun(rij, rik, costheta)).dot(mask).dot(mask2)
                    ind += 1

        else: # derivative = True: doubles the size of the output matrix
            out = _np.zeros((N, 2*(Nr*Nt + comb(Nt, 2, exact = True,
                                             repetition = True)*Na)))

            ind = 0 # Counter for the combinations of angle types
            for t, atype in enumerate(self.atomtypes):
                # Mask for the different atom types
                mask = [a[0] == atype for a in geometry]
                for i, rad_fun in enumerate(self.radial_sym_funs):
                    out[:,t*2*Nr+2*i] = (kron_ij * rad_fun(dist_mat)).dot(mask)
                    out[:,t*2*Nr+2*i+1] = (kron_ij *
                                        rad_fun.derivative(dist_mat)).dot(mask)
                for atype2 in self.atomtypes[t:]:
                    # Second mask because two atom types are involved
                    mask2 = [a[0] == atype2 for a in geometry]
                    for j, ang_fun in enumerate(self.angular_sym_funs):
                        out[:,Nt*2*Nr+ind*2*Na+2*j] = (kron_ijk *
                            ang_fun(rij, rik, costheta)).dot(mask).dot(mask2)
                        out[:,Nt*2*Nr+ind*2*Na+2*j+1] = (kron_ijk *
                            ang_fun.derivative(rij, rik, costheta)).dot(mask).dot(mask2)
                    ind += 1
        return out
