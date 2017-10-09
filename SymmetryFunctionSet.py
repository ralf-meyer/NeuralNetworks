from os.path import dirname, abspath, join
from inspect import getsourcefile
from itertools import product, combinations_with_replacement
import SymmetryFunctions as SFs
import numpy as _np
import ctypes as _ct
from scipy.spatial.distance import pdist, squareform
from scipy.misc import comb

try:
    module_path = dirname(abspath(getsourcefile(lambda:0)))
    lib = _ct.cdll.LoadLibrary(join(module_path,"symmetryFunctions/libSymFunSet.so"))
    lib.SymmetryFunctionSet_add_TwoBodySymmetryFunction.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.POINTER(_ct.c_double), _ct.c_int, _ct.c_double)
    lib.SymmetryFunctionSet_add_ThreeBodySymmetryFunction.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.POINTER(_ct.c_double), _ct.c_int, _ct.c_double)
    lib.SymmetryFunctionSet_eval.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 1, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_eval_old.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 1, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_eval_derivatives.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_eval_derivatives_old.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_get_G_vector_size.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int))
except OSError as e:
    # Possibly switch to a python based implementation if loading the dll fails
    raise OSError(e.message)

class SymmetryFunctionSet(object):
    def __init__(self, atomtypes, cutoff = 7.0):
        self.cutoff = cutoff
        self.atomtypes = atomtypes
        self.type_dict = {}
        for i, t in enumerate(atomtypes):
            self.type_dict[t] = i
            self.type_dict[i] = i
        self.obj = lib.create_SymmetryFunctionSet(_ct.c_int(len(atomtypes)))

    def add_TwoBodySymmetryFunction(self, type1, type2, funtype, prms,
            cuttype, cutoff):
        ptr = (_ct.c_double*len(prms))(*prms)
        lib.SymmetryFunctionSet_add_TwoBodySymmetryFunction(self.obj,
            self.type_dict[type1], self.type_dict[type2], funtype, len(prms),
            ptr, cuttype, cutoff)

    def add_ThreeBodySymmetryFunction(self, type1, type2, type3, funtype, prms,
            cuttype, cutoff):
        ptr = (_ct.c_double*len(prms))(*prms)
        lib.SymmetryFunctionSet_add_ThreeBodySymmetryFunction(self.obj,
            self.type_dict[type1], self.type_dict[type2], self.type_dict[type3],
            funtype, len(prms), ptr, cuttype, cutoff)

    def add_radial_functions(self, rss, etas):
        for rs in rss:
            for eta in etas:
                for (ti, tj) in product(self.atomtypes, repeat = 2):
                    self.add_TwoBodySymmetryFunction(ti, tj, 0, [eta, rs],
                        1, self.cutoff)

    def add_radial_functions_evenly(self, N):
        rss = _np.linspace(0.,self.cutoff,N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        for rs, eta in zip(rss, etas):
            self.add_radial_functions([rs], [eta], self.cutoff)

    def add_angular_functions(self, etas, zetas, lambs):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    for ti in self.atomtypes:
                        for (tj, tk) in combinations_with_replacement(self.atomtypes, 2):
                            self.add_ThreeBodySymmetryFunction(
                                ti, tj, tk, 0, [lamb, zeta, eta], 1, self.cutoff)

    def print_symFuns(self):
        lib.SymmetryFunctionSet_print_symFuns(self.obj)

    def available_symFuns(self):
        lib.SymmetryFunctionSet_available_symFuns(self.obj)

    def eval(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(self.obj,
            len(types), types_ptr)
        out = _np.zeros(len_G_vector)
        lib.SymmetryFunctionSet_eval(self.obj, len(types), types_ptr, xyzs, out)
        return out

    def eval_old(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(self.obj,
            len(types), types_ptr)
        out = _np.zeros(len_G_vector)
        lib.SymmetryFunctionSet_eval_old(
            self.obj, len(types), types_ptr, xyzs, out)
        return out

    def eval_geometry(self, geo):
        types = [a[0] for a in geo]
        xyzs = _np.array([a[1] for a in geo])
        return self.eval(types, xyzs)

    def eval_geometry_old(self, geo):
        types = [a[0] for a in geo]
        xyzs = _np.array([a[1] for a in geo])
        return self.eval_old(types, xyzs)

    def eval_derivatives(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(
            self.obj, len(types), types_ptr)
        out = _np.zeros((len_G_vector, 3*len(types)))
        lib.SymmetryFunctionSet_eval_derivatives(
            self.obj, len(types), types_ptr, xyzs, out)
        return out

    def eval_derivatives_old(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(
            self.obj, len(types), types_ptr)
        out = _np.zeros((len_G_vector, 3*len(types)))
        lib.SymmetryFunctionSet_eval_derivatives_old(
            self.obj, len(types), types_ptr, xyzs, out)
        return out

class SymmetryFunctionSet_py(object):

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
                        if (atype == atype2):
                            out[:,Nt*Nr+ind*Na+j] = 0.5*(kron_ijk *
                                ang_fun(rij, rik, costheta)).dot(mask).dot(mask2)
                        else:
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
