import SymmetryFunctions as SFs
import numpy as _np
from scipy.spatial.distance import pdist, squareform

class SymmetryFunctionSet(object):
    
    def __init__(self, atomtypes, cutoff = 7.):
        self.atomtypes = atomtypes
        self.cutoff = cutoff
        self.symmetry_functions = {}
        
    def add_radial_functions(self, rss, etas):
        for a in self.atomtypes:
            for b in self.atomtypes:
                if not (a,b) in self.symmetry_functions:
                    self.symmetry_functions[(a,b)]= [
                        SFs.RadialSymmetryFunction(rs, eta, self.cutoff)
                        for rs, eta in zip(rss, etas)]
                else:
                    self.symmetry_functions[(a,b)].extend([
                        SFs.RadialSymmetryFunction(rs, eta, self.cutoff)
                        for rs, eta in zip(rss, etas)])
                                                              
    def add_angluar_functions(self, etas, zetas, lambs):
        for a in self.atomtypes:
            for b in self.atomtypes:
                for c in self.atomtypes:
                    if not (a,b,c) in self.symmetry_functions:
                        self.symmetry_functions[(a,b,c)]= [
                            SFs.AngularSymmetryFunction(eta, zeta, lamb, self.cutoff)
                            for eta, zeta, lamb in zip(etas, zetas, lambs)]
                    else:
                        self.symmetry_functions[(a,b,c)].extend([
                            SFs.AngularSymmetryFunction(eta, zeta, lamb, self.cutoff)
                            for eta, zeta, lamb in zip(etas, zetas, lambs)])
    
    def add_radial_functions_evenly(self, N):
        rss = _np.linspace(0.,self.cutoff,N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        self.add_radial_functions(rss, etas)
    
    def eval_geometry(self, geometry):
        out = []        
        for i in geometry:
            i_type = i[0]      
            vals = {}
            for j in geometry:
                j_type = j[0]
                key = (i_type, j_type)
                if key in self.symmetry_functions:
                    if not key in vals:
                        vals[key] = _np.zeros(len(self.symmetry_functions[key]))        
                    if not i is j:
                        rij = _np.linalg.norm(i[1]-j[1])
                        vals[key] += _np.fromiter(map(lambda f: f(rij), self.symmetry_functions[key]),dtype=_np.float)
                    for k in geometry:
                        k_type = k[0]
                        key2 = (i_type, j_type, k_type)
                        if key2 in self.symmetry_functions:
                            if not key2 in vals:
                                vals[key2] = _np.zeros(len(self.symmetry_functions[key2]))
                            if not ((i is j) or (i is k) or (j is k)):
                                rij = _np.linalg.norm(i[1]-j[1])
                                rik = _np.linalg.norm(i[1]-k[1])
                                costheta = _np.dot(i[1]-j[1],i[1]-k[1])/(rij*rik)
                                vals[key2] += _np.fromiter(map(lambda f: f(rij, rik, costheta),
                                    self.symmetry_functions[key2]),dtype=_np.float)
            #out.append(vals)            
            out.append(_np.hstack(vals.values()))
        return out      
        
    def eval_geometry_new(self, geometry):
        # Calculate distance matrix. Should be solvable without using 
        # squareform! TODO: rewrite even more efficient
        dist_mat = squareform(pdist([g[1] for g in geometry]))