import SymmetryFunctions as SFs
import numpy as _np

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
                        for eta in etas for rs in rss]
                else:
                    self.symmetry_functions[(a,b)].extend([
                        SFs.RadialSymmetryFunction(rs, eta, self.cutoff)
                        for eta in etas for rs in rss])
                                                              
    def add_angluar_functions(self, etas, zetas, lambs):
        for a in self.atomtypes:
            for b in self.atomtypes:
                for c in self.atomtypes:
                    if not (a,b,c) in self.symmetry_functions:
                        self.symmetry_functions[(a,b,c)]= [
                            SFs.AngularSymmetryFunction(eta, zeta, lamb, self.cutoff)
                            for eta in etas for zeta in zetas for lamb in lambs]
                    else:
                        self.symmetry_functions[(a,b,c)].extend([
                            SFs.AngularSymmetryFunction(eta, zeta, lamb, self.cutoff)
                            for eta in etas for zeta in zetas for lamb in lambs])
    
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
                        vals[key] += map(lambda f: f(rij), self.symmetry_functions[key])
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
                                vals[key2] += map(lambda f: f(rij, rik, costheta), 
                                    self.symmetry_functions[key2])
            #out.append(vals)            
            out.append(_np.array(vals.values()).flatten())
        return out        