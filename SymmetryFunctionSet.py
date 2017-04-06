import SymmetryFunctions

class SymmetryFunctionSet(object):
    
    def __init__(self, atomtypes, cutoff = 7.):
        self.atomtypes = atomtypes
        self.cutoff = cutoff
        self.symmetry_functions = {}
        
    def add_radial_functions(self, etas):
        for a in self.atomtypes:
            for b in self.atomtypes:
                for eta in etas:
                    if not (a,b) in self.symmetry_functions:
                        self.symmetry_functions[(a,b)]= [lambda r: 
                            SymmetryFunctions.radial_function(r, eta, 
                                                              self.cutoff)]
                    else:
                        self.symmetry_functions[(a,b)].append(lambda r: 
                            SymmetryFunctions.radial_function(r, eta, 
                                                              self.cutoff))