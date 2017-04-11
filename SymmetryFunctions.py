import numpy as _np

def cutoff_function(r,cut):
    # Weird formulation allows for calls using vectors
    return (r <= cut)*0.5*(_np.cos(_np.pi*r/cut)+1)
    
class RadialSymmetryFunction(object):
    def __init__(self, rs, eta, cut):
        self.rs = rs
        self.eta = eta
        self.cut = cut
        
    def __call__(self, r):
        return _np.exp(-self.eta*(r-self.rs)**2)*cutoff_function(r,self.cut)

class AngularSymmetryFunction(object):
    def __init__(self, eta, zeta, lamb, cut):
        self.eta = eta
        self.zeta = zeta
        self.lamb = lamb
        self.cut = cut
        
    def __call__(self, rij, rik, costheta):
        return 2**(1-self.zeta)* ((1 + self.lamb*costheta)**self.zeta * 
            _np.exp(-self.eta*(rij**2+rik**2)) * cutoff_function(rij,self.cut)*
            cutoff_function(rik,self.cut))

def radial_function(rij, rs, eta, cut):
    return _np.exp(-eta*(rij-rs)**2)*cutoff_function(rij,cut)
    
def angular_function(rij, rik, costheta, eta, zeta, lamb,  cut):
    return 2**(1-zeta)* ((1 + lamb*costheta) * _np.exp(-eta*(rij**2+rik**2)) *
        cutoff_function(rij,cut)*cutoff_function(rik,cut))