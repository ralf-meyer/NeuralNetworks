import numpy as _np

def cutoff_function(r,cut):
    # Weird formulation allows for calls using vectors
    return (r <= cut)*0.5*(_np.cos(_np.pi*r/cut)+1)

def radial_function(r, rs, eta, cut):
    return _np.exp(-eta*(r-rs)**2)*cutoff_function(r,cut)
    
def angular_function(r, zeta, cut):
    pass