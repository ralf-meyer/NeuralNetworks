import numpy as _np

def cutoff_function(r,cut):
    if r <= cut:
        return 0.5*(_np.cos(_np.pi*r/cut)+1)
    else:
        return 0.0
