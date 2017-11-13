import numpy as _np



def get_temperature(pset):

    kb=1.38064852e-23
    abs_v=_np.linalg.norm(pset.V,axis=1)/pset.unit
    T=_np.mean(abs_v**2*pset.M[:]/pset.mass_unit)/(3*kb)

    return T

class BerensdenNVT(object):

    def __init__(self,pset):
        self.pset=pset
        self.deltat=0
        self.t_thermo=0
        self.hb_temp=0

    def get_lambda(self):

        lamb = _np.sqrt(1 + self.deltat / self.t_thermo * (self.hb_temp / get_temperature(self.pset) - 1))

        #hard borders
        if lamb< 0.9:
            lamb = 0.9

        if  lamb > 1e5:
            lamb = 1e5

        return lamb

