import numpy as _np
from scipy.constants import k as k_B


def get_temperature(pset):

    abs_v=_np.linalg.norm(pset.V,axis=1)/pset.unit
    T=_np.mean(abs_v**2*pset.M[:]/pset.mass_unit)/(3*k_B)

    return T

class BerensdenNVT(object):

    def __init__(self,pset,deltat,t_thermo,hb_temp):
        self.pset=pset
        self.deltat=deltat
        self.t_thermo=t_thermo
        self.hb_temp=hb_temp
        self.temperature=0

    def get_lambda(self):

        self.temperature = get_temperature(self.pset)
        lamb = _np.sqrt(1 + self.deltat / self.t_thermo * (self.hb_temp /self.temperature  - 1))

        #hard borders
        if lamb< 0.9:
            lamb = 0.9

        if  lamb > 1.1:
            lamb = 1.1

        return lamb

