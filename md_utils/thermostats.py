import numpy as _np



def get_temperature(pset):

    kb=1.38064852e-23
    abs_v=_np.linalg.norm(pset.V,axis=1)/pset.unit
    T=_np.mean(abs_v**2*pset.M[:]/pset.mass_unit)/(3*kb)

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

