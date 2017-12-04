import numpy as _np
import itertools
from scipy.constants import k as k_B
from scipy.constants import pi


def get_temperature(pset):

    abs_v = _np.linalg.norm(pset.V,axis=1) / pset.unit
    T = _np.mean(abs_v**2 * pset.M[:] / pset.mass_unit) / (3 * k_B)

    return T

def set_temperature(pset):
    m = pset.masses

    # calculate for each kind of mass
    for m_i in list(set(m)):
        # get indices of particles with mass m_i
        indices = [j for j, m_j in enumerate(m) if m_j == m_i ]
        v = Sampler.draw_boltzman_velocities(T, m_i)
        pset.V[indices] = v
    
    return pset

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

class Sampler(object):
    """will draw samples from given distributions"""

    def __init__(self):
        #all masses in u!
        self._mass_conversion = 1

    def _maxwell_boltzman(self, v, T, m):
        a = _np.sqrt(k_B * T / (m * self._mass_conversion))

        return _np.sqrt(2 / pi) * (v**2 * _np.exp(-v**2/(2 * a**2))) / a**3

    def _normal(self, x, mu, sigma):
        """normal distribution"""
        return  _np.exp(-(x - mu)**2 / (2 * sigma**2)) / _np.sqrt(2 * pi * sigma**2)

    def draw_boltzman_scalars(self, nsamples, T, m):
        """Draw maxwell-boltzman distributed for temperature T (in K) and 
        with masses m in atomic masses via rejection method."""

        result = []

        # scaling (not really used)
        c = 1

        # use expectation and variance of boltzman to make sure normal is always higher
        a = _np.sqrt(k_B * T / (m * self._mass_conversion))
        mu = 2 * a * _np.sqrt(2 / pi)
        sigma = _np.sqrt(a**2 * (3 * pi - 8) / pi)

        # number of trials to be done at once
        n_trial = nsamples

        while len(result) < nsamples:
            
            # draw trials
            x = _np.random.normal(mu, sigma, n_trial)

            # reject/accept
            lhs = _np.random.rand(*x.shape) * c * self._normal(x, mu, sigma)
            rhs = self._maxwell_boltzman(x, T, m)
            accepted = lhs < rhs

            # log and see how many are missing
            result += list(x[accepted])
            n_trial = nsamples - len(result)

        return _np.array(result[:nsamples])
        
    def draw_boltzman_velocities(self, nsamples, T, m):
        """Draw maxwell-boltzman distributed velocites (3-Vectors) for temperature T (in K) and 
        with masses m in atomic masses via rejection method."""

        # draw absolute values
        v = self.draw_boltzman_scalars(nsamples, T, m)

        # draw directions in spherical coordinates (uniform distributed)
        phi = _np.random.rand(nsamples) * 2 * pi
        theta = _np.random.rand(nsamples) * pi
        e = _np.array([
            _np.cos(phi) * _np.sin(theta),
            _np.sin(phi) * _np.sin(theta),
            _np.cos(theta)
            ]).reshape((nsamples, 3))

            #todo: possibly reshape
        # subtract average
        e -= _np.mean(e, 0)

        return v * e
        

