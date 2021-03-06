import numpy as _np
from scipy.optimize import approx_fprime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class NNForce(object):


    def __init__(self,Net,size, dim=3, m=None, Consts=1.0):
        self.__dim = dim
        self.__size = size
        self.__A = _np.zeros((size, dim))
        self.__Epot = 0
        self.__Etot = 0
        self.__Fm = _np.zeros( ( size , size ) )
        self.__V = _np.zeros( ( size , size ) )
        self.__D = _np.zeros( ( size , size ) )
        self.__M = _np.zeros( ( size , size ) )
        self.Net=Net
        self.types=[]


    def set_masses(self, m):
        """
        Set the masses used for computing the forces.
        """
        self.__M[:, :] = m

    def to_nn_input(self,x):
        """Converts the raw geometry into a
        ("type",np.array(geometry)) geometry"""

        nn_geom=[]
        reshaped_x=_np.asarray(x).reshape(self.__size,3)
        for i,type in enumerate(self.types):
            nn_geom.append((type,reshaped_x[i]))

        return nn_geom

    def fun(self,x):
        return self.Net.energy_for_geometry(self.to_nn_input(x))

    def update_force(self,pset):
        """Evaluates the force and calulates the acceleration
        for a given PyParticles particle set"""

        coordinates=pset.X
        self.types=list(pset.label[:])
        geometry=[]
        for i in range(len(pset.label)):
            geometry.append((pset.label[i],_np.asarray(coordinates[i])))

        e_temp,forces=self.Net.energy_and_force_for_geometry(geometry)
        self.__Epot= e_temp/len(pset.label[:])
        abs_v = _np.linalg.norm(pset.V, axis=1) / pset.unit
        self.__Etot=self.__Epot+_np.sum(abs_v**2*(pset.M[:]/pset.mass_unit)/2)*6.242e18
        self.__Etot=self.__Etot/len(pset.label[:])
        x=_np.asarray(pset.X[:]).flatten()
        #grad=approx_fprime(x,self.fun,1e-7)
        #forces=-_np.asarray(grad)
        forces=forces.reshape((len(pset.label),3))
        self.__A=(forces[:]/(pset.M[:]/pset.mass_unit))*1.6021766208-19

        return self.__A

    def getA(self):
        """
        Return the currents accelerations of the particles
        """
        return self.__A

    A = property(getA, doc="Return the current accelerations of the particles (getter only)")

    def getEpot(self):
        """
        Return the currents energy of the particles
        """
        return self.__Epot

    Epot = property(getEpot, doc="Return the current potential energy of the particles (getter only)")

    def getEtot(self):
        """
        Return the currents energy of the particles
        """
        return self.__Etot

    Etot = property(getEtot, doc="Return the current total energy of the particles (getter only)")

    def getF(self):
        """
        Return the currents forces on the particles
        """
        return (self.__A.T * self.__M[:, 0]).T

    F = property(getF, doc="Return the current forces on the particles (getter only)")


