import numpy as _np


class NNForce(object):


    def __init__(self,Net,size, dim=3, m=None, Consts=1.0):
        self.__dim = dim
        self.__size = size
        self.__A = _np.zeros((size, dim))
        self.__Fm = _np.zeros( ( size , size ) )
        self.__V = _np.zeros( ( size , size ) )
        self.__D = _np.zeros( ( size , size ) )
        self.__M = _np.zeros( ( size , size ) )
        self.Net=Net


    def set_masses(self, m):
        """
        Set the masses used for computing the forces.
        """
        self.__M[:, :] = m

    def update_force(self,pset):
        """Evaluates the force and calulates the accelration
        for a given PyParticels particle set"""
        coordinates=pset.X
        geometry=[]
        for i in range(len(pset.label)):
            geometry.append((pset.label[i],_np.asarray(coordinates[i])))

        energy,forces=self.Net.energy_and_force_for_geometry(geometry)

        forces=_np.asarray(forces)
        forces=forces.reshape((len(pset.label),3))

        self.__A=(forces[:]/pset.M[:])*1e28/1.66

        return self.__A

    def getA(self):
        """
        Return the currents accelerations of the particles
        """
        return self.__A

    A = property(getA, doc="Return the currents accelerations of the particles (getter only)")

    def getF(self):
        """
        Return the currents forces on the particles
        """
        return (self.__A.T * self.__M[:, 0]).T

    F = property(getF, doc="Return the currents forces on the particles (getter only)")
