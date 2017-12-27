# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as _np
from NeuralNetworks.md_utils.ode import ode_solver as os
from NeuralNetworks.md_utils import thermostats
from scipy.constants import k as k_B

class LeapfrogSolverBerendsen( os.OdeSolver ) :
    """acutally velocity-verlet"""
    def __init__( self , force , p_set , dt ):
        super(LeapfrogSolverBerendsen,self).__init__( force , p_set , dt )
        self.__Ai = _np.zeros( self.force.A.shape )
        self.__thermo = os.thermo.BerensdenNVT(p_set, dt, p_set.thermostat_coupling_time,
                                                 p_set.thermostat_temperature)


    def __step__( self , dt ):
        self.pset.X[:] = self.pset.X + self.pset.V * dt + 0.5*self.force.A * dt**2.0
        self.__Ai[:] = self.force.A
        self.force.update_force( self.pset )
        lamb = self.__thermo.get_lambda()
        self.pset.V[:] = self.pset.V* lamb + 0.5 * ( self.__Ai + self.force.A ) * dt
        self.pset.update_boundary() 




class LeapfrogSolverLangevin(os.OdeSolver):
    """A Velocity-Verlet ode Solver with Langevin 'thermostat'.
    """
    def __init__(self, force, p_set, dt, gamma):
        """
        Ctor

        Args:\n
            force: the force model the integrator works on \n
            p_set: the particle set the integrator works on \n
            dt: the time step (in seconds) \n
            gamma: the damping constant (in mass units of the p_set)
        """

        self._gamma = gamma #/ p_set.mass_unit
        self._temperature = p_set.thermostat_temperature

        super(LeapfrogSolverLangevin, self).__init__(force, p_set, dt)

        # used as buffer for old acceleration (Langevin modified)
        self._save_force = _np.zeros(self.force.A.shape)

        # boltzman constant in units of the the pset
        # J=kg*m**2/s**2 -> u*Angstroem/s**2
        self.k = k_B * (p_set.mass_unit * p_set.unit**2)

    def langevin_force(self, force, p_set):
        """The Lagevin version of the force model: keeps the system at
        Temperature T.

        m D[r,{t,2}] = - gamma D[r,{t,1}] + Sqrt[2*gamma*k_B*T]*R(t) + force,
        with D[x,{y,z}] the z-fold partial derivative of x w/ respect to y,
        gamma a friction coefficient,
        k_B the Boltzman constant,
        R(t) a random vector of the size of the force (centered around zero,
            variance about 1),  <R(t)> = 0.
        and force the original force model.

        See also: https://en.wikipedia.org/wiki/Langevin_dynamics
        """

        # apply random force normally distributed around 0, var= 2gammak_BT
        F_random = _np.random.normal(
            0,
            _np.sqrt(2 * self._gamma * p_set.M * self.k * self._temperature),
            force.shape
        )

        F_friction = - self._gamma * p_set.V * p_set.M 

        return force + F_random + F_friction



    def __step__(self, dt):
        """
        Advances system in phase space via velocity-verlet
        r(t + dt) = r(r) + dt * v(t) + dt^2/2*f(r(t),t)
        v(t + dt) = v(t) + dt/2 * (f(r(t),t) + f(r(t + dt), t + dt))
        Args:
            dt: time step

        Returns:
            Nothing
        """

        # store current foce (Langevin manipulated)
        old_acceleration = self._save_force / self.pset.M[:]

        # calculate r(t + dt)
        self.pset.X[:] = \
            self.pset.X + self.pset.V * dt + 0.5 * old_acceleration * dt ** 2.0



        # calculate f(r(t+dt), t + dt)
        self.force.update_force(self.pset)

        # calculate langevin force F(r(r+dt), t+dt), save for next step and
        # calc new accellarations
        self._save_force  = self.langevin_force(self.force.A * self.pset.M, self.pset) 
        new_acceleration = self._save_force / self.pset.M[:]

        # calculate v(t + dt)
        self.pset.V[:] = \
            self.pset.V + 0.5 * (
                old_acceleration +
                new_acceleration
            ) * dt

        #print(self.pset.V[:])

        self.pset.update_boundary()