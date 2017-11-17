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

class LeapfrogSolverBerendsen( os.OdeSolver ) :
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

