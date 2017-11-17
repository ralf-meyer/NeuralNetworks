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


import numpy as np
import sys

from NeuralNetworks.md_utils.ode import sim_time as st
from mpl_toolkits.mplot3d import Axes3D
import NeuralNetworks.md_utils.thermostats as thermo
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as _np
import time as _time


class OdeSolver(object) :
    """
    Base abstract class for defining the integration method of the ordinary differential equation like Runge Kutta or Euler method,
    the user must overide the method **__step__**
    
    Example (Euler method):
    ::
    
        import numpy as np
        import pyparticles.ode.ode_solver as os
       
        class EulerSolver( os.OdeSolver ) :
           def __init__( self , force , p_set , dt ):
               super(EulerSolver,self).__init__( force , p_set , dt )
           
        def __step__( self , dt ):
   
           self.force.update_force( self.pset )
           
           self.pset.V[:] = self.pset.V + self.force.A * dt
           self.pset.X[:] = self.pset.X + self.pset.V * dt
           
           self.pset.update_boundary() 
    
    Constructor:
    
    :param force:      the force model
    :param p_set:      the particle set
    :param dt:         delta time
    """
    def __init__( self , force , p_set , dt, log_file_path = "" ):
        self.__force = force
        self.__p_set = p_set
        self.__dt = dt
        
        self.__sim_time = st.SimTime( self )
        self.logger=Logger(log_file_path)
        self.__steps_cnt = 0
        self.all_forces=[]
        self.all_epot=[]
        self.all_etot=[]
        self.all_temp=[]
        self.steps=1000
        self.plot = True
        self.fig = None
        self.ax1 = None
        self.scat = None
        self.eplot=None


    def init_plot(self, x):
        """Initialize the animation"""
        self.fig = plt.figure(figsize=plt.figaspect(2.))
        self.ax1 = self.fig.add_subplot(211, projection='3d')
        self.ax2 = self.fig.add_subplot(212)

        self.setup_plot()


    def setup_plot(self):
        """Sets up the plots"""
        x=self.__p_set.X[:]
        self.scat = self.ax1.scatter(x[:, 0],
                                    x[:, 1],
                                    x[:, 2],
                                    animated=False, marker='o', alpha=1, s=50,c='b')
        plt_epot = np.asarray(self.all_epot).flatten()
        plt_etot = np.asarray(self.all_etot).flatten()
        self.epot_plot,=self.ax2.plot(np.arange(1,len(plt_epot)+1),plt_epot,c='b',label="E_pot")
        self.etot_plot, = self.ax2.plot(np.arange(1, len(plt_etot) + 1), plt_etot, c='r',label="E_tot")
        plt.legend(handles=[self.epot_plot,self.etot_plot],loc=1)
        plt.show(block=False)

    def update_plot(self,i):
        """Update the plots."""
        self.scat._offsets3d = (_np.ma.ravel(self.pset.X[:, 0]),
                                _np.ma.ravel(self.pset.X[:, 1]),
                                _np.ma.ravel(self.pset.X[:, 2])
                                )
        plt_epot = np.asarray(self.all_epot).flatten()
        plt_etot = np.asarray(self.all_etot).flatten()

        self.epot_plot,=self.ax2.plot(np.arange(1,len(plt_epot)+1),plt_epot,c='b')
        self.etot_plot, = self.ax2.plot(np.arange(1, len(plt_etot) + 1), plt_etot, c='r')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def start(self):
        if self.fig == None:
            self.init_plot(self.pset.X[:])

        for i in range(self.steps):
            if self.plot:
                self.update_plot(i)
            self.step(self.dt)

    
    def get_dt( self ):
        return self.__dt

    def set_dt( self , dt ):
        self.__dt = dt

    dt = property( get_dt , set_dt , doc="get and set the delta time of the step")
    
    
    def get_steps(self):
        return self.__steps_cnt
    
    def del_steps(self):
        self.__steps_cnt = 0
    
    steps_cnt = property( get_steps , fdel=del_steps , doc="return the count of the performed steps")
    
    
    def get_time(self):
        return self.__sim_time.time
    
    def set_time( self , t0 ):
        self.__sim_time.time = t0
        
    time = property( get_time , set_time , doc="get and set the current simulation time" )
    
    
    def get_sim_time( self ):
        return self.__sim_time
    
    sim_time = property( get_sim_time , doc="get the reference to the SimTime object, used for storing and sharing the current simulation time" )
    
    
    def get_force( self ):
        return self.__force
    
    def set_force( self , force ):
        self.__force = force
        
    force = property( get_force , set_force , doc="get and set the used force model")
    
    
    def update_force( self ):
        self.__force.update_force( self.pset )
    
    def get_pset( self ):
        return self.__p_set
    
    def set_pset( self , p_set ):
        self.__p_set = p_set
        
    pset = property( get_pset , set_pset , "get and set the used particles set")
    
    
    def step( self , dt=None ):
        """
        Perform an integration step. If the dt is not given (reccomended) it uses the stored *dt*.
        You must alway use this method for executing a step.
        """
        if dt == None:
            dt = self.dt

        self.__sim_time.time += dt
        self.__steps_cnt += 1
        self.__step__( dt )

        self.all_epot.append(self.force.Epot)
        self.all_etot.append(self.force.Etot)
        self.all_forces.append(self.force.F)
        self.all_temp.append(thermo.get_temperature(self.pset))

        self.logger.log(self)

        
    def __step__( self , dt ):
        """
        Abstract methos that contain the code for computing the new status of the particles system. This methos must be overidden by the user.
        """
        NotImplementedError(" %s : is virtual and must be overridden." % sys._getframe().f_code.co_name )

class Logger(object):

    def __init__(self, log_file_path="./mdrun.log", step_index=0):

        if not log_file_path:
            log_file_path = "./mdrun.log"

        self._log_file = open(log_file_path, 'a')

        self._step_index = step_index


    def log(self, solver):
        from os import linesep

        step = self._step_index
        time = solver.sim_time.time

        species = ["??" for i in range(solver.pset.size)]
        positions = solver.pset.X
        velocities = solver.pset.V
        forces = solver.force.F

        E_tot = solver.force.Etot
        E_pot = solver.force.Epot

        temperature = thermo.get_temperature(solver.pset)

        # write system properties
        log_str = "Step: " + str(step) + linesep
        log_str += "Time: " + str(time) + linesep
        log_str += "Total Energy: " + str(E_tot) + linesep
        log_str += "Potential Energy: " + str(E_pot) + linesep
        log_str += "Temperature: " + str(temperature) + linesep

        #write header for particle properties
        log_str += self._atmic_data_to_string(
            "Species",
            ["x","y", "z"],
            ["v_x", "v_y", "v_z"],
            ["f_x", "f_y", "f_z"]
        ) + linesep

        # wirte particle properties
        for i in range(solver.pset.size):
            log_str += self._atmic_data_to_string(
                species[i],
                positions[i],
                velocities[i],
                forces[i]
            ) + linesep

        self._log_file.write(log_str)

    def _atmic_data_to_string(self, species, postions, velocities, forces):
        log_str = species + " "
        log_str += " ".join(map(str, postions)) + " "
        log_str += " ".join(map(str, velocities)) + " "
        log_str += " ".join(map(str, forces))

        return log_str







