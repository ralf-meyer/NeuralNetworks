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
from copy import deepcopy
from NeuralNetworks.md_utils.ode import sim_time as st
from mpl_toolkits.mplot3d import Axes3D
import NeuralNetworks.md_utils.thermostats as thermo
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as _np
import time as _time
from os import linesep
import os as _os




class OdeSolver(object) :
    """
    ToDO: this should be replaced by a propper doc string with the new funcitonalities!!
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
    def __init__( self , force , p_set , dt):
        self.__force = force
        self.__p_set = p_set
        self.__dt = dt
        self.types=[]
        self.__sim_time = st.SimTime( self )
        self.__steps_cnt = 0
        self.all_geometries=[]
        self.all_forces=[]
        self.all_epot=[]
        self.all_etot=[]
        self.all_temp=[]
        self.steps=1000
        self.plot = True # if no live plotting is desired, this should be set false
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.scat = None
        self.eplot=None
        self.save_png=False
        self.png_path=""
        self.save_data=True
        self.data_path=""
        self.counter=0
        self.plot_steps=100

        # Instatiate a logger (but deactivate logging by default)
        #TODO: log file paths should not be speciefied in ctor of solver
        self._logger=Logger("", "")
        self._enable_full_log = ""
        self._enable_xyz_log = ""

    def enable_full_log(self, path="./mdrun.log"):
        """Activates logging of positions, forces, etc. log file in path
        To deactivate set path to empty string."""
        self._logger.full_log_file = path

    def enable_xyz_log(self, path="./mdrun.xyz"):
        """Activates logging to xyz file in path. To deactivate set path to
        empty string."""
        self._logger.xyz_log_file = path

    def init_plot(self, x):
        """Initialize the animation"""
        self.fig = plt.figure(figsize=plt.figaspect(3))
        self.ax1 = self.fig.add_subplot(311, projection='3d')
        self.ax1.set_xlim(-10,10)
        self.ax1.set_ylim(-10,10)
        self.ax1.set_zlim(-10,10)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        if self.save_png:
            if not _os.path.exists(self.png_path):
                _os.makedirs(self.png_path)

            self.fig.savefig(_os.path.join(self.png_path,"pic_"+str(self.counter)))
        self.counter+=1
        self.setup_plot()


    def setup_plot(self):
        """Sets up the plots"""
        x=self.__p_set.X[:]
        # Get coloring for atoms
        my_colors=[]
        template="byrcmgkw"
        for atom_type in self.pset.label:
            if atom_type not in self.types:
                self.types.append(atom_type)

        for i,atom_type in enumerate(self.pset.label):

            if atom_type == "Ni":
                my_colors.append('b')
            elif atom_type == "Au":
                my_colors.append('y')
            else:
                my_colors.append('k')

        self.scat = self.ax1.scatter(x[:, 0],
                                    x[:, 1],
                                    x[:, 2],
                                    animated=False, marker='o', alpha=0.8, s=150,c=my_colors)
        t=np.arange(0,len(self.all_epot))*self.__dt
        plt_epot = np.asarray(self.all_epot).flatten()
        plt_etot = np.asarray(self.all_etot).flatten()
        plt_temp = np.asarray(self.all_temp).flatten()

        self.epot_plot,=self.ax2.plot(t,plt_epot,c='C0')
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("y")
        self.ax1.set_zlabel("z")
        self.etot_plot, = self.ax2.plot(t, plt_etot, c='C1')
        self.ax2.set_xlabel("t / s")
        self.ax2.set_ylabel("E / eV")
        self.temp_plot, = self.ax3.plot(t, plt_temp, c='C2',label="Temperature /K")
        self.ax3.set_xlabel("t / s")
        self.ax3.set_ylabel("T / K")
        self.fig.legend(handles=[self.scat,self.epot_plot,self.etot_plot], labels=["3D-view of system","Potential energy/atom eV", "Total energy/atom /ev"], loc=1)
        plt.legend(handles=[self.temp_plot], loc=1)
        plt.show(block=False)

    def update_plot(self,i):
        """Update the plots."""
        x = _np.ma.ravel(self.pset.X[:, 0]) - np.mean(self.pset.X[:, 0])
        y = _np.ma.ravel(self.pset.X[:, 1]) - np.mean(self.pset.X[:, 1])
        z = _np.ma.ravel(self.pset.X[:, 2]) - np.mean(self.pset.X[:, 2])
        self.scat._offsets3d = (x,y,z)
        t = np.arange(0, len(self.all_epot)) * self.__dt
        plt_epot = np.asarray(self.all_epot).flatten()

        plt_etot = np.asarray(self.all_etot).flatten()
        plt_temp = np.asarray(self.all_temp).flatten()
        self.epot_plot,=self.ax2.plot(t,plt_epot,c='C0')
        self.etot_plot, = self.ax2.plot(t, plt_etot, c='C1')
        self.temp_plot, = self.ax3.plot(t, plt_temp, c='C2', label="Temperature /K")

        self.counter+=1
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def start(self):

        if self.fig == None and self.plot:
            self.init_plot(self.pset.X[:])

        for i in range(self.steps):
            if self.plot and i%self.plot_steps==0:
                self.update_plot(i)
            self.step(self.dt)
            print("Step "+str(i+1))

        if self.save_data:
            if self.data_path=="":
                self.data_path = _os.path.join(_os.getcwd(),"data.npy")
            if not _os.path.exists(self.data_path):
                _os.makedirs(self.data_path)
            nrfiles=len([name for name in _os.listdir(self.data_path) if _os.path.isfile(name)])
            filename="Data_"+str(nrfiles+1)+".npy"
            np.save(_os.path.join(self.data_path,filename),
                    np.asarray([self.all_epot, self.all_etot, self.all_temp]))
            print("Data saved to "+str(self.data_path))

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

        self.all_geometries.append(deepcopy(self.pset.X))
        if hasattr(self.force, "Epot"):
            self.all_epot.append(deepcopy(self.force.Epot))
        if hasattr(self.force, "Etot"):
            self.all_etot.append(deepcopy(self.force.Etot))
        if hasattr(self.force, "F"):
            self.all_forces.append(deepcopy(self.force.F))
        self.all_temp.append(deepcopy(thermo.get_temperature(self.pset)))

        # will log current positons/forces etc. if for activated logging types
        self._logger.log(self)


    def __step__( self , dt ):
        """
        Abstract methos that contain the code for computing the new status of the particles system. This methos must be overidden by the user.
        """
        NotImplementedError(" %s : is virtual and must be overridden." % sys._getframe().f_code.co_name )

    def update_animation(self, num):
        self.ax1.relim()
        self.ax1.autoscale_view()
        geometry = self.all_geometries[num]
        x = _np.ma.ravel(geometry[:, 0]) - np.mean(geometry[:, 0])
        y = _np.ma.ravel(geometry[:, 1]) - np.mean(geometry[:, 1])
        z = _np.ma.ravel(geometry[:, 2]) - np.mean(geometry[:, 2])
        self.scat._offsets3d = (x,y,z)
        if self.save_png:
            self.fig.savefig(_os.path.join(self.png_path, "pic_" + str(num)))
        return self.scat,

    def animate_run(self):

        plt.ioff()
        # Get coloring for atoms
        my_colors=[]
        template="byrcmgkw"
        for atom_type in self.pset.label:
            if atom_type not in self.types:
                self.types.append(atom_type)

        for i,atom_type in enumerate(self.pset.label):

            if atom_type == "Ni":
                my_colors.append('b')
            elif atom_type == "Au":
                my_colors.append('y')
            else:
                my_colors.append('k')

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111, projection='3d')

        initial_geometry = self.all_geometries[0]

        self.scat = self.ax1.scatter(initial_geometry[:,0], initial_geometry[:,1], initial_geometry[:,2],                                     marker='o',
                                      alpha=0.8,
                                      s=150,
                                      c=my_colors)
        ani = animation.FuncAnimation(self.fig,self.update_animation,
                                      frames=_np.arange(0,len(self.all_geometries),10),
                                      interval=1,blit=False)
        plt.show()

    def plot_results(self):
        plt.ioff()
        self.fig = plt.figure(figsize=plt.figaspect(2))
        self.ax2 = self.fig.add_subplot(211)
        self.ax3 = self.fig.add_subplot(212)

        t = np.arange(0, len(self.all_epot)) * self.__dt
        plt_epot = np.asarray(self.all_epot).flatten()

        plt_etot = np.asarray(self.all_etot).flatten()
        plt_temp = np.asarray(self.all_temp).flatten()
        self.epot_plot,=self.ax2.plot(t,plt_epot,c='C0')
        self.etot_plot, = self.ax2.plot(t, plt_etot, c='C1')
        self.ax2.set_xlabel("t / s")
        self.ax2.set_ylabel("E / eV")
        self.temp_plot, = self.ax3.plot(t, plt_temp, c='C2',label="Temperature /K")
        self.ax3.set_xlabel("t / s")
        self.ax3.set_ylabel("T / K")
        self.fig.legend(handles=[self.epot_plot,self.etot_plot], labels=["Potential energy/atom eV", "Total energy/atom /ev"], loc=1)
        plt.legend(handles=[self.temp_plot], loc=1)
        if self.save_png:
            self.fig.savefig(_os.path.join(self.png_path, "../run"))
        plt.show()


class Logger(object):

    def __init__(
        self, 
        log_file_path="./mdrun.log",
        xyz_file_path="./mdrun.xyz",
        step_index=0
        ):

        if log_file_path:
            self._log_file = open(log_file_path, 'a')
        else:
            self._log_file = False

        if xyz_file_path:
            self._log_file_xyz = open(xyz_file_path, 'a')
        else:
            self._log_file_xyz = False
            

        self._step_index = step_index

    @property
    def full_log_file(self):
        return self._log_file

    @full_log_file.setter
    def full_log_file(self, value):
        if value:
            self._log_file = open(value, 'a')
        else:
            raise ValueError("Invalid file path!")
    
    @property
    def xyz_log_file(self):
        return self._log_file_xyz

    @xyz_log_file.setter
    def xyz_log_file(self, value):
        if value:
            self._log_file_xyz = open(value, 'a')
        else:
            raise ValueError("Invalid file path!")

    def log(self, solver):
        
        # log to .xyz if endabled
        if self._log_file_xyz:
            log_str = self._make_xyz_log_str(solver.pset)
            self._log_file_xyz.write(log_str)

        # do full log if enabled
        if self._log_file:
            log_str = self._make_full_log_str(solver)
            self._log_file.write(log_str)

        self._step_index += 1

    def _atmic_data_to_string(self, species, postions, velocities, forces):
        log_str = species + " "
        log_str += " ".join(map(str, postions)) + " "
        # log xyz
        if self._enable_xyz:
            self._log_file_xyz.write(
                self._make_xyz_log_str(species, positions, step)
            )


        self._step_index += 1

    def _atmic_data_to_complete_string(self, species, positions, velocities, forces):
        log_str = self._atomic_data_to_xyz_string(species, positions) + " "
        log_str += " ".join(map(str, velocities)) + " "
        log_str += " ".join(map(str, forces))

        return log_str

    def _atomic_data_to_xyz_string(self, species, positions):
        log_str = species + " " + " ".join(map(str, positions)) + " "

        return log_str

    def _make_xyz_log_str(self, pset):
        time_step = self._step_index
        species = pset.label
        positions = pset.X

        number_of_atoms = len(species)

        log_str = str(number_of_atoms) + linesep
        log_str += "Atoms. Timestep: " + str(time_step) + linesep

        for i in range(number_of_atoms):
            log_str += \
                self._atomic_data_to_xyz_string(species[i], positions[i]) + linesep

        log_str.rstrip(linesep)
        return log_str

    def _make_full_log_str(self, solver):

        step = self._step_index
        time = solver.sim_time.time
        species = solver.pset.label
        positions = solver.pset.X


        # calculate all other properties
        velocities = solver.pset.V
        forces = solver.force.F

        E_tot = solver.force.Etot
        if type(E_tot) == type(np.array([])):
            E_tot = E_tot.flatten()
            if len(E_tot) == 1:
                E_tot = E_tot[0]


        E_pot = solver.force.Epot
        if type(E_pot) == type(np.array([])):
            E_pot = E_pot.flatten()
            if len(E_pot) == 1:
                E_pot = E_pot[0]

        temperature = thermo.get_temperature(solver.pset)

        #--- TODO: refactor this to sepearete funciton! ---
        # write system properties
        log_str = "Step: " + str(step) + linesep
        log_str += "Time: " + str(time) + linesep
        log_str += "Total Energy: " + str(E_tot) + linesep
        log_str += "Potential Energy: " + str(E_pot) + linesep
        log_str += "Temperature: " + str(temperature) + linesep

        #write header for particle properties
        log_str += self._atmic_data_to_complete_string(
            "Species",
            ["x","y", "z"],
            ["v_x", "v_y", "v_z"],
            ["f_x", "f_y", "f_z"]
        ) + linesep

        # wirte particle properties
        for i in range(solver.pset.size):
            log_str += self._atmic_data_to_complete_string(
                species[i],
                positions[i],
                velocities[i],
                forces[i]
            ) + linesep
        #---

        return log_str
