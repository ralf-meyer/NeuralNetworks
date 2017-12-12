"""
Tests the md util for this project.

TODO:
* Test for get_temperature for higher temperature.

"""


from os.path import join, dirname, normpath

from scipy.constants import gravitational_constant as G
from scipy.constants import Boltzmann as k_B
import numpy as np

from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks.md_utils import nn_force, thermostats
from NeuralNetworks.md_utils.ode import leapfrog_solver as svs
import NeuralNetworks.md_utils.pset.particles_set as ps
from NeuralNetworks.data_generation.data_readers import SimpleInputReader

from pyparticles.forces.lennard_jones import LenardJones

import unittest


class PathProvider(object):
    """This class is just used to provide hard coded file paths"""
    _test_files_dir = join(dirname(__file__), normpath("TestData"))
    MD_Utils_test_files_dir = join(_test_files_dir, "MDUtils")


class PSetProvider(object):

    @staticmethod
    def provide_Au3_fast_start(v_max):
        start_geom = [
            ('Au', np.asarray([0., 0., 0.])),
            ('Au', np.asarray([-2.87803, -0.53645, 0.19258])),
            ('Au', np.asarray([-0.87803, -2.53645, 0.19258])),
        ]

        pset = ps.ParticlesSet(len(start_geom), 3, label=True, mass=True)
        pset.thermostat_temperature = 1000

        geom = []
        masses = []
        for i, atom in enumerate(start_geom):
            pset.label[i] = atom[0]
            masses.append([196])
            geom.append(atom[1])

        # Coordinates
        pset.X[:] = np.array(geom)
        # Mass
        pset.M[:] = np.array(masses)
        # Speed
        sigma = np.sqrt(v_max)
        pset.V[:] = np.random.normal(0, sigma, (len(start_geom), 3))
        # (np.random.rand(len(start_geom), 3)-0.5) * v_max * 2

        pset.unit = 1e10
        pset.mass_unit = 1.660e+27

        bound = None
        pset.set_boundary(bound)
        pset.enable_log(True, log_max_size=1000)

        return pset

    @staticmethod
    def provide_Au3_Zero_Kelvin():
        start_geom = [
            ('Au', np.asarray([0., 0., 0.])),
            ('Au', np.asarray([-2.87803, -0.53645, 0.19258])),
            ('Au', np.asarray([-0.87803, -2.53645, 0.19258])),
        ]

        pset = ps.ParticlesSet(len(start_geom), 3, label=True, mass=True)
        pset.thermostat_temperature = 1000

        geom = []
        masses = []
        for i, atom in enumerate(start_geom):
            pset.label[i] = atom[0]
            masses.append([196])
            geom.append(atom[1])

        # Coordinates
        pset.X[:] = np.array(geom)
        # Mass
        pset.M[:] = np.array(masses)
        # Speed
        pset.V[:] = np.zeros((len(start_geom), 3))

        pset.unit = 1e10
        pset.mass_unit = 1.660e+27

        bound = None
        pset.set_boundary(bound)
        pset.enable_log(True, log_max_size=1000)

        return pset

    @staticmethod
    def provide_Au13_Zero_Kelvin():
        #read dodecaeder gold cluster config from input file
        reader = SimpleInputReader()
        reader.read(join(PathProvider.MD_Utils_test_files_dir, "Au13.in"))
        start_geom = reader.geometries

        pset = ps.ParticlesSet(len(start_geom), 3, label=True, mass=True)
        pset.thermostat_temperature = 1000
        geom = []
        masses = []
        for i, atom in enumerate(start_geom):
            pset.label[i] = atom[0]
            masses.append([196])
            geom.append(atom[1])

        # Coordinates
        pset.X[:] = np.array(geom)
        # Mass
        pset.M[:] = np.array(masses)
        # Speed
        pset.V[:] = np.zeros((len(start_geom), 3))

        pset.unit = 1e10
        pset.mass_unit = 1.660e+27

        bound = None
        pset.set_boundary(bound)
        pset.enable_log(True, log_max_size=1000)

        return pset


        

    @staticmethod
    def provide_Au13_1000_Kelvin():
        
        T = 1000 #K


        #read dodecaeder gold cluster config from input file
        # geometries are in angstroem
        reader = SimpleInputReader()
        reader.read(join(PathProvider.MD_Utils_test_files_dir, "Au13.in"))
        start_geom = reader.geometries[0]

        pset = ps.ParticlesSet(len(start_geom), 3, label=True, mass=True)

        geom = []
        masses = []
        for i, atom in enumerate(start_geom):
            pset.label[i] = atom[0]
            masses.append([196])
            geom.append(atom[1])

        # Coordinates
        pset.X[:] = np.array(geom)
        # Mass
        pset.M[:] = np.array(masses)
        # Speed
        pset.V[:] = np.zeros((len(start_geom), 3))

        pset.unit = 1e10 # m -> angstroem
        pset.mass_unit = 1.660e+27 

        # set initial velocities
        pset = thermostats.set_temperature(pset, T)

        bound = None
        pset.set_boundary(bound)
        pset.enable_log(True, log_max_size=1000)

        return pset

class ForceModelProvider(object):

    @staticmethod
    def provide_LenardJones_force(pset):

        # depth of the potential (220/Loschmid kJ/per particle in energy units of the system )
        epsilon = 6e-30 * (pset.unit**2) * (pset.mass_unit)
        # 4 Angstroem cutoff (in the units of the used pset)
        sigma = 4e-10 * pset.unit  

        force = LenardJones(pset.size, Consts=( epsilon , sigma ))
        force.set_masses(pset.M) 
        force.update_force(pset)
        return force

class ODESolverProvider(object):
    @staticmethod
    def provide_langevin_velocity_verlet(pset, force_provider, gamma=1e2, dt=2e-15, steps=1000):

        force = force_provider(pset)
        force.update_force(pset)

        # set up solver system
        solver = svs.LeapfrogSolverLangevin(force, pset, dt, gamma)

        return solver

    def provide_velocity_verlet_w_berendsen_thermostat(pset, force_provider, dt=2e-15, steps=1000):
        force = force_provider(pset)
        force.update_force(pset)

        # set up solver system
        solver = svs.LeapfrogSolverBerendsen(force, pset, dt)
        return solver

class _BaseTestWarpper(object):
    class TestODESolver(unittest.TestCase):

        def setUp(self):
            self._pset = None
            self._solver_provider = None
            self._force_provider = None

        def _advance_solver(self, solver, steps):

            solver.update_force()

            for i in range(steps):
                solver.step()

        def _check_conservation_of_energy(self):
            raise NotImplementedError()

        def _check_conservation_of_temperature(self, solver, steps=800, delta=50):

            T = solver.pset.thermostat_temperature

            # check after some steps
            self._advance_solver(solver, steps)
            
            T_actual = self._temperature_of_system(solver.pset)
            
            self.assertAlmostEqual(T, T_actual, delta=delta)

        def _temperature_of_system(self, pset):
            """Based on: E_kin = mv^2/2; <E> = 3 N k_B T / 2 in Md System
            """
            return  thermostats.get_temperature(pset)

class TestLangevinVelocityVerlet(_BaseTestWarpper.TestODESolver):
    def setUp(self):

       
        self._solver_provider = \
            ODESolverProvider.provide_langevin_velocity_verlet

    def test_no_crash(self):
        """make sure solver does not collapse (raise an exception) during x 
        steps."""

        dt = 2e-15
        steps = 1000
        gamma = 1e-3
        pset = PSetProvider.provide_Au13_1000_Kelvin()

        solver = self._solver_provider(
            pset,
            ForceModelProvider.provide_LenardJones_force,
            dt=dt,
            steps=steps,
            gamma=gamma
        )

        try:
            self._advance_solver(solver, steps)
        except:
            steps = solver.get_steps()
            self.fail(
                "Langevin Velocity-Verlet failed after {0} steps!".format(
                    solver.get_steps()
                )
            )
    

    def _test_temperature_conservation(self):

        dt = 2e-15
        steps = 100000
        gamma = 1e6 #TODO vernuenfitger wert!!!

        #Au13 staring at 1000 K
        pset = PSetProvider.provide_Au13_1000_Kelvin()

        solver = self._solver_provider(
            pset,
            ForceModelProvider.provide_LenardJones_force,
            dt=dt,
            steps=steps,
            gamma=gamma
        )

        solver.plot = True

        self._check_conservation_of_temperature(
            solver, 
            steps=steps
        )

class TestThermostat(unittest.TestCase):

    def test_get_temperature(self):
        #test 0 K ensemble
        pset_0K = PSetProvider.provide_Au3_Zero_Kelvin()
        self.assertEqual(0, thermostats.get_temperature(pset_0K))

class TestSampler(unittest.TestCase):

    def setUp(self):
        self._sampler = thermostats.Sampler()

    def test_sampled_temperature(self):
        """Sample a bunch of speeds and see if their temperature is correct"""

        nsamples = int(1e6)
        delta = 10 # must be within 1000 +-10 K

        # Gold mass in u
        m = 196
        T = 1000 # K

        # draw velocities
        v = self._sampler.draw_boltzman_scalars(nsamples, T, m, 1.66e-27)
    
        T_actual = self._calculate_temperature_from_scalar_velocities(v, m)

        self.assertAlmostEqual(T, T_actual, delta=delta)

    def _calculate_temperature_from_scalar_velocities(self, v, m, mass_unit=1.66e-27):
        """calculates the temperature from a bunch of velocities
        
        1/2<m v^2> = 3/2 k_B T
        https://en.wikipedia.org/wiki/Thermal_velocity

        mass_unit: factor to convert given mass into kg.

        """
        T = np.mean(v**2 * m * mass_unit  / (3 * k_B))
        return T    


if __name__ == '__main__':
    unittest.main()