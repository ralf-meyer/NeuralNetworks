"""
Tests the md util for this project.

TODO:
* 
* include a unit test for the temeperature calculation in thermostats

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
        start_geom = reader.geometries

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
        thermostats.set_temperature(pset, T)

        bound = None
        pset.set_boundary(bound)
        pset.enable_log(True, log_max_size=1000)

        return pset


class ForceModelProvider(object):

    @staticmethod
    def provide_LenardJones_force(pset):

        force = LenardJones(pset.size)
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

    #@staticmethod
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

        def _check_conservation_of_temperature(self, solver, steps=800, delta=2):

            T = solver.pset.thermostat_temperature

            # check after 500 steps
            self._advance_solver(solver, steps)
            self.assertAlmostEqual(
                1,
                self._temerature_of_system(solver.pset) / T,
                delta=delta
            )

        def _temerature_of_system(self, pset):
            """Based on: E_kin = mv^2/2; <E> = 3 N k_B T / 2 in Md System
            """
            return  thermostats.get_temperature(pset)

class TestLangevinVelocityVerlet(_BaseTestWarpper.TestODESolver):
    def setUp(self):

        def pset_callback():
            v_max = 0#1.5e-20
            return PSetProvider.provide_Au3_fast_start(v_max)

        self._pset_provider = pset_callback
        self._solver_provider = \
            ODESolverProvider.provide_langevin_velocity_verlet

    def test_no_crash(self):
        """make sure solver does not collapse (raise an exception) during x 
        steps."""

        dt = 2e-15
        steps = 1000
        gamma = 1e-3
        v_max=1e-8
        pset = self._pset_provider()

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
    

    def test_temperature_conservation(self):

        dt = 2e-15
        steps = 1000
        gamma = 1e-3
        v_max=1e-8

        #Au13 staring at 1000 K
        pset = PSetProvider.provide_Au13_1000_Kelvin()

        solver = self._solver_provider(
            pset,
            ForceModelProvider.provide_LenardJones_force,
            dt=dt,
            steps=steps,
            gamma=gamma
        )

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
        delta = 1

        # Gold mass in u
        m = 196
        T = 1000 # K

        # draw velocities
        v = self._sampler.draw_boltzman_scalars(nsamples, T, m, 1.66e-27)
    
        T_actual = self._calculate_temperature_from_scalar_velocities(v, m)

        self.assertAlmostEqual(1, T_actual / T, delta)

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