import pyparticles.pset.particles_set as ps
import pyparticles.forces.gravity as gr

from scipy.constants import gravitational_constant as G
from scipy.constants import Boltzmann as k_B
import numpy as np

from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks.md_utils.ode import leapfrog_solver as svs
from md_utils import nn_force

import unittest



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

        def _check_conservation_of_temperature(self, solver, steps=800):

            T = solver.pset.thermostat_temperature

            # check after 500 steps
            self._advance_solver(solver, steps)
            self.assertAlmostEqual(
                T,
                self._temerature_of_system(solver.pset),
                delta=2
            )

        def _temerature_of_system(self, pset):
            """Based on: E_kin = mv^2/2; <E> = 3 N k_B T / 2 in Md System
            """

            # sum first over vx,vy,vz for each particle, multiply by mass and sum again
            kinetic_energy = np.sum(np.sum(pset.V**2, 1) * pset.M[:]) / 2
            N = pset.size
            temperature = kinetic_energy * 2 / (3 * N * k_B)

            return  temperature

class TestLangevinVelocityVerlet(_BaseTestWarpper.TestODESolver):
    def setUp(self):

        def pset_callback():
            v_max = 0#1.5e-20
            return PSetProvider.provide_Au3_fast_start(v_max)

        self._pset_provider = pset_callback
        self._solver_provider = \
            ODESolverProvider.provide_langevin_velocity_verlet



    def test_temperature_conservation_in_gravity_field(self):
        dt = 2e-15
        steps = 10000

        solver = self._solver_provider(
            self._pset_provider(),
            ForceModelProvider.provide_gravity_force,
            dt=dt,
            steps=steps
        )

        self._check_conservation_of_temperature(solver, 7000)


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
    def provide_Au3_slow_start():
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

class ForceModelProvider(object):
    @staticmethod
    def provide_gravity_force(pset):
        grav = gr.Gravity(pset.size, Consts=G)
        return grav

    @staticmethod
    def provide_NN_force(pset):
        # --- get Network and force ---
        Training = NeuralNetworkUtilities.AtomicNeuralNetInstance()
        Training.prepare_evaluation(
            "/home/jcartus/Downloads/Au_test2",
            atom_types=["Au"],
            nr_atoms_per_type=[pset.size]
        )

        NNForce = nn_force.NNForce(Training, pset.size)
        NNForce.set_masses(pset.M)
        NNForce.update_force(pset)
        # ---

        return NNForce

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


if __name__ == '__main__':
    unittest.main()