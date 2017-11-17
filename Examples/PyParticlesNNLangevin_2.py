import pyparticles.pset.particles_set as ps
import pyparticles.animation.animated_scatter as anim
import pyparticles.forces.gravity as gr

from scipy.constants import gravitational_constant as G
from scipy.constants import Boltzmann as k_B

from NeuralNetworks import NeuralNetworkUtilities
import NeuralNetworks.md_utils 
from NeuralNetworks.md_utils.ode import leapfrog_solver as svs
from md_utils import nn_force
from NeuralNetworks import ReadLammpsData as _ReaderLammps
import numpy as np
import time
import unittest
import matplotlib.pyplot as plt




def provide_solver(pset, force_provider=None, dt = 2e-15, steps = 1000):

    if force_provider is None:
        force_provider = provide_gravity_force

    force = force_provider(pset)
    force.update_force(pset)
    # ---

    # set up solver system
    gamma = 0.2
    solver = svs.LeapfrogSolverLangevin(force, pset, dt, gamma)

    return solver

def provide_gravity_force(pset):
    grav = gr.Gravity(pset.size, Consts=G, m=pset.M)
    return grav

def provide_NN_force(pset):
    # --- get Network and force ---
    Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
    Training.prepare_evaluation(
    "/home/jcartus/Downloads/Au_test2",
    atom_types = ["Au"],
    nr_atoms_per_type = [pset.size]
    )

    NNForce = nn_force.NNForce(Training, pset.size)
    NNForce.set_masses(pset.M)
    NNForce.update_force(pset)
    #---

    return NNForce

def provide_pset():
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
    #pset.V[:] = np.random.normal(0, 1, (len(start_geom), 3))

    pset.unit = 1e10
    pset.mass_unit = 1.660e+27

    bound = None
    pset.set_boundary(bound)
    pset.enable_log(True, log_max_size=1000)

    return pset

def main():
    #--- set md run parameters ---
    dt = 2e-15
    steps = 10000

    pset = provide_pset()
    #---

    solver = provide_solver(pset)
    solver.start()
    """
    #--- run solver and display movement ---
    a = anim.AnimatedScatter()
    a.xlim=(-5e-10, 5e-10)
    a.ylim=(-5e-10, 5e-10)
    a.zlim=(-5e-10, 5e-10)
    #a=acli.AnimatedCLI()
    a.trajectory = True
    a.trajectory_step = 1
    a.ode_solver = solver
    a.pset = pset
    a.steps = steps
    a.build_animation(interval=1)
    a.start()
    """
    #---

if __name__ == '__main__':
    main()
