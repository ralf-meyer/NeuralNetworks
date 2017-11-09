import numpy as np
import pyparticles.pset.particles_set as ps
import pyparticles.forces.gravity as gr
import pyparticles.ode.euler_solver as els
import pyparticles.ode.leapfrog_solver as lps
import pyparticles.ode.runge_kutta_solver as rks
import pyparticles.ode.stormer_verlet_solver as svs
import pyparticles.ode.midpoint_solver as mds
import pyparticles.animation.animated_ogl as aogl
import pyparticles.animation.animated_scatter as  anim
import pyparticles.animation.animated_cli as acli
from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks import ReadLammpsData as _ReaderLammps
import numpy as np
import time


start_geom=[('Au', np.asarray([ 0.,  0.,  0.])),
            ('Au', np.asarray([-4.87803, -0.53645,  0.19258])),
            ('Au', np.asarray([-0.87803, -4.53645,  0.19258]))
            ]

Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.prepare_evaluation("/home/afuchs/Git/NeuralNetworks/Au_test2",atom_types=["Au"],nr_atoms_per_type=[3])

dt = 1e-14
steps = 1000


pset = ps.ParticlesSet( 3 , 3 , label=True,mass=True)

geom=[]
masses=[]
for i,atom in enumerate(start_geom):
    pset.label[i] = atom[0]
    masses.append([196*(1.6/1.66)*1e-28])
    geom.append(atom[1])

# Coordinates
pset.X[:] = np.array(geom)
# Mass
pset.M[:] = np.array(masses)
# Speed
pset.V[:] = np.zeros((3,3))

pset.unit = 1#1e-10
pset.mass_unit =1# 1.660e-27


bound = None
pset.set_boundary( bound )
pset.enable_log( True , log_max_size=1000 )

NNForce=NeuralNetworkUtilities.PyParticlesNNForce(Training,pset.size)
NNForce.set_masses(pset.M)
NNForce.update_force(pset)

#solver = rks.RungeKuttaSolver( grav , pset , dt )

#solver = mds.MidpointSolver( grav , pset , dt )
#solver = els.EulerSolver( grav , pset , dt )
#solver = lps.LeapfrogSolver( grav , pset , dt )
solver = svs.StormerVerletSolver( NNForce , pset , dt )
#a = aogl.AnimatedGl()

a = anim.AnimatedScatter()
a.xlim=(-10,10)
a.ylim=(-10,10)
a.zlim=(-10,10)
#a=acli.AnimatedCLI()
a.trajectory = True
a.trajectory_step = 1
a.ode_solver = solver
a.pset = pset
a.steps = steps
a.build_animation(interval=0.2)
a.start()

