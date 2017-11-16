import numpy as np
import pyparticles.pset.particles_set as ps
import pyparticles.forces.gravity as gr
import pyparticles.ode.euler_solver as els
import pyparticles.ode.leapfrog_solver as lps
import pyparticles.ode.runge_kutta_solver as rks
from NeuralNetworks.md_utils.ode import leapfrog_solver as svs
import pyparticles.ode.midpoint_solver as mds
import pyparticles.animation.animated_ogl as aogl
import pyparticles.animation.animated_scatter as  anim
import pyparticles.animation.animated_cli as acli
from NeuralNetworks import NeuralNetworkUtilities
from md_utils import nn_force
from NeuralNetworks import ReadLammpsData as _ReaderLammps
import numpy as np
import time

import pyparticles.forces.gravity as grav


start_geom=[('Au', np.asarray([ 0.,  0.,  0.])),
            ('Au', np.asarray([-2.87803, -0.53645,  0.19258])),
            ('Au', np.asarray([-0.87803, -2.53645,  0.19258])),
            ]
print(len(start_geom))
#Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
#Training.prepare_evaluation("/home/jcartus/Downloads/Model_Gold",atom_types=["Au"],nr_atoms_per_type=[len(start_geom)])

dt = 2e-15
steps = 1000


pset = ps.ParticlesSet( len(start_geom) , 3 , label=True,mass=True)
pset.thermostat_coupling_time=dt*10
pset.thermostat_temperature=1000
geom=[]
masses=[]
for i,atom in enumerate(start_geom):
    pset.label[i] = atom[0]
    masses.append([196])
    geom.append(atom[1])

# Coordinates
pset.X[:] = np.array(geom)
# Mass
pset.M[:] = np.array(masses)
# Speed
pset.V[:] = np.zeros((len(start_geom),3))

pset.unit = 1e10
pset.mass_unit =1.660e+27


bound = None
pset.set_boundary( bound )
pset.enable_log( True , log_max_size=1000 )

#NNForce=nn_force.NNForce(Training,pset.size)
#NNForce.set_masses(pset.M)
#NNForce.update_force(pset)

force=grav.Gravity(pset.size)

#solver = rks.RungeKuttaSolver( grav , pset , dt )

#solver = mds.MidpointSolver( grav , pset , dt )
#solver = els.EulerSolver( grav , pset , dt )
#solver = lps.LeapfrogSolver( grav , pset , dt )
#solver = svs.LeapfrogSolverBerendsen( NNForce , pset , dt )
solver = svs.LeapfrogSolverBerendsen( force , pset , dt )

#a = aogl.AnimatedGl()

a = anim.AnimatedScatter()
a.xlim=(-5e-10,5e-10)
a.ylim=(-5e-10,5e-10)
a.zlim=(-5e-10,5e-10)
#a=acli.AnimatedCLI()
a.trajectory = True
a.trajectory_step = 1
a.ode_solver = solver
a.pset = pset
a.steps = steps
a.build_animation(interval=0.1)
a.start()

