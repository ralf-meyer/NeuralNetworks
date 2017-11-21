from NeuralNetworks.md_utils.ode import leapfrog_solver as svs
from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks.md_utils import nn_force
from NeuralNetworks.md_utils.pset import particles_set as ps
import numpy as np



start_geom=[('Au', np.asarray([ 0.,  0.,  0.])),
            ('Au', np.asarray([-2.87803, -0.53645,  0.19258])),
            ('Au', np.asarray([-0.87803, -2.53645,  0.19258])),
            ]
print(len(start_geom))
Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.prepare_evaluation("/home/afuchs/Documents/Au_training/Au_test4",atom_types=["Au"],nr_atoms_per_type=[len(start_geom)])

dt = 2e-14
steps = 1000


pset = ps.ParticlesSet( len(start_geom) , 3 , label=True,mass=True)
pset.thermostat_coupling_time=dt*10
pset.thermostat_temperature=3000
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

NNForce=nn_force.NNForce(Training,pset.size)
NNForce.set_masses(pset.M)
NNForce.update_force(pset)

solver = svs.LeapfrogSolverBerendsen( NNForce , pset , dt )
solver.start()

