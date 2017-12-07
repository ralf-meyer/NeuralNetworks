from NeuralNetworks.md_utils.ode import leapfrog_solver as svs
from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks.md_utils import nn_force
from NeuralNetworks.md_utils.pset import particles_set as ps
from NeuralNetworks.data_generation import data_readers
import numpy as np



start_geom=[('Ni', np.asarray([ 0.001,  0.001,  0.001])),
            ('Au', np.asarray([-0.000000000,1.538402219,2.256385110])),
            ('Au', np.asarray([1.538402219 , 2.256385110, -0.000000000])),
            ('Au', np.asarray([2.256385110 ,-0.000000000 , 1.538402219])),
            ('Au', np.asarray([-1.538402219 ,2.256385110 , -0.000000000])),
            ('Au', np.asarray([-2.256385110 , 0.000000000 , 1.538402219])),
            ('Au', np.asarray([-0.000000000 , -1.538402219 ,  2.256385110])),
            ('Au', np.asarray([-1.538402219 , -2.256385110 ,  0.000000000])),
            ('Au', np.asarray([1.538402219 , -1.256385110 ,  0.000000000])),
            ('Au', np.asarray([2.356385110 ,  0.000000000 , -1.538402219])),
            ('Au', np.asarray([-0.000000000 ,  1.538402219 , -2.256385110])),
            ('Au', np.asarray([-2.756385110  , 0.000000000 , -1.638402219])),
            ('Au', np.asarray([-0.000000000  ,-1.538402219 , -2.256385110])),
            ]
#print(len(start_geom))
input_reader=data_readers.SimpleInputReader()
input_reader.read("/home/afuchs/Documents/Validation_geometries/ico_NiAu54.xyz",skip=3)
start_geom=input_reader.geometries[0]
Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.TextOutput=False
Training.prepare_evaluation("/home/afuchs/Documents/NiAu_Training/NiAu_Test_without_prex",nr_atoms_per_type=[1,54])

dt = 1e-15
steps = 10000


pset = ps.ParticlesSet( len(start_geom) , 3 , label=True,mass=True)
pset.thermostat_coupling_time=dt*5000
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

NNForce=nn_force.NNForce(Training,pset.size)
NNForce.set_masses(pset.M)
NNForce.update_force(pset)

solver = svs.LeapfrogSolverBerendsen( NNForce , pset , dt )
solver.save_png=True
solver.png_path="/home/afuchs/Documents/NiAu_Md/Ni1Au54/"
solver.start()

