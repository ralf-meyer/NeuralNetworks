from NeuralNetworks.md_utils.ode import leapfrog_solver as svs
from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks.md_utils import nn_force
from NeuralNetworks.md_utils.pset import particles_set as ps
from NeuralNetworks.md_utils import thermostats
from NeuralNetworks.data_generation import data_readers
from NeuralNetworks.optimize import optimizer
import numpy as np
import time



start_geom=[('Ni', np.array([-0.03069231,  0.077     , -0.00761538])),
            ('Au', np.array([-0.04213387,  1.65109681,  2.56282261])),
            ('Au', np.array([ 1.45360026,  2.70028588, -0.02522022])),
            ('Au', np.array([ 2.52987894,  0.14755908,  1.58095615])),
            ('Au', np.array([-1.68161921,  2.59877743, -0.01953711])),
            ('Au', np.array([-2.59021207, -0.02168922,  1.58115213])),
            ('Au', np.array([ 0.04017376, -1.48468366,  2.56943435])),
            ('Au', np.array([-1.5562628 , -2.52255613, -0.00521194])),
            ('Au', np.array([ 1.64448684, -2.42859689, -0.03629008])),
            ('Au', np.array([ 2.51447433,  0.13478519, -1.62126059])),
            ('Au', np.array([-0.08720525,  1.67958769, -2.55978744])),
            ('Au', np.array([-2.6155158 , -0.00973963, -1.55559049])),
            ('Au', np.array([ 0.02202719, -1.52082656, -2.56285199]))]

input_reader=data_readers.SimpleInputReader()
input_reader.read("/home/afuchs/Documents/Validation_geometries/ico_NiAu54.xyz",skip=3)
#start_geom=input_reader.geometries[0]
Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.CalcDatasetStatistics=False
Training.TextOutput=False
Training.prepare_evaluation("/home/afuchs/Documents/NiAu_Training/multi_more_radial_force2",nr_atoms_per_type=[1,12])

opt=optimizer.Optimizer(Training,start_geom)
#opt.check_gradient()
start_geom=opt.start_conjugate_gradient()#opt.start_bfgs(norm=10,gtol=1-05)
# print(start_geom)
#print(len(start_geom))
input_reader=data_readers.SimpleInputReader()
input_reader.read("/home/afuchs/Documents/Validation_geometries/ico_NiAu54.xyz",skip=3)
#start_geom=input_reader.geometries[0]
Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.TextOutput=False
Training.CalcDatasetStatistics=False
Training.prepare_evaluation("/home/afuchs/Documents/NiAu_Training/multi_more_radial_force2",nr_atoms_per_type=[1,12])

dt = 5e-15
steps = 300000


pset = ps.ParticlesSet( len(start_geom) , 3 , label=True,mass=True)
pset.thermostat_coupling_time=dt*200
pset.thermostat_temperature=100
pset.dT=float(400)/steps
pset.unit = 1e10
pset.mass_unit =1.660e+27
geom=[]
masses=[]
for i,atom in enumerate(start_geom):
    pset.label[i] = atom[0]
    masses.append([196])
    geom.append(atom[1])


masses[0]=[59]
# Coordinates
pset.X[:] = np.array(geom)
# Mass
pset.M[:] = np.array(masses)
# Speed
pset=thermostats.set_temperature(pset,100)




bound = None
pset.set_boundary( bound )
pset.enable_log( True , log_max_size=1000 )

NNForce=nn_force.NNForce(Training,pset.size)
NNForce.set_masses(pset.M)
NNForce.update_force(pset)
solver = svs.LeapfrogSolverBerendsen( NNForce , pset , dt )
#solver =svs.LeapfrogSolverLangevin(NNForce,pset,dt,1e10)
solver.plot=True
solver.plot_steps=10000
solver.steps=steps
solver.save_png=True
solver.png_path="/home/afuchs/Documents/NiAu_Md/Ni1Au12/"
start=time.time()
solver.start()
np.save("/home/afuchs/Documents/MD_Runs/Ni1Au12.npy",np.asarray([solver.all_epot,solver.all_etot,solver.all_temp]))
solver.plot_results()
solver.animate_run()
