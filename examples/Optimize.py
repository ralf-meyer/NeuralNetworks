from NeuralNetworks import NeuralNetworkUtilities
import numpy as np
from NeuralNetworks.optimize import optimizer

start_geom=[('Au', np.asarray([ 0.,  0.,  0.])),
            ('Au', np.asarray([-2.87803, -0.53645,  0.19258])),
            ('Au', np.asarray([-0.87803, -2.53645,  0.19258])),
            ]
Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.prepare_evaluation("/home/afuchs/Documents/Au_training/Au_test2",atom_types=["Au"],nr_atoms_per_type=[len(start_geom)])

opt=optimizer.Optimizer(Training,start_geom)
opt.plot=True
#opt.check_gradient()
res=opt.start_bfgs()
print(res)