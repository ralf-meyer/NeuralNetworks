from NeuralNetworks import NeuralNetworkUtilities
import numpy as np
from NeuralNetworks.optimize import optimizer
from NeuralNetworks.data_generation import data_readers
import random

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


input_reader=data_readers.SimpleInputReader()
input_reader.read("/home/afuchs/Documents/Validation_geometries/ico_NiAu54.xyz",skip=3)
#start_geom=input_reader.geometries[0]
Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.CalcDatasetStatistics=False
Training.TextOutput=True
Training.prepare_evaluation("/home/afuchs/Documents/NiAu_Training_part/1/",nr_atoms_per_type=[1,12])

opt=optimizer.Optimizer(Training,start_geom)
opt.save_png=True
opt.png_path="/home/afuchs/Documents/NiAu_Opt/Ni1Au54_final/"
opt.plot=True
opt.check_gradient()
#res=opt.start_bfgs(norm=10,gtol=1-05)
#res=opt.start_conjugate_gradient()
#opt.start_nelder_mead()
print("Resulting geometry: "+str(res))
