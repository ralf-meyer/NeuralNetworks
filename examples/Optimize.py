from NeuralNetworks import NeuralNetworkUtilities
import numpy as np
from NeuralNetworks.optimize import optimizer
from NeuralNetworks.data_generation import data_readers
import random

start_geom=[('Ni', np.asarray([ 0.001,  0.001,  0.001])),
            ('Au', np.asarray([-0.000000000,1.538402219,2.256385110])),
            ('Au', np.asarray([1.638402219 , 2.256385110, -0.000000000])),
            ('Au', np.asarray([2.256385110 ,-0.000000000 , 1.238402219])),
            ('Au', np.asarray([-1.78402219 ,2.256385110 , -0.000000000])),
            ('Au', np.asarray([-2.456385110 , 0.000000000 , 1.338402219])),
            ('Au', np.asarray([-0.000000000 , -1.538402219 ,  2.256385110])),
            ('Au', np.asarray([-1.538402219 , -2.256385110 ,  0.000000000])),
            ('Au', np.asarray([1.538402219 , -1.556385110 ,  0.000000000])),
            ('Au', np.asarray([2.656385110 ,  0.100000000 , -1.538402219])),
            ('Au', np.asarray([-2.000000000 ,  1.538402219 , -2.256385110])),
            ('Au', np.asarray([-2.756385110  , 0.000000000 , -1.638402219])),
            ('Au', np.asarray([-0.000000000  ,-1.738402219 , -2.256385110])),
            ]


input_reader=data_readers.SimpleInputReader()
input_reader.read("/home/afuchs/Documents/Validation_geometries/ico_NiAu54.xyz",skip=3)
#start_geom=input_reader.geometries[0]
Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.TextOutput=False
Training.prepare_evaluation("/home/afuchs/Documents/NiAu_Training/NiAu_Test_without_prex4",nr_atoms_per_type=[1,12],atom_types= ["Ni","Au"])

opt=optimizer.Optimizer(Training,start_geom)
#opt.save_png=True
#opt.png_path="/home/afuchs/Documents/NiAu_Opt/PNGs/"
opt.plot=True
#opt.check_gradient()
res=opt.start_bfgs()
#opt.start_dogleg()
#print(res)