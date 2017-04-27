#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:34:43 2017

@author: alexf1991
"""

import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

NrAu=2
NrNi=1

Data=NN.DataInstance()

Data.XYZfile="NiAu_data_2AU1Ni.xyz"
Data.Logfile="log.3atoms"
Data.SymmFunKeys=["1","2"]
Data.Rs=np.arange(0.1,7,1).tolist()
Data.Etas=np.arange(0.1,1.1,1).tolist()
Data.Lambs=[1.0,-1.0]
Data.Zetas=np.arange(0.1,2,0.5).tolist()

Data.read_files()
Batches=Data.get_data(100,70)
ValidationBatches=Data.get_data(100,30)
#Batches=Data.get_data(0,1,True)

Training=NN.AtomicNeuralNetInstance()
Training.Structures.append([Data.SizeOfInputs[0],10,1])
Training.NumberOfSameNetworks.append(NrNi)
Training.Structures.append([Data.SizeOfInputs[1],10,1])
Training.NumberOfSameNetworks.append(NrAu)
Training.HiddenType="truncated_normal"
Training.HiddenData=None
Training.BiasData=None
Training.ActFun="tanh"
Training.ActFunParam=None
Training.LearningRate=0.000001
Training.Epochs=1000
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.make_and_initialize_network()

#With batches
Training.TrainingBatches=Batches
Training.ValidationBatches=ValidationBatches
Training.start_batch_training()

#Witout batches

#Training.TrainingInputs=Batches[0][0]
#Training.TrainingOutputs=Batches[0][1]
#Training.start_training()