#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:34:43 2017

@author: alexf1991
"""

import NeuralNetworkUtilities as NN
import numpy as np

NrAu=5
NrNi=5

Data=NN.DataInstance()

Data.XYZfile="NiAu_data.xyz"
Data.Logfile="log.2atoms"
Data.SymmFunKeys=["1","2"]
Data.Rs=np.arange(0.1,7,1).tolist()
Data.Etas=np.arange(0.1,2,0.5).tolist()
Data.Lambs=[1.0,-1.0]
Data.Zetas=np.arange(0.1,2,0.5).tolist()

Data.read_files()
Batches=Data.get_data(1000)

Training=NN.AtomicNeuralNetInstance()
Training.Structures.append([Data.SizeOfInputs[0],500,1])
Training.NumberOfSameNetworks.append(1)
Training.Structures.append([Data.SizeOfInputs[1],500,1])
Training.NumberOfSameNetworks.append(1)
Training.make_and_initialize_network()
Training.MakePlots=True
#With batches
Training.Epochs=500
Training.TrainingBatches=Batches
Training.start_batch_training()

#Witout batches
#Batches=Data.get_data(0,70,True)
#Training.TrainingInputs=Batches[0][0]
#Training.TrainingOutputs=Batches[0][1]
#Training.start_training()