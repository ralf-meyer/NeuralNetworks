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

Training=NN.AtomicNeuralNetInstance()

Training.XYZfile="NiAu_data_2AU1Ni.xyz"
Training.Logfile="log.3atoms"
Training.Atomtypes=["1","2"]
Training.NumberOfRadialFunctions=6
#angular symmetry function settings
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]

Training.read_files()
Batches=Training.get_data(100,70)
ValidationBatches=Training.get_data(100,30)
#Batches=Data.get_data(0,1,True)


Training.Structures.append([Training.SizeOfInputs[0],25,25,10,1])
Training.NumberOfSameNetworks.append(NrNi)
Training.Structures.append([Training.SizeOfInputs[1],25,25,10,1])
Training.NumberOfSameNetworks.append(NrAu)
Training.HiddenType="truncated_normal"
Training.HiddenData=list()
Training.BiasData=list()
Training.ActFun="elu"
Training.ActFunParam=None
Training.LearningRate=0.001
Training.dE_Criterium=0.043
Training.Epochs=1500
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.CostFunType="other"
Training.make_and_initialize_network()

#With batches
Training.TrainingBatches=Batches
Training.ValidationBatches=ValidationBatches
Training.start_batch_training()

#Witout batches

#Training.TrainingInputs=Batches[0][0]
#Training.TrainingOutputs=Batches[0][1]
#Training.start_training()