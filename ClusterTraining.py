#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:16:01 2017

@author: alexf1991
"""

import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")


#Load first trainings data 
Training=NN.AtomicNeuralNetInstance()

Training.XYZfile="NiAu_data.xyz"
Training.Logfile="log.md"
Training.SymmFunKeys=["1","2"]
Training.NumberOfRadialFunctions=7
Training.Lambs=[1.0,-1.0]
Training.Zetas=np.arange(0.1,5,0.5).tolist()

Training.read_files(True)
Training.make_training_and_validation_data(1000,70,30)

#Train with first data 
NrNi=1
NrAu=12

Training.Structures.append([Training.SizeOfInputs[0],150,150,1])
Training.NumberOfSameNetworks.append(NrNi)
Training.Structures.append([Training.SizeOfInputs[1],150,150,1])
Training.NumberOfSameNetworks.append(NrAu)
Training.HiddenType="truncated_normal"
Training.HiddenData=list()
Training.BiasData=list()
Training.ActFun="elu"
Training.ActFunParam=None
Training.LearningRate=0.00001
Training.CostCriterium=0.0015
Training.Epochs=1000
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.RegularizationParam=0.0001
#Training.expand_existing_net()
Training.make_and_initialize_network()

#Start training

Training.start_batch_training()
