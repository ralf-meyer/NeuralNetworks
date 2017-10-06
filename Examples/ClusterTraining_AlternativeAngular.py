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

Training.XYZfile="2ClusterNiAu_data.xyz"
Training.Logfile="2cluster.md"
Training.atomtypes=["1","2"]
Training.NumberOfRadialFunctions=6
#angular symmetry function settings
Training.Lambs=[1.5]
Training.Zetas=[-np.pi/2,-np.pi/3,-np.pi/4,0,np.pi/4,np.pi/3,np.pi/2,np.pi]
Training.Etas=[0.1]

Training.read_files()
Training.make_training_and_validation_data(100,70,30)

#Train with first data 
NrNi=12
NrAu=14

Training.Structures.append([Training.SizeOfInputs[0],80,80,15,1])
Training.NumberOfSameNetworks.append(NrNi)
Training.Structures.append([Training.SizeOfInputs[1],80,80,15,1])
Training.NumberOfSameNetworks.append(NrAu)
Training.HiddenType="truncated_normal"
Training.HiddenData=list()
Training.BiasData=list()
Training.ActFun="elu"
Training.ActFunParam=None
Training.LearningRate=0.001
Training.dE_Criterium=0.03
Training.Epochs=1500
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.CostFunType="Adaptive_2"
Training.LearningRateType="exponential_decay"
Training.SavingDirectory="save_cluster"
Training.LearningDecayEpochs=100
Training.RegularizationParam=0.001
Training.InputDerivatives=True
#Training.expand_existing_net()
Training.make_and_initialize_network()

#Start training

Training.start_batch_training()
