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
Training.NumberOfRadialFunctions=7
#angular symmetry function settings
Training.Lambs=[1.0,-1.0]
#Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Zetas=[0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3]
Training.Etas=[0.1]#,0.2,0.5,1,2]
#Training.Rs=[0,1,1,1.5,2,2.5,3,3.5,4,5,6]

Training.read_files(True)
Training.make_training_and_validation_data(100,70,30)

#Train with first data 
NrNi=12
NrAu=14

Training.Structures.append([Training.SizeOfInputs[0],80,80,15,1])
Training.NumberOfAtomsPerType.append(NrNi)
Training.Structures.append([Training.SizeOfInputs[1],80,80,15,1])
Training.NumberOfAtomsPerType.append(NrAu)
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
#Training.expand_existing_net()
Training.make_and_initialize_network()

#Start training

Training.start_batch_training()
