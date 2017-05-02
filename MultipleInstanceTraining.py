#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:08:14 2017

@author: alexf1991
"""

import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")


#Load first trainings data 
Training=NN.AtomicNeuralNetInstance()
Training2=NN.AtomicNeuralNetInstance()
#Training3=NN.AtomicNeuralNetInstance()

Training.XYZfile="curve.xyz"
Training.Logfile="md.curve"
Training.SymmFunKeys=["1","2"]
Training.NumberOfRadialFunctions=5
Training.Lambs=[1.0,-1.0]
Training.Zetas=np.arange(0.1,5,0.5).tolist()

Training.read_files()
Training.make_training_and_validation_data(100,70,30)
Training.NumberOfSameNetworks.append(1)
Training.NumberOfSameNetworks.append(1)


#Load second trainings data

Training2.XYZfile="NiAu_data_2AU1Ni.xyz"
Training2.Logfile="log.3atoms"
Training2.SymmFunKeys=["1","2"]
Training2.NumberOfRadialFunctions=5
Training2.Lambs=[1.0,-1.0]
Training2.Zetas=np.arange(0.1,5,0.5).tolist()
Training2.read_files()
Training2.make_training_and_validation_data(100,70,30)
Training2.NumberOfSameNetworks.append(2)
Training2.NumberOfSameNetworks.append(1)

Multi=NN.MultipleInstanceTraining()
Multi.EpochsPerCycle=5
Multi.GlobalEpochs=100
Multi.GlobalLearningRate=0.001
Multi.GlobalOptimizer="Adam"
Multi.GlobalRegularization="L2"
Multi.GlobalRegularizationParam=0.0001
Multi.MakePlots=True
Multi.GlobalStructures.append([Training.SizeOfInputs[0],50,50,1])
Multi.GlobalStructures.append([Training.SizeOfInputs[0],50,50,1])
Multi.TrainingInstances.append(Training)
Multi.TrainingInstances.append(Training2)
Multi.initialize_multiple_instances()
Multi.train_multiple_instances()