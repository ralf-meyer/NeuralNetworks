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
Training2=NN.AtomicNeuralNetInstance()

Training.XYZfile="2ClusterNiAu_data.xyz"
Training.Logfile="2cluster.md"
Training.atomtypes=["1","2"]
Training.NumberOfRadialFunctions=7
#angular symmetry function settings
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]

Training.read_files(True)

Training.make_training_and_validation_data(100,70,30)



#Load second trainings data

Training2.TrainingBatches=Training.TrainingBatches
Training2.ValidationBatches=Training.ValidationBatches


#Train with first data 
NrAu=12
NrNi=14

Training.Structures.append([Training.SizeOfInputs[0],100,80,1])
Training.Dropout=[0.25,0.1,0]
Training.NumberOfAtomsPerType.append(NrNi)
Training.Structures.append([Training.SizeOfInputs[1],100,80,1])
Training.NumberOfAtomsPerType.append(NrAu)
Training.HiddenType="truncated_normal"
Training.HiddenData=list()
Training.BiasData=list()
Training.ActFun="elu"
Training.ActFunParam=None
Training.LearningRate=0.001
Training.CostCriterium=0
Training.Epochs=5000
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.RegularizationParam=0.0001
Training.MakeLastLayerConstant=True
Training.make_and_initialize_network()

#Start first training

Training.start_batch_training()


#Train with second data

Training2.Structures.append([Training.SizeOfInputs[0],100,80,20,1])#the first to parameters can be anything
Training2.Structures.append([Training.SizeOfInputs[1],100,80,20,1])
Training2.Dropout=[0,0,0]
Training2.LearningRate=0.001
Training2.CostCriterium=0
Training2.Epochs=1000
Training2.MakePlots=True
Training2.ActFun="elu"
Training2.OptimizerType="Adam"
Training2.MakeLastLayerConstant=True
Training2.MakeAllVariable=False

#Evaluate quality of learning transfer
NrAu=12
NrNi=14
Training2.NumberOfAtomsPerType.append(NrNi)
Training2.NumberOfAtomsPerType.append(NrAu)
Training2.expand_existing_net()
Training2.start_batch_training()
