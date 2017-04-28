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
Data1=NN.DataInstance()

Data1.XYZfile="curve.xyz"
Data1.Logfile="md.curve"
Data1.SymmFunKeys=["1","2"]
Data1.NumberOfRadialFunctions=10
Data1.Lambs=[1.0,-1.0]
Data1.Zetas=np.arange(0.1,5,0.5).tolist()

Data1.read_files()
Batches1=Data1.get_data(10,70)
ValidationBatches1=Data1.get_data(10,30)


#Load second trainings data
Data2=NN.DataInstance()

Data2.XYZfile="NiAu_data_2AU1Ni.xyz"
Data2.Logfile="log.3atoms"
Data2.SymmFunKeys=["1","2"]
Data2.NumberOfRadialFunctions=10
Data2.Lambs=[1.0,-1.0]
Data2.Zetas=np.arange(0.1,5,0.5).tolist()

Data2.read_files()
Batches2=Data2.get_data(100,70)
ValidationBatches2=Data2.get_data(100,30)

#Train with first data 
NrAu=1
NrNi=1
Training=NN.AtomicNeuralNetInstance()
Training.Structures.append([Data1.SizeOfInputs[0],50,50,1])
Training.NumberOfSameNetworks.append(NrNi)
Training.Structures.append([Data1.SizeOfInputs[1],50,50,1])
Training.NumberOfSameNetworks.append(NrAu)
Training.HiddenType="truncated_normal"
Training.HiddenData=list()
Training.BiasData=list()
Training.ActFun="tanh"
Training.ActFunParam=None
Training.LearningRate=0.001
Training.CostCriterium=0.005
Training.Epochs=500
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.make_and_initialize_network()

#Start first training
Training.TrainingBatches=Batches1
Training.ValidationBatches=ValidationBatches1
Training.start_batch_training()

#Train with second data
NrAu=1
NrNi=2
Training2=NN.AtomicNeuralNetInstance()

NrHiddenOld=list()
NrHiddenOld.append(1)
NrHiddenOld.append(1)
Training2.Structures.append([Data1.SizeOfInputs[0],100,100,1])#the first to parameters can be anything
Training2.NumberOfSameNetworks.append(NrNi)
Training2.Structures.append([Data1.SizeOfInputs[1],100,100,1])
Training2.NumberOfSameNetworks.append(NrAu)
Training2.LearningRate=0.00001
Training2.CostCriterium=0.005
Training2.Epochs=1000
Training2.MakePlots=True
Training2.ActFun="tanh"
Training2.OptimizerType="Adam"
Training2.expand_existing_net()
Training2.TrainingBatches=Batches2
Training2.ValidationBatches=ValidationBatches2
Training2.start_batch_training()
