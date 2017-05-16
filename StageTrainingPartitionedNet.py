#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:11:23 2017

@author: alexf1991
"""

import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")


#Load first trainings data 
Training=NN.AtomicNeuralNetInstance()
Training2=NN.AtomicNeuralNetInstance()

Training.XYZfile="curve.xyz"
Training.Logfile="md.curve"
Training.SymmFunKeys=["1","2"]
Training.NumberOfRadialFunctions=5
Training.Lambs=[1.0,-1.0]
Training.Zetas=np.arange(0.1,5,0.5).tolist()
Training.Etas=np.arange(0.1,7,1).tolist()

Training.read_files(True)
Training.make_training_and_validation_data(100,70,30)



#Load second trainings data

Training2.XYZfile="NiAu_data_2AU1Ni.xyz"
Training2.Logfile="log.3atoms"
Training2.SymmFunKeys=["1","2"]
Training2.NumberOfRadialFunctions=5
Training2.Lambs=[1.0,-1.0]
Training2.Zetas=np.arange(0.1,5,0.5).tolist()
Training2.Etas=np.arange(0.1,7,1).tolist()

Training2.read_files()
Training2.make_training_and_validation_data(100,70,30)


#Train with first data 
NrAu=1
NrNi=1
MyStructure=NN.PartitionedStructure()
MyStructure.RadialNetworkStructure=[Training.TotalNrOfRadialFuns,5,5,1]
Training.Structures.append(MyStructure)
Training.NumberOfSameNetworks.append(NrNi)
Training.Structures.append(MyStructure)
Training.NumberOfSameNetworks.append(NrAu)
Training.LearningRate=0.001
Training.CostCriterium=0.0001
Training.Epochs=500
Training.MakePlots=True
Training.ActFun="elu"
Training.IsPartitioned=True
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.RegularizationParam=0.0001
Training.make_and_initialize_network()

#Start first training

Training.start_batch_training()


#Train with second data
MyStructure2=NN.PartitionedStructure()
MyStructure2.RadialNetworkStructure=[Training2.TotalNrOfRadialFuns,5,5,1]
MyStructure2.AngularNetworkStructure=[Training2.SizeOfInputs[0]-Training2.TotalNrOfRadialFuns,15,15,1]
#MyStructure2.CorrectionNetworkStructure=[Training2.SizeOfInputs[0],15,15,1]
Training2.Structures.append(MyStructure2)
Training2.Structures.append(MyStructure2)
Training2.IsPartitioned=True
Training2.LearningRate=0.0001
Training2.CostCriterium=0.00001
Training2.Epochs=150
Training2.MakePlots=True
Training2.OptimizerType="Adam"
Training2.ActFun="elu"

#Evaluate quality of learning transfer
NrAu=1
NrNi=1
Training2.NumberOfSameNetworks.append(NrNi)
Training2.NumberOfSameNetworks.append(NrAu)
Training2.expand_existing_net()
plt.ioff()
figure=plt.figure()
Batch=Training.get_data(1,100,True)
Training.TrainingInputs=Batch[-1][0]
Training2.TrainingInputs=Batch[-1][0]
plt.plot(Training.eval_step())
plt.plot(Training2.eval_step())
plt.show(block=False)
plt.ion()
#Train with second data
NrAu=1
NrNi=2
Training2.NumberOfSameNetworks[0]=NrNi
Training2.NumberOfSameNetworks[1]=NrAu
Training2.expand_existing_net()

Training2.start_batch_training()


NrAu=1
NrNi=1
Training2.NumberOfSameNetworks[0]=NrNi
Training2.NumberOfSameNetworks[1]=NrAu
Training2.expand_existing_net()

plt.ioff()
figure=plt.figure()
Training2.TrainingInputs=Batch[-1][0]
plt.plot(Training.eval_step())
plt.plot(Training2.eval_step())
plt.show(block=False)
plt.ion()

#Train with second data
NrAu=1
NrNi=2
Training2.NumberOfSameNetworks[0]=NrNi
Training2.NumberOfSameNetworks[1]=NrAu
Training2.expand_existing_net()

Training2.start_batch_training()