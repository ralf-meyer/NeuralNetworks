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

Training.XYZfile="NiAu_data.xyz"
Training.Logfile="log.md"
Training.Atomtypes=["1","2"]
Training.NumberOfRadialFunctions=6
#angular symmetry function settings
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]

Training.read_files(True)
Training.make_training_and_validation_data(100,70,30)



#Load second trainings data

Training2.XYZfile="NiAu_data_2AU1Ni.xyz"
Training2.Logfile="log.3atoms"
Training2.Atomtypes=["1","2"]
Training2.NumberOfRadialFunctions=6
#angular symmetry function settings
Training2.Lambs=[1.0,-1.0]
Training2.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training2.Etas=[0.1]

Training2.read_files()
Training2.make_training_and_validation_data(100,70,30)


#Train with first data 
NrAu=1
NrNi=1
MyStructure=NN.PartitionedStructure()
MyStructure.ForceFieldNetworkStructure=[Training.SizeOfInputs[0],10,10,1]
Training.Structures.append(MyStructure)
Training.NumberOfAtomsPerType.append(NrNi)
Training.Structures.append(MyStructure)
Training.NumberOfAtomsPerType.append(NrAu)
Training.LearningRate=0.001
Training.CostFunType="Adaptive_2"
Training.dE_Criterium=0.03
Training.Epochs=1500
Training.MakePlots=True
Training.ActFun="elu"
Training.IsPartitioned=True
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.RegularizationParam=0.0001
Training.LearningRateType="exponential_decay"
Training.LearningRateDecaySteps=1000
Training.SavingDirectory="save_stage_1"
Training.make_and_initialize_network()

#Start first training

Training.start_batch_training()


#Train with second data
MyStructure2=NN.PartitionedStructure()
MyStructure2.ForceFieldNetworkStructure=[Training.SizeOfInputs[0],10,10,1]
MyStructure2.CorrectionNetworkStructure=[Training.SizeOfInputs[0],80,80,15,1]
#MyStructure2.CorrectionNetworkStructure=[Training2.SizeOfInputs[0],15,15,1]
Training2.Structures.append(MyStructure2)
Training2.Structures.append(MyStructure2)
Training2.IsPartitioned=True
Training2.LearningRate=0.001
Training2.CostFunType="Adaptive_2"
Training2.dE_Criterium=0.03
Training2.Epochs=1500
Training2.MakePlots=True
Training2.OptimizerType="Adam"
Training2.ActFun="elu"
Training2.Regularization="L2"
Training2.RegularizationParam=0.0001
Training2.LearningRateType="exponential_decay"
Training2.LearningRateDecaySteps=1000
Training2.SavingDirectory="save_stage_2"
NrAu=1
NrNi=2
Training2.NumberOfAtomsPerType.append(NrNi)
Training2.NumberOfAtomsPerType.append(NrAu)
Training2.expand_existing_net(ModelName="save_stage_1/trained_variables")

Training2.start_batch_training()

