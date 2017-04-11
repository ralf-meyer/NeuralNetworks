#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:59:22 2017

@author: alexf1991
"""

import NeuralNetworkUtilities as NN
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import figure, show
import random 
import warnings
warnings.filterwarnings('ignore')


#Create network
Structures=list()
NrSame=list()
HiddenType="truncated_normal"
HiddenData=None
BiasData=None
ActFun="tanh"
ActFunParam=None
LearningRate=0.01
Epochs=1000
NrNi=5
NrAu=5
CostCriterium=0.001
TrainingInputs=list()
TrainingOutputs=list()
#Create input data
steps=20*NrNi+30*NrAu
for i in range(0,NrNi):
    dsIn=[]
    rnd1=random.random()*10
    rnd2=random.random()*10
    for j in range(i*20,20*(i+1)):
        val=np.sin(rnd1*np.pi*j/steps)*np.sin(rnd2*np.pi*j/steps)
        dsIn=dsIn+[val]
    TrainingInputs.append(dsIn)
    
for i in range(0,NrAu):
    dsIn=[]
    rnd1=random.random()*10
    rnd2=random.random()*10
    for j in range(20*NrNi+i*30,20*NrNi+(i+1)*30):
        val=np.sin(rnd1*np.pi*j/steps)*np.sin(rnd2*np.pi*j/steps)
        dsIn=dsIn+[val]
    TrainingInputs.append(dsIn)
    
TrainingOutputs.append(5)

#is list of size structures otherwise
Types=None

Structures.append([20,50,50,1])
NrSame.append(NrNi)
Structures.append([30,50,50,1])
NrSame.append(NrAu)  
AtomicNNs,AllHiddenLayers=NN.make_atomic_networks(Structures,NrSame,Types,HiddenType,HiddenData,BiasData,ActFun,ActFunParam)
Session,TrainedNetwork,TrainCosts,ValidationCosts=NN.train_atomic_networks(AtomicNNs,TrainingInputs,TrainingOutputs,Epochs,LearningRate,None,None,CostCriterium)
Energy=NN.evaluateAllAtomicNNs(Session,AtomicNNs,TrainingInputs)
TrainedData=NN.get_trained_variables(Session,AllHiddenLayers)


tf.reset_default_graph()
Session.close()
#Energy after training
print(Energy)


#compare results

fig = figure(1)

ax1 = fig.add_subplot(211)
ax1.plot(TrainCosts)
ax1.grid(True)
ax1.set_ylim((0, 2))



show(block=False)

#create new network with trained data
AtomicNNs=NN.expand_neuralnet(TrainedData,NrSame)
Session = tf.InteractiveSession() 
Session.run(tf.global_variables_initializer())

Energy=NN.evaluateAllAtomicNNs(Session,AtomicNNs,TrainingInputs)
#Energy after new creation of network
print(Energy)
Session.close()
