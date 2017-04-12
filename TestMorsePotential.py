#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:31:46 2017

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
LearningRate=0.001
Epochs=20000
NrNi=1
NrAu=1
CostCriterium=0.00001
TrainingInputs=list()
TrainingOutputs=list()
#Parameters for Morse Potential
R=5
De=1
Re=0.25
a=4
#Create input data
steps=500
dsIn=np.empty((steps,20))
dsOut=np.empty((steps,1))
for j in range(0,steps):
    r=float(R)*float(j)/float(steps)
    temp=np.cos(np.pi*np.arange(0,20)/20)
    dsIn[j][:]=r*temp
    
TrainingInputs.append(dsIn)
    
dsIn=np.empty((steps,30))
dsOut=np.empty((steps,1))
for j in range(0,steps):
    r=float(R)*float(j)/float(steps)
    temp=np.cos(np.pi*np.arange(0,30)/30)
    dsIn[j][:]=r*temp
    dsOut[j]=De*(1-np.exp(-a*(r-Re)))**2
    
TrainingInputs.append(dsIn)
TrainingOutputs=dsOut

#is list of size structures otherwise
Types=None

Structures.append([20,10,10,1])
NrSame.append(NrNi)
Structures.append([30,10,10,1])
NrSame.append(NrAu)  
OptimizerType="Adagrad"
AtomicNNs,AllHiddenLayers=NN.make_atomic_networks(Structures,NrSame,Types,HiddenType,HiddenData,BiasData,ActFun,ActFunParam)
Session,TrainedNetwork,TrainCosts,ValidationCosts=NN.train_atomic_networks(AtomicNNs,TrainingInputs,TrainingOutputs,Epochs,LearningRate,None,None,CostCriterium,OptimizerType)
Energy=NN.evaluateAllAtomicNNs(Session,AtomicNNs,TrainingInputs)
TrainedData=NN.get_trained_variables(Session,AllHiddenLayers)


tf.reset_default_graph()
Session.close()
#Energy after training
#print(Energy)


#compare results

fig = figure(1)

ax1 = fig.add_subplot(211)
ax1.plot(TrainCosts)
ax1.grid(True)
#ax1.set_ylim((0, 10))

ax2 = fig.add_subplot(212)
r_plot=np.linspace(0,R,steps)
ax2.plot(r_plot,Energy)
ax2.plot(r_plot,TrainingOutputs)
ax2.grid(True)




show(block=False)

#create new network with trained data
AtomicNNs=NN.expand_neuralnet(TrainedData,NrSame)
Session = tf.InteractiveSession() 
Session.run(tf.global_variables_initializer())

Energy=NN.evaluateAllAtomicNNs(Session,AtomicNNs,TrainingInputs)
#Energy after new creation of network
#print(Energy)
Session.close()
