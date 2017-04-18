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

tf.reset_default_graph()

#make graph
Graph=tf.Graph()
    
with Graph.as_default():
    #Create network
    ThisInstance=NN.AtomicNeuralNetInstance()
    alpha=tf.placeholder(tf.float32, shape=[None,1])
    ThisInstance.HiddenType="truncated_normal"
    ThisInstance.HiddenData=None
    ThisInstance.BiasData=None
    ThisInstance.ActFun="tanh"
    ThisInstance.ActFunParam=None
    ThisInstance.LearningRate=0.0005
    ThisInstance.Epochs=1000
    NrNi=1
    NrAu=1
    ThisInstance.CostCriterium=0.00001
    #Parameters for Morse Potential
    R=5
    De=1
    Re=0.25
    a=4
    #Create input data
    steps=50
    dsIn=np.empty((steps,20))
    dsOut=np.empty((steps,1))
    for j in range(0,steps):
        r=float(R)*float(j)/float(steps)
        temp=np.cos(np.pi*np.arange(0,20)/20)
        dsIn[j][:]=r*temp
        
    ThisInstance.TrainingInputs.append(dsIn)
        
    dsIn=np.empty((steps,30))
    dsOut=np.empty((steps,1))
    for j in range(0,steps):
        r=float(R)*float(j)/float(steps)
        temp=np.cos(np.pi*np.arange(0,30)/30)
        dsIn[j][:]=r*temp
        dsOut[j]=De*(1-np.exp(-a*(r-Re)))**2
        
    ThisInstance.TrainingInputs.append(dsIn)
    ThisInstance.TrainingOutputs=dsOut
    #Create validation data
    steps=500
    dsIn=np.empty((steps,20))
    dsOut=np.empty((steps,1))
    for j in range(0,steps):
        r=float(R)*float(j)/float(steps)
        temp=np.cos(np.pi*np.arange(0,20)/20)
        dsIn[j][:]=r*temp
        
    ThisInstance.ValidationInputs.append(dsIn)
        
    dsIn=np.empty((steps,30))
    dsOut=np.empty((steps,1))
    for j in range(0,steps):
        r=float(R)*float(j)/float(steps)
        temp=np.cos(np.pi*np.arange(0,30)/30)
        dsIn[j][:]=r*temp
        dsOut[j]=De*(1-np.exp(-a*(r-Re)))**2
        
    ThisInstance.ValidationInputs.append(dsIn)
    ThisInstance.ValidationOutputs=dsOut
    
    #is list of size structures otherwise
    Types=None
    
    ThisInstance.Structures.append([20,500,1])
    G1=[]
    for i in range(0,20):
        G1+=[tf.sin(alpha)]
    ThisInstance.Gs.append(G1)
    ThisInstance.NumberOfSameNetworks.append(NrNi)
    ThisInstance.Structures.append([30,500,1])
    G2=[]
    for i in range(0,30):
        G2+=[tf.sin(alpha)]
    ThisInstance.Gs.append(G2)
    ThisInstance.NumberOfSameNetworks.append(NrAu)  
    ThisInstance.OptimizerType="Adagrad"

    ThisInstance.start_training_instance()
    Energy=NN.evaluateAllAtomicNNs(ThisInstance.Session,ThisInstance.AtomicNNs,ThisInstance.TrainingInputs)
    

    Session.close()
    #Energy after training
    #print(Energy)
    
    
    #compare results
    
    fig = figure(1)
    
    ax1 = fig.add_subplot(211)
    ax1.plot(ThisInstance.TrainingCosts)
    ax1.plot(ThisInstance.ValidationCosts)
    ax1.grid(True)
    #ax1.set_ylim((0, 10))
    
    ax2 = fig.add_subplot(212)
    r_plot=np.linspace(0,R,50)
    ax2.plot(r_plot,Energy)
    ax2.plot(r_plot,TrainingOutputs)
    ax2.grid(True)
    
    show(block=False)
    

with tf.Session(graph=Graph) as ThisInstance.Session:
    
    alphaVal=np.ones((1,1))*0.5
    #create new network with trained data
    ThisInstance.start_evaluation_instance()
    
    ThisInstance.Session.run(tf.global_variables_initializer())
    
    Energy=NN.evaluateAllAtomicNNs(ThisInstance.Session,ThisInstance.AtomicNNs,ThisInstance.TrainingInputs)
    #print(Energy)
    Force=NN.total_force(ThisInstance.Session,ThisInstance.AtomicNNs,alpha,alphaVal)
    #print(Force)
    #Energy after new creation of network
    #print(Energy)
    ThisInstance.Session.close()
    
