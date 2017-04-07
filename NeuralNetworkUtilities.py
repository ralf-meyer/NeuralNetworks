#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:30:10 2017

@author: Fuchs Alexander
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def construct_input_layer(InputUnits):
    #Construct inputs for the NN
    Inputs=tf.placeholder(tf.float32, shape=[None, InputUnits])
    
    return Inputs

def construct_hidden_layer(LayerBeforeUnits,HiddenUnits,InitType=None,InitData=None,BiasData=None):
    #Construct the weights for this layer
    if InitType!=None:
        if InitType == "zeros":
            Weights=tf.Variable(tf.zeros([LayerBeforeUnits,HiddenUnits]))
        elif InitType =="ones":
            Weights=tf.Variable(tf.ones([LayerBeforeUnits,HiddenUnits]))
        elif InitType == "fill":
            Weights=tf.Variable(tf.fill([LayerBeforeUnits,HiddenUnits],InitData))
        elif InitType == "random_normal":
            Weights=tf.Variable(tf.random_normal([LayerBeforeUnits,HiddenUnits]))
        elif InitType == "truncated_normal":
            Weights=tf.Variable(tf.truncated_normal([LayerBeforeUnits,HiddenUnits]))
        elif InitType == "random_uniform":
            Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]))
        elif InitType == "random_shuffle":
            Weights=tf.Variable(tf.random_shuffle([LayerBeforeUnits,HiddenUnits]))
        elif InitType == "random_crop":
            Weights=tf.Variable(tf.random_crop([LayerBeforeUnits,HiddenUnits],InitData))
        elif InitType == "random_gamma":
            Weights=tf.Variable(tf.random_gamma([LayerBeforeUnits,HiddenUnits],InitData))
        else:
            if InitData!=None and InitData.shape==[LayerBeforeUnits,HiddenUnits]:
                Weights=tf.Variable(InitData)
            else:
                #Assume random weights if no InitType is given
                Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]))
    else:
        #Assume random weights if no InitType is given
        Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]))
    #Construct the bias for this layer
    if BiasData!=None and BiasData.shape==[HiddenUnits]:
        Biases = tf.Variable(BiasData)
    else:
        Biases = tf.Variable(tf.zeros([HiddenUnits]))
        
    return Weights,Biases

def construct_output_layer(OutputUnits):
    #Construct the output for the NN
    Outputs = tf.placeholder(tf.float32, shape=[None, OutputUnits])
    
    return Outputs

def construct_not_trainable_layer(NrInputs,NrOutputs):
    #make a not trainable layer with the weights one
    Weights=tf.Variable(tf.ones([NrInputs,NrOutputs]), trainable=False)
    Biases=tf.Variable(tf.zeros([HiddenUnits]))
    return Weights,Biases
    
def connect_layers(InputsForLayer,Layer1Weights,Layer1Bias,ActFun=None,FunParam=None):
    #connect the outputs of the layer before to current layer
    if ActFun!=None:
        if ActFun=="sigmoid":
            Out=tf.nn.sigmoid(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="tanh":
            Out=tf.nn.tanh(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="relu":
            Out=tf.nn.relu(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)  
        elif ActFun=="relu6":
            Out=tf.nn.relu6(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias) 
        elif ActFun=="crelu":
            Out=tf.nn.crelu(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias) 
        elif ActFun=="elu":
            Out=tf.nn.elu(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)     
        elif ActFun=="softplus":
            Out=tf.nn.softplus(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)     
        elif ActFun=="dropout":
            Out=tf.nn.dropout(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias,FunParam) 
        elif ActFun=="bias_add":
            Out=tf.nn.bias_add(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias,FunParam)     
    else:
        Out=tf.nn.sigmoid(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
    
    return Out

def connect_networks(InputsForLayer,Layer1Weights,Layer1Bias,ActFun=None,FunParam=None):
    #connect different NNs
    if ActFun!=None:
        if ActFun=="sigmoid":
            Out=tf.nn.sigmoid(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="tanh":
            Out=tf.nn.tanh(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="relu":
            Out=tf.nn.relu(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)  
        elif ActFun=="relu6":
            Out=tf.nn.relu6(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias) 
        elif ActFun=="crelu":
            Out=tf.nn.crelu(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias) 
        elif ActFun=="elu":
            Out=tf.nn.elu(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)     
        elif ActFun=="softplus":
            Out=tf.nn.softplus(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)     
        elif ActFun=="dropout":
            Out=tf.nn.dropout(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias,FunParam) 
        elif ActFun=="bias_add":
            Out=tf.nn.bias_add(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias,FunParam)     
    else:
        Out=tf.nn.sigmoid(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        
    return Out
    

def make_standard_neuralnetwork(Structure,HiddenType=None,HiddenData=None,BiasData=None,ActFun=None,ActFunParam=None):
    #Construct the NN
    
    #Make inputs
    NrInputs=Structure[0]
    InputLayer=construct_input_layer(NrInputs)
    #Make hidden layers
    HiddenLayers=list()
    for i in range(1,len(Structure)):
        NrIn=Structure[i-1]
        NrHidden=Structure[i]
        HiddenLayers.append(construct_hidden_layer(NrIn,NrHidden,HiddenType,HiddenData,BiasData))
        

    #Make output layer
    OutputLayer=construct_output_layer(Structure[-1])

   #Connect input to first hidden layer
    FirstWeights=HiddenLayers[0][0]
    FirstBiases=HiddenLayers[0][1]
    InConnection=connect_layers(InputLayer,FirstWeights,FirstBiases,ActFun,ActFunParam)
    

    for j in range(1,len(HiddenLayers)):
       #Connect ouput of in layer to second hidden layer
        if j==1 :
            SecondWeights=HiddenLayers[j][0]
            SecondBiases=HiddenLayers[j][1]
            LastConnection=connect_layers(InConnection,SecondWeights,SecondBiases,ActFun,ActFunParam)
        else:
            Weights=HiddenLayers[j][0]
            Biases=HiddenLayers[j][1]
            LastConnection=connect_layers(LastConnection,Weights,Biases,ActFun,ActFunParam)
    
               
    
    return LastConnection,InputLayer,OutputLayer

def make_atomic_neuralnetworks(Structures,LearningRate,NumberOfSameNetworks,Types=None,HiddenType=None,HiddenData=None,BiasData=None,ActFun=None,ActFunParam=None):
    
    AtomicNN=list()
    #make all the networks for the different atom types
    for i in range(0,len(Structures)):
        Network,InputLayer,OutputLayer=make_standard_neuralnetwork(Structures[i],HiddenType=None,HiddenData=None,BiasData=None,ActFun=None,ActFunParam=None)
        #CostFunction=cost_function(Network,OutputLayer)
        #Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(CostFunction)
        if Types!=None:
            AtomicNN.append([NumberOfSameNetworks,Network,InputLayer,OutputLayer,Types[i]])
        else:
            AtomicNN.append([NumberOfSameNetworks,Network,InputLayer,OutputLayer])
    
    return AtomicNN
    
def cost_per_atomic_network(TotalEnergy,AllEnergies,ReferenceValue):
    
    Costs=list()
    for Energy in AllEnergies:
        Costs.append((TotalEnergy-ReferenceValue)*Energy/TotalEnergy)
        
    return Costs

def cost_for_atomic_network(TotalEnergy,ReferenceValue,Ei):
    
    Cost=(TotalEnergy-ReferenceValue)*Ei/TotalEnergy
        
    return Cost

def total_cost_for_network(TotalEnergy,ReferenceValue):
    
    return (TotalEnergy-ReferenceValue)**2/2

def cost_function(Network,Output,CostFunType=None,RegType=None,RegParam=None):
    #define a cost function for the NN   
    if CostFunType != None:
        if CostFunType=="error":
            CostFunction = 0.5 * tf.reduce_sum(tf.subtract(Network, Output) * tf.subtract(Network, Output))   
    else:
        CostFunction = 0.5 * tf.reduce_sum(tf.subtract(Network, Output) * tf.subtract(Network, Output))   
    #to be expanded
    return CostFunction
     
def train(Session,Optimizer,CostFun,InputLayer,OutputLayer,TrainInputs,TrainOutputs,Epochs,ValidationInputs=None,ValidationOutputs=None):
    #Train with specifications and return loss   
    TrainCost=list()
    ValidationCost=list()
    Session.run(tf.global_variables_initializer())
    for i in range(Epochs):
        _,cost=Session.run([Optimizer,CostFun],feed_dict={InputLayer: np.array(TrainInputs),OutputLayer: np.array(TrainOutputs)})
        TrainCost.append(cost)
        #check validation dataset error
        if ValidationInputs!=None:
            ValidationCost.append(Session.run(CostFun, feed_dict={InputLayer:np.array(ValidationInputs),OutputLayer: np.array(ValidationOutputs)}))

                
    return Session,TrainCost,ValidationCost

def evaluate(Session,Network,InputLayer,Data):
    #Evaluate model for given input data
    val=np.array(Data).reshape(1,len(Data))
    return Session.run(Network, feed_dict={InputLayer:val})[0]

def evaluate_all_atomic_networks(Session,AtomicNNs,InData):
    #maybe make parallel in later versions
    TotalEnergy=0
    AllEnergies=list()
    
    for i in range(0,len(AtomicNNs)):
        #Get network data
        AtomicNetwork=AtomicNNs[i]
        NumberOfSameNetworks=AtomicNetwork[0]
        Network=AtomicNetwork[1]
        InputLayer=AtomicNetwork[2]
        OutputLayer=AtomicNetwork[3]
        if len(AtomicNetwork)>3:
            Type=AtomicNetwork[4]
        else:
            Type=None
        #Get input data for network
        for j in range(0,NumberOfSameNetworks):
            if i==0:
                offsetIdx=0
            else:
                #shift index by number of networks of last tpye
                offsetIdx=AtomicNNs[i-1][0]
                
            Data=InData[offsetIdx+j]
            AllEnergies.append(evaluate(Session,Network,InputLayer,Data))
            TotalEnergy+=AllEnergies[-1]
        
    return TotalEnergy,AllEnergies

def atomic_cost_function(Session,AtomicNNs,Inputs,ReferenceOutput,NetworkIdx):
    
    TotalEnergy,AllEnergies=evaluate_all_atomic_networks(Session,AtomicNNs,Inputs)
    Cost=cost_for_atomic_network(TotalEnergy,ReferenceOutput,AllEnergies[NetworkIdx])
    
    return Cost

def train_atomic_networks(Session,AtomicNNs,TrainingInputs,TrainingOutputs,Epochs,LearningRate,ValidationInputs=None,ValidationOutputs=None):
    
    Session.run(tf.global_variables_initializer())
    
    TrainCosts=list()
    ValidateCosts=list()
    
    for i in range(Epochs):

        
        for j in range(0,len(AtomicNNs)):
        #Get network data
            AtomicNetwork=AtomicNNs[j]
            NumberOfSameNetworks=AtomicNetwork[0]
            Network=AtomicNetwork[1]
            InputLayer=AtomicNetwork[2]
            InputLayerSize=InputLayer.shape[1]
            OutputLayer=AtomicNetwork[3]
            if len(AtomicNetwork)>3:
                Type=AtomicNetwork[4]
            else:
                Type=None
            #Get input data for network
            if j==0:
                offsetIdx=0
                #Initialize indizes for input extraction    
                StartIdx=0
                EndIdx=InputLayerSize
            else:
                #shift index by number of networks of last tpye==>number of the atomic network
                offsetIdx+=AtomicNNs[j-1][0]
            

                
            #Because OutputLayer belongs to the atomic NN we need an overall output for the total energy
            TempOut=tf.placeholder(tf.float32, shape=[None, 1])
            #Train for each network of same type
            for k in range(0,NumberOfSameNetworks):
                #Cost function changes for every net so the optimizer has to be adjusted
                Inputs=TrainingInputs
                #make part of input vector variable
                Inputs[:][StartIdx:EndIdx]=InputLayer
                
                CostFun=atomic_cost_function(Session,AtomicNNs,Inputs,TempOut,offsetIdx+k)
                Optimizer=tf.train.GradientDescentOptimizer(LearningRate).minimize(CostFun)
                Session.run(Optimizer,feed_dict={InputLayer: np.array(TrainingInputs[:][StartIdx:EndIdx]),TempOut: np.array(TrainingOutputs)})
                #shift indizes for next network                    
                StartIdx+=InputLayerSize
                if k<NumberOfSameNetworks-1:
                    EndIdx+=InputLayerSize
                else:
                    if j< len(AtomicNNs)-1:
                        NextLayerSize=AtomicNNs[j+1][2].shape[1]
                        EndIdx+=NextLayerSize
                        
        #get cost for epoch for training data
        TrainCost=0
        for x in range(0,len(TrainingInputs)):
            Input=TrainingInputs[x]
            Output=TrainingOutputs[x]
            TotalEnergy,AllEnergies=evaluate_all_atomic_networks(Session,AtomicNNs,Input)
            if x==0:
                TrainCost=total_cost_for_network(TotalEnergy,Output)
            else:
                TrainCost=(TrainCost+total_cost_for_network(TotalEnergy,Output))/2
                
        TrainCosts.append(TrainCost)
            
        #get cost for epoch for training data
        ValidateCost=0
        for x in range(0,len(TrainingInputs)):
            Input=ValidationInputs[x]
            Output=ValidationOutputs[x]
            TotalEnergy,AllEnergies=evaluate_all_atomic_networks(Session,AtomicNNs,Input)
            if x==0:
                ValidateCost=total_cost_for_network(TotalEnergy,Output)
            else:
                ValidateCost=(ValidateCost+total_cost_for_network(TotalEnergy,Output))/2   
            
        ValidateCosts.append(ValidateCost)
        
        return Session,TrainCosts,ValidateCosts
        
    
    