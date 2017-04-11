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
            if InitData!=None:
                Weights=tf.constant(InitData)
            else:
                #Assume random weights if no InitType is given
                Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]))
    else:
        #Assume random weights if no InitType is given
        Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]))
    #Construct the bias for this layer
    if BiasData!=None:
        Biases = tf.constant(BiasData)
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
    Biases=tf.Variable(tf.zeros([HiddenUnits]),trainable=False)
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

def connect_input_to_network(InputsForLayers,Layers,ActFun=None,FunParam=None):
    #connect the outputs of the layer before to current layer
    if ActFun!=None:
        if ActFun=="sigmoid":
            Out=tf.nn.sigmoid(tf.matmul(InputsForLayers, Layers))
        elif ActFun=="tanh":
            Out=tf.nn.tanh(tf.matmul(InputsForLayers, Layers))
        elif ActFun=="relu":
            Out=tf.nn.relu(tf.matmul(InputsForLayers, Layers))
        elif ActFun=="relu6":
            Out=tf.nn.relu6(tf.matmul(InputsForLayers, Layers)) 
        elif ActFun=="crelu":
            Out=tf.nn.crelu(tf.matmul(InputsForLayers, Layers)) 
        elif ActFun=="elu":
            Out=tf.nn.elu(tf.matmul(InputsForLayers, Layers))     
        elif ActFun=="softplus":
            Out=tf.nn.softplus(tf.matmul(InputsForLayers, Layers))     
        elif ActFun=="dropout":
            Out=tf.nn.dropout(tf.matmul(InputsForLayers, Layers)) 
        elif ActFun=="bias_add":
            Out=tf.nn.bias_add(tf.matmul(InputsForLayers, Layers))     
    else:
        Out=tf.nn.sigmoid(tf.matmul(InputsForLayers, Layers))
    
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

def make_atomic_networks(Structures,NumberOfSameNetworks,Types=None,HiddenType=None,HiddenData=None,BiasData=None,ActFun=None,ActFunParam=None):
    
    AtomicNN=list()
    AllHiddenLayers=list()
    #make all the networks for the different atom types
    for i in range(0,len(Structures)):
        #Make hidden layers
        HiddenLayers=list()
        Structure=Structures[i]
        if HiddenData!=None:
            for j in range(1,len(Structure)):
                NrIn=Structure[j-1]
                NrHidden=Structure[j]
                HiddenLayers.append(construct_hidden_layer(NrIn,NrHidden,HiddenType,HiddenData[i][j-1],BiasData[i][j-1]))
        else:
            for j in range(1,len(Structure)):
                NrIn=Structure[j-1]
                NrHidden=Structure[j]
                HiddenLayers.append(construct_hidden_layer(NrIn,NrHidden,HiddenType,None,None))
                
        AllHiddenLayers.append(HiddenLayers)
     
        for k in range(0,NumberOfSameNetworks[i]):
            #Make input layer
            NrInputs=Structure[0]
            InputLayer=construct_input_layer(NrInputs)
            #Make output layer
            OutputLayer=construct_output_layer(Structure[-1])
            #Connect input to first hidden layer
            FirstWeights=HiddenLayers[0][0]
            FirstBiases=HiddenLayers[0][1]
            InConnection=connect_layers(InputLayer,FirstWeights,FirstBiases,ActFun,ActFunParam)
    
            for l in range(1,len(HiddenLayers)):
                #Connect ouput of in layer to second hidden layer
                if l==1 :
                    SecondWeights=HiddenLayers[l][0]
                    SecondBiases=HiddenLayers[l][1]
                    Network=connect_layers(InConnection,SecondWeights,SecondBiases,ActFun,ActFunParam)
                else:
                    Weights=HiddenLayers[l][0]
                    Biases=HiddenLayers[l][1]
                    Network=connect_layers(Network,Weights,Biases,ActFun,ActFunParam)
                    
            if Types!=None:
                AtomicNN.append([NumberOfSameNetworks[i],Network,InputLayer,OutputLayer,Types[i]])
            else:
                AtomicNN.append([NumberOfSameNetworks[i],Network,InputLayer,OutputLayer])

    return AtomicNN,AllHiddenLayers
    
def cost_per_atomic_network(TotalEnergy,AllEnergies,ReferenceValue):
    
    Costs=list()
    for Energy in AllEnergies:
        Costs.append((TotalEnergy-ReferenceValue)*Energy/TotalEnergy)
        
    return Costs

def cost_for_atomic_network(TotalEnergy,ReferenceValue,Ei):
    
    Cost=(TotalEnergy-ReferenceValue)**2*Ei/TotalEnergy
        
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

def train_step(Session,Optimizer,Layers,Data,CostFun):
    
    _,Cost=Session.run([Optimizer,CostFun],feed_dict={i: np.array(d) for i, d in zip(Layers,Data)})
    return Cost

def validate_step(Session,Layers,Data,CostFun):
    
    _,Cost=Session.run(CostFun,feed_dict={i: np.array(d) for i, d in zip(Layers,Data)})
    return Cost

def make_layers(InputLayer,OutputLayer=None):
    
    Layers=list()
    Layers.append(InputLayer)
    if OutputLayer!=None:
        Layers.append(OutputLayer)
    return Layers

def make_data(InData,OutData=None):
    
    Data=list()
    Data.append(InData)
    if OutData!=None:
        Data.append(OutData)
    
    return Data

def prepare_data_environment(InputLayer,InData,OutputLayer=None,OutData=None):
    
    Layers=make_layers(InputLayer,OutputLayer)
    Data=make_data(InData,OutData)
    
    return Layers,Data
    

def make_layers_for_atomicNNs(AtomicNNs,OutputLayer=None):
    
    Layers=list()
    for AtomicNetwork in AtomicNNs:
        Layers.append(AtomicNetwork[2])
    if OutputLayer!=None:
        Layers.append(OutputLayer)
    
    return Layers

def make_data_for_atomicNNs(InData,OutData=None):
    
    CombinedData=list()
    for Data in InData:
        CombinedData.append([Data])
    if OutData!=None:
        CombinedData.append([OutData])
    
    return CombinedData

def prepare_data_environment_for_atomicNNs(AtomicNNs,InData,OutputLayer=None,OutData=None):
    
    Layers=make_layers_for_atomicNNs(AtomicNNs,OutputLayer)
    Data=make_data_for_atomicNNs(InData,OutData)
    
    return Layers,Data
    
    
def train(Session,Optimizer,CostFun,Layers,TrainingData,Epochs,ValidationData=None,CostCriterium=None):
    #Train with specifications and return loss   
    TrainCost=list()
    ValidationCost=list()
    Session.run(tf.global_variables_initializer())
    for i in range(Epochs):
        Cost=train_step(Session,Optimizer,Layers,TrainingData,CostFun)
        TrainCost.append(Cost)
        #check validation dataset error
        if ValidationData!=None:
            ValidationCost.append(validate_step(Session,Layers,ValidationData,CostFun))
            
        if Cost<CostCriterium:
            break

                
    return Session,TrainCost,ValidationCost

def evaluate(Session,Network,Layers,Data):
    #Evaluate model for given input data
    if len(Layers)==1:
        Data=np.reshape(Data,(1,len(Data[0])))
        return Session.run(Network, feed_dict={Layers[0]:Data})
    else:
        return Session.run(Network, feed_dict={i: np.array(d) for i, d in zip(Layers,Data)})
        

def all_perms(elements):
    #Returns  all permutations of input

    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # nb elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]
  

def transform_permuted_data(PermutationData):
    #Transform data to an input vector format
    OutPermutedData=list()
    for Permutation in PermutationData:
        OutPermutation=[]
        for i in range(0,len(Permutation)-1):
            if i==0:
                OutPermutation=Permutation[i]
            Next=Permutaion[i+1]
            OutPermutation=OutPermutation+Next
    OutPermutedData.append(OutPermutation)
    
    return OutPermutedData
    

def output_of_all_atomic_networks(Session,AtomicNNs):

    TotalEnergy=0
    AllEnergies=list()
    
    for i in range(0,len(AtomicNNs)):
        #Get network data
        AtomicNetwork=AtomicNNs[i]
        Network=AtomicNetwork[1]
        if len(AtomicNetwork)>4:
            Type=AtomicNetwork[4]
        else:
            Type=None
            
        if i==0:
            offsetIdx=0
        else:
            #shift index by number of networks of last tpye
            offsetIdx+=AtomicNNs[i-1][0]
        #Get input data for network              
        AllEnergies.append(Network)
        TotalEnergy+=AllEnergies[-1]
        
    return TotalEnergy,AllEnergies

def atomic_cost_function(Session,AtomicNNs,ReferenceOutput):
    
    TotalEnergy,AllEnergies=output_of_all_atomic_networks(Session,AtomicNNs)
    Cost=total_cost_for_network(TotalEnergy,ReferenceOutput)[0]
    
    return Cost

def get_all_input_layers_as_single_input(AtomicNNs):
    for i in range(0,len(AtomicNNs)-1):
        ThisNetwork=AtomicNNs[i]
        NextNetwork=AtomicNNs[i+1]
        if i==0:
            Out=ThisNetwork[2]
            
        t1=NextNetwork[2]
        Out=tf.concat(1,[Out,t1])
            
    return Out

def get_data_for_specifc_networks(AtomicNNs,InData):
    #Return input permutation as input vectors for neural nets
    OutData=list()
    offset=0

    for i in range(0,len(AtomicNNs)):
        AtomicNetwork=AtomicNNs[i]
        NumberOfSameNetworks=AtomicNetwork[0]
        tempData=InData[offset:offset+NumberOfSameNetworks]
        offset+=NumberOfSameNetworks
        OutData.append(tempData)

    return OutData

def create_single_input_layer(AtomicNNs):
    
    for i in range(0,len(AtomicNNs)):
        if i==0:
            Out=AtomicNNs[i][2]
            print(AtomicNNs[i][2])
        else:
            Out=tf.concat([Out,AtomicNNs[i][2]], 1)

    return Out

def create_single_input_vector(AllData):
    
    AllInputs=[]
    for Data in AllData:
        AllInputs+=Data
        
    return [AllInputs]

def train_atomic_networks(AtomicNNs,TrainingInputs,TrainingOutputs,Epochs,LearningRate,ValidationInputs=None,ValidationOutputs=None,CostCriterium=None):
        
      
    ValidationCost=0
    TrainCost=0
    #Start Session
    Session = tf.InteractiveSession() 
    #Make virtual output layer for feeding the data to the cost function
    OutputLayer=construct_output_layer(1)
    #Prepare data environment for training
    Layers,Data=prepare_data_environment_for_atomicNNs(AtomicNNs,TrainingInputs,OutputLayer,TrainingOutputs)
    #Make validation input vector
    if ValidationInputs != None:
        ValidationData=make_data_for_atomicNNs(ValidationInputs,ValidationOutputs)
    else:
        ValidationData=None
    #Cost function changes for every net so the optimizer has to be adjusted
    CostFun=atomic_cost_function(Session,AtomicNNs,OutputLayer)
    Optimizer=tf.train.AdagradOptimizer(LearningRate).minimize(CostFun)
    #Start training of the atomic network
    Session,TrainCost,ValidationCost=train(Session,Optimizer,CostFun,Layers,Data,Epochs,ValidationData,CostCriterium)
    TrainedNetwork=tf.trainable_variables()
        
    return Session,TrainedNetwork,TrainCost,ValidationCost

def get_weights_biases_from_data(TrainedData):
    
    Weights=list()
    Biases=list()
    for i in range(0,len(TrainedData)):
        NetworkData=TrainedData[i]
        ThisWeights=list()
        ThisBiases=list()
        for j in range(0,len(NetworkData)):
            ThisWeights.append(NetworkData[j][0])
            ThisBiases.append(NetworkData[j][1])
        Weights.append(ThisWeights)
        Biases.append(ThisBiases)
    
    return Weights,Biases

def get_structure_from_data(TrainedData):
    
    Structures=list()
    for i in range(0,len(TrainedData)):
        ThisNetwork=TrainedData[i]
        Structure=[]
        for j in range(0,len(ThisNetwork)):
            Weights=ThisNetwork[j][0]
            Structure+=[Weights.shape[0]]
        Structure+=[1]
        Structures.append(Structure)
        
    return Structures

def expand_neuralnet(TrainedData,nAtoms):
    
    AtomicNNs=list()
    Structures=get_structure_from_data(TrainedData)
    Weights,Biases=get_weights_biases_from_data(TrainedData)
    AtomicNNs,_=make_atomic_networks(Structures,nAtoms,None,"custom",Weights,Biases,"tanh")
            
    return AtomicNNs

def get_trained_variables(Session,AllHiddenLayers):
    
    NetworkData=list()
    for HiddenLayers in AllHiddenLayers:
        Layers=list()
        for i in range(0,len(HiddenLayers)):
            Weights=Session.run(HiddenLayers[i][0])
            Biases=Session.run(HiddenLayers[i][1])
            Layers.append([Weights,Biases])
        NetworkData.append(Layers)
    return NetworkData

def evaluateAllAtomicNNs(Session,AtomicNNs,InData):
    
    Energy=0
    Layers,Data=prepare_data_environment_for_atomicNNs(AtomicNNs,InData,None,None)

    for i in range(0,len(AtomicNNs)):
        AtomicNetwork=AtomicNNs[i]   
        Energy+=evaluate(Session,AtomicNetwork[1],[Layers[i]],Data[i])
        
    return Energy
    
    