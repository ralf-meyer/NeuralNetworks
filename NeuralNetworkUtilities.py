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
        if i==1 and HiddenData=="fix_first":
            HiddenLayers.append(construct_not_trainable_layer(NrIn,NrHidden))
        else:
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

def make_atomic_training_networks(Structures,NumberOfSameNetworks,Types=None,HiddenType=None,HiddenData=None,BiasData=None,ActFun=None,ActFunParam=None):
    
    AtomicNN=list()
    #make all the networks for the different atom types
    for i in range(0,len(Structures)):
        Network,InputLayer,OutputLayer=make_standard_neuralnetwork(Structures[i],HiddenType=None,HiddenData=None,BiasData=None,ActFun=None,ActFunParam=None)
        #CostFunction=cost_function(Network,OutputLayer)
        #Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(CostFunction)
        if Types!=None:
            AtomicNN.append([NumberOfSameNetworks[i],Network,InputLayer,OutputLayer,Types[i]])
        else:
            AtomicNN.append([NumberOfSameNetworks[i],Network,InputLayer,OutputLayer])
    
    return AtomicNN
    
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

def train_step(Session,Optimizer,InputLayer,OutputLayer,TrainInputs,TrainOutputs,CostFun):
    _,Cost=Session.run([Optimizer,CostFun],feed_dict={InputLayer: np.array(TrainInputs),OutputLayer: np.array(TrainOutputs)})
    return Cost
     
def train(Session,Optimizer,CostFun,InputLayer,OutputLayer,TrainInputs,TrainOutputs,Epochs,ValidationInputs=None,ValidationOutputs=None):
    #Train with specifications and return loss   
    TrainCost=list()
    ValidationCost=list()
    Session.run(tf.global_variables_initializer())
    for i in range(Epochs):
        Cost=train_step(Session,Optimizer,InputLayer,OutputLayer,TrainInputs,TrainOutputs,CostFun)
        TrainCost.append(Cost)
        #check validation dataset error
        if ValidationInputs!=None:
            ValidationCost.append(Session.run(CostFun, feed_dict={InputLayer:np.array(ValidationInputs),OutputLayer: np.array(ValidationOutputs)}))

                
    return Session,TrainCost,ValidationCost

def evaluate(Session,Network,InputLayer,Data):
    #Evaluate model for given input data
    return Session.run(Network, feed_dict={InputLayer:np.array(Data)})[0]

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
            Next=Permuation[i+1]
            OutPermutation=OutPermutation+Next
    OutPermutedData.append(OutPermutation)
    
    return OutPermutedData
    

def evaluate_all_atomic_networks(Session,AtomicNNs,Data):

    TotalEnergy=0
    AllEnergies=list()
    
    for i in range(0,len(AtomicNNs)):
        #Get network data
        AtomicNetwork=AtomicNNs[i]
        Network=AtomicNetwork[1]
        InputLayer=AtomicNetwork[2]
        OutputLayer=AtomicNetwork[3]
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
        AllEnergies.append(evaluate(Session,Network,InputLayer,Data))
        TotalEnergy+=AllEnergies[-1]
        
    return TotalEnergy,AllEnergies

def atomic_cost_function(Session,AtomicNNs,Inputs,ReferenceOutput,NetworkIdx):
    
    TotalEnergy,AllEnergies=output_of_all_atomic_networks(Session,AtomicNNs,Inputs)
    Cost=cost_for_atomic_network(TotalEnergy,ReferenceOutput,AllEnergies[NetworkIdx])
    
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

def train_atomic_networks(AtomicNNs,TrainingInputs,TrainingOutputs,Epochs,LearningRate,ValidationInputs=None,ValidationOutputs=None):
        
    TrainCosts=list()
    ValidationCosts=list()
    TrainedNetworks=list()
    #create datasets for training and validation
    SortedInData=get_data_for_specifc_networks(AtomicNNs,TrainingInputs)
    SortedOutData=get_data_for_specifc_networks(AtomicNNs,TrainingOutputs)
    if ValidationInputs != None:
        SortedValInData=get_data_for_specifc_networks(AtomicNNs,ValidationInputs)
        SortedValOutData=get_data_for_specifc_networks(AtomicNNs,ValidationOutputs)
        
    ValidationCost=0
    TrainCost=0
    #Start Session
    Session = tf.InteractiveSession() 
    
    for j in range(0,len(AtomicNNs)):
        #Get data for network
        InDataForNetwork=SortedInData[j]
        OutDataForNetwork=SortedOutData[j]
        if ValidationInputs != None:
            ValInDataForNetwork=SortedValInData[j]
            ValOutDataForNetwork=SortedValOutData[j]
        else:
            ValInDataForNetwork=None
            ValOutDataForNetwork=None
        #Get network data
        AtomicNetwork=AtomicNNs[j]
        Network=AtomicNetwork[1]
        InputLayer=AtomicNetwork[2]
        OutputLayer=AtomicNetwork[3]
 
        #Cost function changes for every net so the optimizer has to be adjusted
        CostFun=cost_function(Network,OutputLayer)
        Optimizer=tf.train.GradientDescentOptimizer(LearningRate).minimize(CostFun)
        #Start training of the atomic network
        Session,TrainCost,ValidationCost=train(Session,Optimizer,CostFun,InputLayer,OutputLayer,InDataForNetwork,OutDataForNetwork,Epochs,ValInDataForNetwork,ValOutDataForNetwork)
        TrainedNetworks.append(tf.trainable_variables())
        #Store costs per epoche for each network
        TrainCosts.append(TrainCost)    
        ValidationCosts.append(ValidationCost)
        
    return Session,TrainedNetworks,TrainCosts,ValidationCosts

def expand_neuralnet(Structures,TrainedNetworks,nAtoms):
    
    AllAtomicNNs=list()
    InputLayers=list()
    for i in range(0,len(Structures)):
        InputUnits=Structures[i][0]
        Network=TrainedNetworks[i]
        for j in range(0,nAtoms[i]):
            InputLayer=construct_input_layer(InputUnits)
            print(InputLayer)
            print(Network)
            AllAtomicNNs.append(connect_input_to_network(InputLayer,Network))
            InputLayers.append(InputLayer)
            
    return AllAtomicNNs,InputLayers

def evaluateAllAtomicNNs(Session,AllAtomicNNs,InputLayers,Data):
    
    Energy=0
    for i in range(0,len(AllAtomicNNs)):
        Energy+=evaluate(Session,AllAtomicNNs[i],InputLayers[i],Data[i])
        
    return Energy
    
    