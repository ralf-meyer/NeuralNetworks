#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:30:10 2017

@author: Fuchs Alexander
"""
import numpy as np
import tensorflow as tf


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
    Biases=tf.Variable(tf.zeros([NrOutputs]),trainable=False)
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

def make_force_networks(Structure,HiddenData,BiasData):
    
    ForceNetworks=list()
    for i in range(1,len(Structure)-1):
        Network,InputLayer,OutputLayer=make_standard_neuralnetwork(Structure[0:i+1],None,HiddenData,BiasData)
        ForceNetworks.append([Network,InputLayer,OutputLayer])        
    return ForceNetworks

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
        HiddenLayers.append(construct_hidden_layer(NrIn,NrHidden,HiddenType,HiddenData[i-1],BiasData[i-1]))
        

    #Make output layer
    OutputLayer=construct_output_layer(Structure[-1])

   #Connect input to first hidden layer
    FirstWeights=HiddenLayers[0][0]
    FirstBiases=HiddenLayers[0][1]
    InConnection=connect_layers(InputLayer,FirstWeights,FirstBiases,ActFun,ActFunParam)
    LastConnection=InConnection

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
    
    Cost=Session.run(CostFun,feed_dict={i: np.array(d) for i, d in zip(Layers,Data)})
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
        CombinedData.append(Data)
    if OutData!=None:
        CombinedData.append(OutData)

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
        TrainCost.append(sum(Cost))

        #check validation dataset error
        if ValidationData!=None:
            ValidationCost.append(sum(validate_step(Session,Layers,ValidationData,CostFun)))
            
        if TrainCost[-1]<CostCriterium and Cost!=0:
            print(TrainCost[-1])
            break

                
    return Session,TrainCost,ValidationCost

def evaluate(Session,Network,Layers,Data):
    #Evaluate model for given input data
    if len(Layers)==1:
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
            Next=Permutation[i+1]
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
    Cost=total_cost_for_network(TotalEnergy,ReferenceOutput)

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
        else:
            Out=tf.concat([Out,AtomicNNs[i][2]], 1)

    return Out

def create_single_input_vector(AllData):
    
    AllInputs=[]
    for Data in AllData:
        AllInputs+=Data
        
    return [AllInputs]


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

def expand_neuralnet(TrainedData,nAtoms,Gs):
    
    AtomicNNs=list()
    Structures=get_structure_from_data(TrainedData)
    Weights,Biases=get_weights_biases_from_data(TrainedData)
    AtomicNNs,_=make_atomic_networks(Structures,nAtoms,Gs,"custom",Weights,Biases,"tanh")
            
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

def all_derivatives_of_Gij_wrt_alpha(Session,GsForAtom,Alpha,AlphaValue):
    
    Derivatives=list()
    for i in range(0,len(GsForAtom)):
        Derivatives.append(1)#not correct yet


    return Derivatives
    

def all_derivatives_of_Ei_wrt_Gij(Session,InputLayer,Network,G_values):
    
    Derivatives=tf.gradients(Network,InputLayer)
    DerivativeValues=Session.run(Derivatives,feed_dict={InputLayer:G_values})[0]
        
    return DerivativeValues

def get_g_values(Session,GsForAtom,Alpha,AlphaVal):
    
    G_values=np.empty((1,len(GsForAtom)))
    for i in range(0,len(GsForAtom)):
        G_values[0][i]=Session.run(GsForAtom[i],feed_dict={Alpha:AlphaVal})[0]
    
    return G_values


def force_for_atomicnetwork_internal(Session,AtomicNetwork,Alpha,AlphaValue):
    
    Out=0
    Network=AtomicNetwork[1]
    InputLayer=AtomicNetwork[2]
    GsForAtom=AtomicNetwork[8]
    G_values=get_g_values(Session,GsForAtom,Alpha,AlphaValue)
    part1=all_derivatives_of_Ei_wrt_Gij(Session,InputLayer,Network,G_values)[0]
    part2=all_derivatives_of_Gij_wrt_alpha(Session,GsForAtom,Alpha,AlphaValue)
    

    for i in range(0,len(part2)):
        Out+=part1[i]*part2[i][0][0]
        
    return Out

def total_force_internal(Session,AtomicNNs,Alpha,AlphaValue):
    
    Force=0
    for i in range(0,len(AtomicNNs)):
        AtomicNetwork=AtomicNNs[i]
        Force+=force_for_atomicnetwork(Session,AtomicNetwork,Alpha,AlphaValue)
        
    return Force

def actfun(Session,ActFun,Argument):
    
    Out=0
    if ActFun=="sigmoid":
        Out=tf.nn.sigmoid(Argument)
    elif ActFun=="tanh":
        Out=tf.nn.tanh(Argument)
    elif ActFun=="relu":
        Out=tf.nn.relu(Argument)
    elif ActFun=="relu6":
        Out=tf.nn.relu6(Argument) 
    elif ActFun=="crelu":
        Out=tf.nn.crelu(Argument) 
    elif ActFun=="elu":
        Out=tf.nn.elu(Argument)     
    elif ActFun=="softplus":
        Out=tf.nn.softplus(Argument)     
    elif ActFun=="dropout":
        Out=tf.nn.dropout(Argument) 
    elif ActFun=="bias_add":
        Out=tf.nn.bias_add(Argument)  

    return Session.run(Out)[0]

def actfun_derivative(Session,ActFun,Argument):
    
    Out=0
    if ActFun=="sigmoid":
        Out=tf.nn.sigmoid(Argument)*(1-tf.nn.sigmoid(Argument))
    elif ActFun=="tanh":
        Out=1-tf.nn.tanh(Argument)**2
    #to be expanded for other functions

    return Session.run(Out)[0]

def bias_plus_sum_weights_times_argument(Bias,Weights,Argument):
    
    return np.matmul(Argument,Weights)+Bias

def evaluate_force_networks(Session,ForceNetworks,Gs,Bias,Weights,ActFunStr):
    
    Activations=list()
    matWeights=Weights[0]
    matBias=Bias[0]
    Argument=bias_plus_sum_weights_times_argument(matBias,matWeights,Gs)
    Activations.append(Argument)
    for i in range(0,len(ForceNetworks)):
        Network=ForceNetworks[i][0]
        InputLayer=[ForceNetworks[i][1]]
        matWeights=Weights[i+1]
        matBias=Bias[i+1]
        #Value of force network
        NetworkOut=evaluate(Session,Network,InputLayer,Gs)
        Argument=bias_plus_sum_weights_times_argument(matBias,matWeights,NetworkOut)
        Activations.append(Argument)
        
    return Activations

def evaluate_derivatives(Session,ActFunStr,Arguments):
    
    for i in range(0,len(Arguments)):
        if i==0:
            Derivatives=actfun_derivative(Session,ActFunStr,Arguments[i])
        else:
            Derivatives=Derivatives*actfun_derivative(Session,ActFunStr,Arguments[i])
       
    return Derivatives

def force_for_atomicnetwork(Session,AtomicNetwork,Alpha,AlphaValue):
    #Get Data from network
    ForceNetworks=AtomicNetwork[4]
    Weights=AtomicNetwork[5]
    Bias=AtomicNetwork[6]
    ActFun=AtomicNetwork[7]
    GsForAtom=AtomicNetwork[8]
    Gs=get_g_values(Session,GsForAtom,Alpha,AlphaValue)
    #Calculate the dE/dG part of the force
    Activations=evaluate_force_networks(Session,ForceNetworks,Gs,Bias,Weights,ActFun)
    Derivatives=evaluate_derivatives(Session,ActFun,Activations)
    
    dE_dG=np.matmul(Weights[0],Derivatives)

    dG_dAlpha=all_derivatives_of_Gij_wrt_alpha(Session,GsForAtom,Alpha,AlphaValue)#not correct yet derivatives have to be implemented

    F=-sum(dE_dG*dG_dAlpha)
    
    return F

def total_force(Session,AtomicNNs,Alpha,AlphaValue):
    #Analytic calculation for faster evaluation
    Force=0
    for i in range(0,len(AtomicNNs)):
        AtomicNetwork=AtomicNNs[i]
        Force+=force_for_atomicnetwork(Session,AtomicNetwork,Alpha,AlphaValue)#not correct yet derivatives have to be implemented
        
    return Force
    
def evaluateAllAtomicNNs(Session,AtomicNNs,InData):
    
    Energy=0
    Layers,Data=prepare_data_environment_for_atomicNNs(AtomicNNs,InData,None,None)

    for i in range(0,len(AtomicNNs)):
        AtomicNetwork=AtomicNNs[i]   
        Energy+=evaluate(Session,AtomicNetwork[1],[Layers[i]],Data[i])
        
    return Energy

def make_atomic_networks(Structures,NumberOfSameNetworks,Gs=None,HiddenType=None,HiddenData=None,BiasData=None,ActFun=None,ActFunParam=None):


    AllHiddenLayers=list()
    AtomicNNs=list()
    
    #make all the networks for the different atom types
    for i in range(0,len(Structures)):
        #Make hidden layers
        HiddenLayers=list()
        Structure=Structures[i]
        if HiddenData!=None:
            RawWeights=HiddenData[i]
            RawBias=BiasData[i]
            ForceNetworks=make_force_networks(Structure,RawWeights,RawBias)
            
            for j in range(1,len(Structure)):
                NrIn=Structure[j-1]
                NrHidden=Structure[j]
                HiddenLayers.append(construct_hidden_layer(NrIn,NrHidden,HiddenType,HiddenData[i][j-1],BiasData[i][j-1]))
        else:
            RawWeights=None
            RawBias=None
            ForceNetworks=None
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

            if Gs[i]!=None:
                AtomicNNs.append([NumberOfSameNetworks[i],Network,InputLayer,OutputLayer,ForceNetworks,RawWeights,RawBias,ActFun,Gs[i]])
            else:
                AtomicNNs.append([NumberOfSameNetworks[i],Network,InputLayer,OutputLayer,ForceNetworks,RawWeights,RawBias,ActFun])

    return AtomicNNs,AllHiddenLayers

def train_atomic_networks(AtomicNNs,TrainingInputs,TrainingOutputs,Epochs,LearningRate,ValidationInputs=None,ValidationOutputs=None,CostCriterium=None,OptimizerType=None,OptimizerProp=None):
        
    
    ValidationCost=0
    TrainCost=0
    #Start Session 
        
    Session=tf.Session()

    #Make virtual output layer for feeding the data to the cost function
    OutputLayer=construct_output_layer(1)
    #Prepare data environment for training
    Layers,Data=prepare_data_environment_for_atomicNNs(AtomicNNs,TrainingInputs,OutputLayer,TrainingOutputs)
    #Make validation input vector
    if ValidationInputs != None:
        ValidationData=make_data_for_atomicNNs(ValidationInputs,ValidationOutputs)
    else:
        ValidationData=None
    #Cost function changes for every net 
    CostFun=atomic_cost_function(Session,AtomicNNs,OutputLayer)
    #Add optimizer
    if OptimizerType==None:
        Optimizer=tf.train.GradientDescentOptimizer(LearningRate).minimize(CostFun)
    else:
        if OptimizerType=="GradientDescent":
            Optimizer=tf.train.GradientDescentOptimizer(LearningRate).minimize(CostFun)
        elif OptimizerType=="Adagrad":
            Optimizer=tf.train.AdagradOptimizer(LearningRate).minimize(CostFun)
        elif OptimizerType=="Adadelta":
            Optimizer=tf.train.AdadeltaOptimizer(LearningRate).minimize(CostFun)
        elif OptimizerType=="AdagradDA":
            Optimizer=tf.train.AdagradDAOptimizer(LearningRate,OptimizerProp).minimize(CostFun)
        elif OptimizerType=="Momentum":
            Optimizer=tf.train.MomentumOptimizer(LearningRate,OptimizerProp).minimize(CostFun)
        elif OptimizerType=="Adam":
            Optimizer=tf.train.AdamOptimizer(LearningRate).minimize(CostFun)
        elif OptimizerType=="Ftrl":
            Optimizer=tf.train.FtrlOptimizer(LearningRate).minimize(CostFun)   
        elif OptimizerType=="ProximalGradientDescent":
            Optimizer=tf.train.ProximalGradientDescentOptimizer(LearningRate).minimize(CostFun)  
        elif OptimizerType=="ProximalAdagrad":
            Optimizer=tf.train.ProximalAdagradOptimizer(LearningRate).minimize(CostFun)   
        elif OptimizerType=="RMSProp":
            Optimizer=tf.train.RMSPropOptimizer(LearningRate).minimize(CostFun)  
        else:
            Optimizer=tf.train.GradientDescentOptimizer(LearningRate).minimize(CostFun)
    #Start training of the atomic network
    Session,TrainCost,ValidationCost=train(Session,Optimizer,CostFun,Layers,Data,Epochs,ValidationData,CostCriterium)
    TrainedNetwork=tf.trainable_variables()
        
    return Session,TrainedNetwork,TrainCost,ValidationCost

class AtomicNeuralNetInstance(object):
    
    def __init__(self):
        self.Structures=list()
        self.NumberOfSameNetworks=list()
        self.AtomicNNs=list()
        self.TrainingInputs=list()
        self.TrainingOutputs=list()
        self.Epochs=1000
        self.LearningRate=0.001
        self.ValidationInputs=list()
        self.ValidationOutputs=list()
        self.Gs=list()
        self.HiddenType=list()
        self.HiddenData=list()
        self.BiasData=list()
        
        self.ActFun=None
        self.ActFunParam=None
        self.CostCriterium=None
        self.OptimizerType=None
        self.OptimizerProp=None
        
        self.Session=[]
        self.TrainedNetwork=[]
        self.TrainingCosts=[]
        self.ValidationCosts=[]
        self.TrainedVariables=[]
        self.VariablesDictionary={}
        
    def start_training_instance(self):
        
        self.AtomicNNs,self.VariablesDictionary=make_atomic_networks(self.Structures,self.NumberOfSameNetworks,self.Gs,self.HiddenType,self.HiddenData,self.BiasData,self.ActFun,self.ActFunParam)
        self.Session,self.TrainedNetwork,self.TrainingCosts,self.ValidationCosts=train_atomic_networks(self.AtomicNNs,self.TrainingInputs,self.TrainingOutputs,self.Epochs,self.LearningRate,self.ValidationInputs,self.ValidationOutputs,self.CostCriterium,self.OptimizerType,self.OptimizerProp)
        self.TrainedVariables=get_trained_variables(self.Session,self.VariablesDictionary)


    def start_evaluation_instance(self):
        
        self.AtomicNNs=expand_neuralnet(self.TrainedVariables,self.NumberOfSameNetworks,self.Gs)
        
    