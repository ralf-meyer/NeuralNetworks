#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:30:10 2017

@author: Fuchs Alexander
"""
import numpy as np
import tensorflow as tf
import DataSet
import SymmetryFunctionSet
import random as rand
import matplotlib.pyplot as plt
import multiprocessing



plt.ion()
tf.reset_default_graph()

def construct_input_layer(InputUnits):
    #Construct inputs for the NN
    Inputs=tf.placeholder(tf.float32, shape=[None, InputUnits])

    return Inputs

def construct_hidden_layer(LayerBeforeUnits,HiddenUnits,InitType=None,InitData=[],BiasType=None,BiasData=[],MakeAllVariable=False,Mean=0.0,Stddev=1.0):
    #Construct the weights for this layer

    if len(InitData)==0:
        if InitType!=None:
            if InitType == "zeros":
                Weights=tf.Variable(tf.zeros([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
            elif InitType =="ones":
                Weights=tf.Variable(tf.ones([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
            elif InitType == "fill":
                Weights=tf.Variable(tf.fill([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
            elif InitType == "random_normal":
                Weights=tf.Variable(tf.random_normal([LayerBeforeUnits,HiddenUnits],mean=Mean,stddev=Stddev),dtype=tf.float32,name="variable")
            elif InitType == "truncated_normal":
                Weights=tf.Variable(tf.truncated_normal([LayerBeforeUnits,HiddenUnits],mean=Mean,stddev=Stddev),dtype=tf.float32,name="variable")
            elif InitType == "random_uniform":
                Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
            elif InitType == "random_shuffle":
                Weights=tf.Variable(tf.random_shuffle([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
            elif InitType == "random_crop":
                Weights=tf.Variable(tf.random_crop([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
            elif InitType == "random_gamma":
                Weights=tf.Variable(tf.random_gamma([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
            else:
                #Assume random weights if no InitType is given
                Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
        else:
            #Assume random weights if no InitType is given
            Weights=tf.Variable(tf.random_uniform([LayerBeforeUnits,HiddenUnits]),dtype=tf.float32,name="variable")
    else:
        if MakeAllVariable==False:
            Weights=tf.constant(InitData,dtype=tf.float32,name="constant")
        else:
            Weights=tf.Variable(InitData,dtype=tf.float32,name="variable")
    #Construct the bias for this layer
    if len(BiasData)!=0:

        if MakeAllVariable==False:
            Biases=tf.constant(BiasData,dtype=tf.float32,name="bias")
        else:
            Biases=tf.Variable(BiasData,dtype=tf.float32,name="bias")

    else:
        if InitType == "zeros":
            Biases=tf.Variable(tf.zeros([HiddenUnits]),dtype=tf.float32,name="bias")
        elif InitType =="ones":
            Biases=tf.Variable(tf.ones([HiddenUnits]),dtype=tf.float32,name="bias")
        elif InitType == "fill":
            Biases=tf.Variable(tf.fill([HiddenUnits],BiasData),dtype=tf.float32,name="bias")
        elif InitType == "random_normal":
            Biases=tf.Variable(tf.random_normal([HiddenUnits],mean=Mean,stddev=Stddev),dtype=tf.float32,name="bias")
        elif InitType == "truncated_normal":
            Biases=tf.Variable(tf.truncated_normal([HiddenUnits],mean=Mean,stddev=Stddev),dtype=tf.float32,name="bias")
        elif InitType == "random_uniform":
            Biases=tf.Variable(tf.random_uniform([HiddenUnits]),dtype=tf.float32,name="bias")
        elif InitType == "random_shuffle":
            Biases=tf.Variable(tf.random_shuffle([HiddenUnits]),dtype=tf.float32,name="bias")
        elif InitType == "random_crop":
            Biases=tf.Variable(tf.random_crop([HiddenUnits],BiasData),dtype=tf.float32,name="bias")
        elif InitType == "random_gamma":
            Biases=tf.Variable(tf.random_gamma([HiddenUnits],InitData),dtype=tf.float32,name="bias")
        else:
            Biases = tf.Variable(tf.random_uniform([HiddenUnits]),dtype=tf.float32,name="bias")
    
    return Weights,Biases

def construct_output_layer(OutputUnits):
    #Construct the output for the NN
    Outputs = tf.placeholder(tf.float32, shape=[None, OutputUnits])

    return Outputs


def construct_trainable_layer(NrInputs,NrOutputs,Min):
    #make a not trainable layer with the weights one

    Weights=tf.Variable(np.ones([NrInputs,NrOutputs]),dtype=tf.float32)#, trainable=False)
    Biases=tf.Variable(np.zeros([NrOutputs]),dtype=tf.float32)#,trainable=False)
    if Min!=0:
        Biases=tf.add(Biases,Min/NrOutputs)

    return Weights,Biases

def construct_not_trainable_layer(NrInputs,NrOutputs,Min):
    #make a not trainable layer with the weights one

    Weights=tf.constant(np.ones([NrInputs,NrOutputs]),dtype=tf.float32)#, trainable=False)
    Biases=tf.constant(np.zeros([NrOutputs]),dtype=tf.float32)#,trainable=False)
    if Min!=0:
        Biases=tf.add(Biases,Min/NrOutputs)

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
        elif ActFun == "none":
            Out = tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias
    else:
        Out=tf.nn.sigmoid(tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)

    return Out


def make_force_networks(Structure,HiddenData,BiasData):

    ForceNetworks=list()

    for i in range(1,len(Structure)-1):
        Network,InputLayer,OutputLayer=make_standard_neuralnetwork(Structure[0:i+1],None,HiddenData,None,BiasData)
        ForceNetworks.append([Network,InputLayer,OutputLayer])
    return ForceNetworks

def make_standard_neuralnetwork(Structure,HiddenType=None,HiddenData=None,BiasType=None,BiasData=None,ActFun=None,ActFunParam=None):
    #Construct the NN

    #Make inputs
    NrInputs=Structure[0]
    InputLayer=construct_input_layer(NrInputs)
    #Make hidden layers
    HiddenLayers=list()
    for i in range(1,len(Structure)):
        NrIn=Structure[i-1]
        NrHidden=Structure[i]
        HiddenLayers.append(construct_hidden_layer(NrIn,NrHidden,HiddenType,HiddenData[i-1],BiasType,BiasData[i-1]))


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

def get_my_variables(partial_dict):
    OutVars=[]
    for AtomicNet in partial_dict:
        for SubNet in AtomicNet:
            if len(SubNet)>0:
                for Var in SubNet:
                    OutVars.append(Var[0])
                    
    train_vars=tf.trainable_variables()
    for var in train_vars:
        if var not in OutVars:
            OutVars.append(var)
            
    return OutVars
            

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


def make_data_for_atomicNNs(InData,OutData=[]):

    CombinedData=list()
    for Data in InData:
        CombinedData.append(Data)
    if len(OutData)!=0:
        CombinedData.append(OutData)

    return CombinedData

def prepare_data_environment_for_partitioned_atomicNNs(AtomicNNs,InData,NumberOfRadial,OutputLayer=[],OutData=[]):

    Layers=list()
    CombinedData=list()
    for i in range(0,len(AtomicNNs)):
        Layer_parts=AtomicNNs[i][2]
        Data=InData[i]
        #Append layers for each part
        for j in range(0,3):
            if Layer_parts[j]!=j: #Layer_parts is range(3) if empty
                Layers.append(Layer_parts[j])

                if j==0: #Radial data
                    CombinedData.append(Data[:,0:NumberOfRadial])
                elif j==1: #Angular data
                    CombinedData.append(Data[:,NumberOfRadial:])
                else:
                    CombinedData.append(Data[:])
                
    if OutputLayer!=None:
        Layers.append(OutputLayer)
    if len(OutData)!=0:
        CombinedData.append(OutData)

    return Layers,CombinedData

def prepare_data_environment_for_atomicNNs(AtomicNNs,InData,OutputLayer=[],OutData=[]):

    Layers=make_layers_for_atomicNNs(AtomicNNs,OutputLayer)
    Data=make_data_for_atomicNNs(InData,OutData)

    return Layers,Data


def train(Session,Optimizer,CostFun,Layers,TrainingData,Epochs,ValidationData=None,CostCriterium=None,MakePlot=False):
    #Train with specifications and return loss
    TrainCost=list()
    ValidationCost=list()
    print("Started training...")
    for i in range(Epochs):
        Cost=train_step(Session,Optimizer,Layers,TrainingData,CostFun)
        TrainCost.append(sum(Cost)/len(Cost))
        #check validation dataset error
        if ValidationData!=None:
            ValidationCost.append(sum(validate_step(Session,Layers,ValidationData,CostFun))/len(Cost))

        if i % max(int(Epochs/100),1)==0:
            if MakePlot:
                if i==0:
                    figure,ax,TrainPlot,ValPlot=initialize_cost_plot(TrainCost,ValidationCost)
                else:
                    update_cost_plot(figure,ax,TrainPlot,TrainCost,ValPlot,ValidationCost)
            print(str(100*i/Epochs)+" %")

        if TrainCost[-1]<CostCriterium:
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

def output_of_all_partitioned_atomic_networks(Session,AtomicNNs):

    TotalEnergy=0
    AllEnergies=list()
    for i in range(0,len(AtomicNNs)):
        #Get network data
        AtomicNetwork=AtomicNNs[i]
        Networks=AtomicNetwork[1]
        for j in range(0,3):
            SubNet=Networks[j]
            if SubNet!=j:
                #Get input data for network
                AllEnergies.append(SubNet)

    TotalEnergy=tf.add_n(AllEnergies)


    return TotalEnergy,AllEnergies

def output_of_all_atomic_networks(Session,AtomicNNs):

    TotalEnergy=0
    AllEnergies=list()

    for i in range(0,len(AtomicNNs)):
        #Get network data
        AtomicNetwork=AtomicNNs[i]
        Network=AtomicNetwork[1]
        #Get input data for network
        AllEnergies.append(Network)
    
    TotalEnergy=tf.add_n(AllEnergies)

    return TotalEnergy,AllEnergies


def atomic_cost_function(Session,AtomicNNs,ReferenceOutput,Regularization="none",RegularizationParam=0.001,IsPartitioned=False):
    
    if IsPartitioned==True:
        TotalEnergy,AllEnergies=output_of_all_partitioned_atomic_networks(Session,AtomicNNs)
    else:
        TotalEnergy,AllEnergies=output_of_all_atomic_networks(Session,AtomicNNs)
    Cost=total_cost_for_network(TotalEnergy,ReferenceOutput)
    if Regularization=="L1":
        trainableVars=tf.trainable_variables()
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
        Cost += tf.contrib.layers.apply_regularization(l1_regularizer, trainableVars)
    elif Regularization=="L2":
        trainableVars=tf.trainable_variables()
        Cost += tf.add_n([tf.nn.l2_loss(v) for v in trainableVars
                           if 'bias' not in v.name]) * RegularizationParam

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

def get_weights_biases_from_partitioned_data(TrainedData):

    Weights=list()
    Biases=list()
    for i in range(0,len(TrainedData)):
        NetworkData=TrainedData[i]
        ThisWeights=PartitionedNetworkData()
        ThisBiases=PartitionedNetworkData()
        for j in range(0,len(NetworkData)):
            SubNetData=NetworkData[j]
            for k in range(0,len(SubNetData)):
                if j==0:
                    ThisWeights.RadialNetworkData.append(SubNetData[k][0])
                    ThisWeights.RadialVariable=False
                    ThisBiases.RadialNetworkData.append(SubNetData[k][1])
                    
                elif j==1:
                    ThisWeights.AngularNetworkData.append(SubNetData[k][0])
                    ThisWeights.AngularVariable=False
                    ThisBiases.AngularNetworkData.append(SubNetData[k][1])
                elif j==2:
                    ThisWeights.CorrectionNetworkData.append(SubNetData[k][0])
                    ThisWeights.CorretionVariable=False
                    ThisBiases.CorrectionNetworkData.append(SubNetData[k][1])
        Weights.append(ThisWeights)
        Biases.append(ThisBiases)

    return Weights,Biases

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


def get_size_of_input(Data):
    
    Size=list()
    NrAtoms=len(Data)
    for i in range(0,NrAtoms):
        Size.append(len(Data[i]))

    return Size

def get_trained_variables_partitioned(Session,Dict):
    
    NetworkData=list()
    
    for Network in Dict:
        NetLayers=list()
        for i in range(0,len(Network)):
            SubNetLayers=list()
            if Network[i]!=i: #if SNetwork[i]==i means no net data
                SubNet=Network[i]
                for j in range(0,len(SubNet)):
                    Weights=Session.run(SubNet[j][0])
                    Biases=Session.run(SubNet[j][1])
                    SubNetLayers.append([Weights,Biases])
                    
            NetLayers.append(SubNetLayers)
            
        NetworkData.append(NetLayers)

    return NetworkData                
                    
        

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

def evaluate_all_atomicnns(Session,AtomicNNs,InData):

    Energy=0
    Layers,Data=prepare_data_environment_for_atomicNNs(AtomicNNs,InData,list(),list())

    for i in range(0,len(AtomicNNs)):
        AtomicNetwork=AtomicNNs[i]
        Energy+=evaluate(Session,AtomicNetwork[1],[Layers[i]],Data[i])

    return Energy

def evaluate_all_partitioned_atomicnns(Session,AtomicNNs,InData,NumberOfRadial):

    Energy=0
    Layers,Data=prepare_data_environment_for_partitioned_atomicNNs(AtomicNNs,InData,NumberOfRadial,list(),list())
    ct=0
    for i in range(0,len(AtomicNNs)):
        AllAtomicNetworks=AtomicNNs[i][1]
        for j in range(0,3):
            SubNet=AllAtomicNetworks[j]
            if SubNet!=j:
                Energy+=evaluate(Session,SubNet,[Layers[ct]],Data[ct])
                ct=ct+1

    return Energy


def train_atomic_networks(Session,AtomicNNs,TrainingInputs,TrainingOutputs,Epochs,Optimizer,OutputLayer,CostFun,ValidationInputs=None,ValidationOutputs=None,CostCriterium=None,MakePlot=False,IsPartitioned=False,NrOfRadial=None):


    ValidationCost=0
    TrainCost=0
    #Prepare data environment for training
    if IsPartitioned==False:
        Layers,Data=prepare_data_environment_for_atomicNNs(AtomicNNs,TrainingInputs,OutputLayer,TrainingOutputs)
    else:
        Layers,Data=prepare_data_environment_for_partitioned_atomicNNs(AtomicNNs,TrainingInputs,NrOfRadial,OutputLayer,TrainingOutputs)
    #Make validation input vector
    if len(ValidationInputs)>0:
        if IsPartitioned==False:
            ValidationData=make_data_for_atomicNNs(ValidationInputs,ValidationOutputs)
        else:
            _,ValidationData=prepare_data_environment_for_partitioned_atomicNNs(AtomicNNs,TrainingInputs,NrOfRadial,OutputLayer,TrainingOutputs)
    else:
        ValidationData=None
    #Start training of the atomic network
    Session,TrainCost,ValidationCost=train(Session,Optimizer,CostFun,Layers,Data,Epochs,ValidationData,CostCriterium,MakePlot)
    TrainedNetwork=tf.trainable_variables()

    return Session,TrainedNetwork,TrainCost,ValidationCost

def train_atomic_network_batch(Session,Optimizer,Layers,TrainingData,ValidationData,CostFun):

    TrainingCost=0
    ValidationCost=0
    #train batch
    TrainingCost=sum(train_step(Session,Optimizer,Layers,TrainingData,CostFun))[0]

    #check validation dataset error
    if ValidationData!=None:
        ValidationCost=sum(validate_step(Session,Layers,ValidationData,CostFun))[0]

    return TrainingCost,ValidationCost

def guarantee_initialized_variables(session, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   session.run(tf.report_uninitialized_variables(list_of_variables)))
    session.run(tf.initialize_variables(uninitialized_variables))
    return uninitialized_variables

def initialize_cost_plot(TrainingData,ValidationData=[]):

    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscaley_on(True)
    TrainingCostPlot, = ax.plot(np.arange(0,len(TrainingData)),TrainingData)
    if len(ValidationData)!=0:
        ValidationCostPlot,=ax.plot(np.arange(0,len(ValidationData)),ValidationData)
    else:
        ValidationCostPlot=None

    #Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    #We need to draw *and* flush
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig,ax,TrainingCostPlot,ValidationCostPlot

def update_cost_plot(figure,ax,TrainingCostPlot,TrainingCost,ValidationCostPlot=None,ValidationCost=None):

    TrainingCostPlot.set_data(np.arange(0,len(TrainingCost)),TrainingCost)
    if ValidationCostPlot!=None:
        ValidationCostPlot.set_data(np.arange(0,len(ValidationCost)),ValidationCost)

    #Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    #We need to draw *and* flush
    figure.canvas.draw()
    figure.canvas.flush_events()
    
def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in xrange(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            break
        p = beta * p - r
    return x


class AtomicNeuralNetInstance(object):

    def __init__(self):
        #Training variables
        self.Structures=list()
        self.NumberOfSameNetworks=list()
        self.AtomicNNs=list()
        self.TrainingInputs=list()
        self.TrainingOutputs=list()
        self.Epochs=1000
        self.LearningRate=0.01
        self.ValidationInputs=list()
        self.ValidationOutputs=list()
        self.Gs=list()
        self.HiddenType="truncated_normal"
        self.HiddenData=list()
        self.BiasType="zeros"
        self.BiasData=list()
        self.TrainingBatches=list()
        self.ValidationBatches=list()
        self.ActFun="elu"
        self.ActFunParam=None
        self.CostCriterium=0
        self.OptimizerType=None
        self.OptimizerProp=None
        self.Session=[]
        self.TrainedNetwork=[]
        self.TrainingCosts=[]
        self.ValidationCosts=[]
        self.OverallTrainingCosts=[]
        self.OverallValidationCosts=[]
        self.TrainedVariables=[]
        self.VariablesDictionary={}
        self.CostFun=None
        self.Optimizer=None
        self.OutputLayer=None
        self.MakePlots=False
        self.InitMean=0.0
        self.InitStddev=1.0
        self.MakeLastLayerConstant=True
        self.MakeAllVariable=True
        self.Regularization="none"
        self.RegularizationParam=0.001
        self.DeltaE=0
        self.CurrentEpochNr=0
        self.IsPartitioned=False
        #Data variables
        self.AllGeometries=list()
        self.Batches=list()
        self.SizeOfInputs=list()
        self.XYZfile=None
        self.Logfile=None
        self.SymmFunKeys=[]
        self.TotalNrOfRadialFuns=None
        self.NumberOfRadialFunctions=7
        self.Rs=[]
        self.Etas=[]
        self.Zetas=[]
        self.Lambs=[]
        self.SymmFunSet=None
        self.Ds=None
        self.MeansOfDs=[]
        self.MinOfOut=None
        self.VarianceOfDs=[]
        
        #Other
        self.Multiple=False


    def initialize_network(self):
        
        try:
            #Make virtual output layer for feeding the data to the cost function
            self.OutputLayer=construct_output_layer(1)
            #Cost function for whole net
            self.CostFun=atomic_cost_function(self.Session,self.AtomicNNs,self.OutputLayer,self.Regularization,self.RegularizationParam,self.IsPartitioned)
            
            #if self.IsPartitioned==True:
            All_Vars=tf.trainable_variables()#get_my_variables(self.VariablesDictionary)
                
                #Set optimizer
            if self.OptimizerType==None:
               self.Optimizer=tf.train.GradientDescentOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
            else:
                if self.OptimizerType=="GradientDescent":
                    self.Optimizer=tf.train.GradientDescentOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="Adagrad":
                    self.Optimizer=tf.train.AdagradOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="Adadelta":
                    self.Optimizer=tf.train.AdadeltaOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="AdagradDA":
                    self.Optimizer=tf.train.AdagradDAOptimizer(self.LearningRate,self.OptimizerProp).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="Momentum":
                    self.Optimizer=tf.train.MomentumOptimizer(self.LearningRate,self.OptimizerProp).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="Adam":
                    self.Optimizer=tf.train.AdamOptimizer(self.LearningRate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="Ftrl":
                   self.Optimizer=tf.train.FtrlOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="ProximalGradientDescent":
                    self.Optimizer=tf.train.ProximalGradientDescentOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="ProximalAdagrad":
                    self.Optimizer=tf.train.ProximalAdagradOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
                elif self.OptimizerType=="RMSProp":
                    self.Optimizer=tf.train.RMSPropOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
                else:
                    self.Optimizer=tf.train.GradientDescentOptimizer(self.LearningRate).minimize(self.CostFun,var_list=All_Vars)
        except:
            print("Evaluation only no training supported!")
        #Initialize session
        self.Session.run(tf.global_variables_initializer())

    def load_model(self,NrHiddenOld,ModelName="trained_variables"):

        if ".npy" not in ModelName:
            ModelName=ModelName+".npy"
            temp=np.load(ModelName)
        self.TrainedVariables=temp[0]
        self.MinOfOut=temp[1]


        return 1

    def expand_existing_net(self,ModelName="trained_variables",MakeAllVariable=True,ModelData=None):
        
        if ModelData==None:
            Success=AtomicNeuralNetInstance.load_model(self,ModelName)
        else:
            self.TrainedVariables=ModelData[0]
            self.MinOfOut=ModelData[1]
            Success=1
        if Success==1:
            if self.IsPartitioned==False:
                self.HiddenData,self.BiasData=get_weights_biases_from_data(self.TrainedVariables)
            else:
                self.HiddenData,self.BiasData=get_weights_biases_from_partitioned_data(self.TrainedVariables)

            self.HiddenType="truncated_normal"
            self.InitMean=0
            self.InitStddev=0.01
            self.BiasType="zeros"
            self.MakeAllVariable=MakeAllVariable
            AtomicNeuralNetInstance.make_and_initialize_network(self)

    def make_network(self):

        Execute=True
        if len(self.Structures)==0:
            print("No structures for the specific nets specified!")
            Execute=False
        if len(self.NumberOfSameNetworks)==0:
            print("No number of specific nets specified!")
            Execute=False

        if Execute==True:
            if self.IsPartitioned==False:
                AtomicNeuralNetInstance.make_atomic_networks(self)
            else:
                AtomicNeuralNetInstance.make_partitioned_atomic_networks(self)

    def make_and_initialize_network(self):

        AtomicNeuralNetInstance.make_network(self)
        AtomicNeuralNetInstance.initialize_network(self)

    def start_training(self):

        Execute=True
        if len(self.AtomicNNs)==0:
            print("No atomic neural nets available!")
            Execute=False
        if len(self.TrainingInputs)==0:
            print("No training inputs specified!")
            Execute=False
        if len(self.TrainingOutputs)==0:
            print("No training outputs specified!")
            Execute=False

        if Execute==True:
            self.Session,self.TrainedNetwork,TrainingCosts,ValidationCosts=train_atomic_networks(self.Session,self.AtomicNNs,self.TrainingInputs,self.TrainingOutputs,self.Epochs,self.Optimizer,self.OutputLayer,self.CostFun,self.ValidationInputs,self.ValidationOutputs,self.CostCriterium,self.MakePlots,self.TotalNrOfRadialFuns)
            self.TrainedVariables=get_trained_variables(self.Session,self.VariablesDictionary)
            #Store variables
            np.save("trained_variables",self.TrainedVariables)

            self.TrainingCosts=TrainingCosts
            self.ValidationCosts=ValidationCosts
            print("Training finished")

        return self.TrainingCosts,self.ValidationCosts

    #def expand_neuralnet(self,TrainedData, nAtoms, Gs):

        #AtomicNNs = list()
        #self.TrainedVariables=TrainedData
        #self.NumberOfSameNetworks=nAtoms
        #self.Gs=Gs
        #Structures = get_structure_from_data(self.TrainedVariables)
        #Weights, Biases = get_weights_biases_from_data(self.TrainedVariables)
        #AtomicNeuralNetInstance.make_atomic_networks(self)

    def start_evaluation(self):

        self.Session,self.AtomicNNs=AtomicNeuralNetInstance.expand_neuralnet(self)
        AtomicNeuralNetInstance.initialize_network(self)

    def eval_step(self):
        Out=0
        if self.IsPartitioned==False:
            Out=evaluate_all_atomicnns(self.Session,self.AtomicNNs,self.TrainingInputs)
        else:
            Out=evaluate_all_partitioned_atomicnns(self.Session,self.AtomicNNs,self.TrainingInputs,self.TotalNrOfRadialFuns)
        return Out

    def start_batch_training(self):
        #Clear cost array for multi instance training
        self.OverallTrainingCosts=list()
        self.OverallValidationCosts=list()
        
        Execute=True
        if len(self.AtomicNNs)==0:
            print("No atomic neural nets available!")
            Execute=False
        if len(self.TrainingBatches)==0:
            print("No training batches specified!")
            Execute=False

        if sum(self.NumberOfSameNetworks)!= len(self.TrainingBatches[0][0]):
            print([self.NumberOfSameNetworks,len(self.TrainingBatches[0][0])])
            print("Input does not match number of specified networks!")
            Execute=False

        if Execute==True:
            if self.Multiple==False:
                print("Started batch training...")
            NrOfTrainingBatches=len(self.TrainingBatches)
            if self.ValidationBatches:
                NrOfValidationBatches=len(self.ValidationBatches)
            
            for i in range(0,self.Epochs):
                self.CurrentEpochNr=i
                for j in range(0,NrOfTrainingBatches):

                    tempTrainingCost=[]
                    tempValidationCost=[]
                    rnd=rand.randint(0,NrOfTrainingBatches-1)
                    self.TrainingInputs=self.TrainingBatches[rnd][0]
                    self.TrainingOutputs=self.TrainingBatches[rnd][1]

                    BatchSize=self.TrainingInputs[0].shape[0]
                    if self.ValidationBatches:
                        rnd=rand.randint(0,NrOfValidationBatches-1)
                        self.ValidationInputs=self.ValidationBatches[rnd][0]
                        self.ValidationOutputs=self.ValidationBatches[rnd][1]
                    #Prepare data and layers for feeding
                    if i==0:
                        if self.IsPartitioned==False:
                            Layers,TrainingData=prepare_data_environment_for_atomicNNs(self.AtomicNNs,self.TrainingInputs,self.OutputLayer,self.TrainingOutputs)
                        else:
                            Layers,TrainingData=prepare_data_environment_for_partitioned_atomicNNs(self.AtomicNNs,self.TrainingInputs,self.TotalNrOfRadialFuns,self.OutputLayer,self.TrainingOutputs)
                    else:
                        if self.IsPartitioned==False:
                            TrainingData=make_data_for_atomicNNs(self.TrainingInputs,self.TrainingOutputs)
                        else:
                            _,TrainingData=prepare_data_environment_for_partitioned_atomicNNs(self.AtomicNNs,self.TrainingInputs,self.TotalNrOfRadialFuns,self.OutputLayer,self.TrainingOutputs)
                    #Make validation input vector
                    if len(self.ValidationInputs)>0:
                        if self.IsPartitioned==False:
                            ValidationData=make_data_for_atomicNNs(self.ValidationInputs,self.ValidationOutputs)
                        else:
                            _,ValidationData=prepare_data_environment_for_partitioned_atomicNNs(self.AtomicNNs,self.TrainingInputs,self.TotalNrOfRadialFuns,self.OutputLayer,self.TrainingOutputs)
                    else:
                        ValidationData=None
                    #Train one batch
                    
                    TrainingCosts,ValidationCosts=train_atomic_network_batch(self.Session,self.Optimizer,Layers,TrainingData,ValidationData,self.CostFun)
                                        
                    tempTrainingCost.append(TrainingCosts)
                    tempValidationCost.append(ValidationCosts)
                    
                    self.OverallTrainingCosts.append(TrainingCosts/BatchSize)
                    self.OverallValidationCosts.append(ValidationCosts/BatchSize)

                if len(tempTrainingCost)>0:
                    self.TrainingCosts=sum(tempTrainingCost)/(len(tempTrainingCost)*BatchSize)
                    self.ValidationCosts=sum(tempValidationCost)/(len(tempValidationCost)*BatchSize)
                else:
                    self.TrainingCosts=1e10
                    self.ValidationCosts=1e10
                    
                if self.ValidationCosts!=0:
                    self.DeltaE=np.sqrt((self.TrainingCosts+self.ValidationCosts)/2)
                else:
                    self.DeltaE=np.sqrt(self.TrainingCosts)

                if self.Multiple==False:
                    if i % max(int(self.Epochs/20),1)==0 or i==(self.Epochs-1):
                        #Cost plot
                        #print([evaluate_all_atomicnns(self.Session,self.AtomicNNs,self.TrainingInputs),self.TrainingOutputs])
                        if self.MakePlots==True:
                            if i ==0:
                                fig,ax,TrainingCostPlot,ValidationCostPlot=initialize_cost_plot(self.OverallTrainingCosts,self.OverallValidationCosts)
                            else:
                                update_cost_plot(fig,ax,TrainingCostPlot,self.OverallTrainingCosts,ValidationCostPlot,self.OverallValidationCosts)
                        #Finished percentage output
                        print(str(100*i/self.Epochs)+" %")
                        #Store variables
                        if self.IsPartitioned==False:
                            self.TrainedVariables=get_trained_variables(self.Session,self.VariablesDictionary)
                        else:
                            self.TrainedVariables=get_trained_variables_partitioned(self.Session,self.VariablesDictionary)

                        np.save("trained_variables",[self.TrainedVariables,self.MinOfOut])
                    

                    #Abort criteria
                    if self.TrainingCosts<=self.CostCriterium and self.ValidationCosts<=self.CostCriterium:
                        if self.ValidationCosts!=0:
                            print("Reached cost criterium: "+str((self.TrainingCosts+self.ValidationCosts)/2))
                            print("delta E = "+str(self.DeltaE))
                            break
                        else:
                            print("Reached cost criterium: "+str(self.TrainingCosts))
                            print("delta E = "+str(self.DeltaE))
                            break
                    if i==(self.Epochs-1):
                        print("Training finished")
                        print("delta E = "+str(self.DeltaE))
                    
            if self.Multiple==True:
                return [self.TrainedVariables,self.MinOfOut]
                    

    def calculate_statistics_for_dataset(self,TakeAsReference):

        print("Converting data to neural net input format...")
        NrGeom=len(self.Ds.geometries)
        AllTemp=list()
        #Get G vectors
        for i in range(0,NrGeom):
            temp=self.SymmFunSet.eval_geometry(self.Ds.geometries[i])
            NrAtoms=len(temp)
            self.AllGeometries.append(temp)
            if i % max(int(NrGeom/25),1)==0:
                print(str(100*i/NrGeom)+" %")
            for j in range(0,len(temp)):
                if i==0:
                    AllTemp.append(np.empty((NrGeom,temp[j].shape[0])))
                    AllTemp[j][i]=temp[j]
                else:
                    AllTemp[j][i]=temp[j]
        #calculate mean and sigmas for all Gs
        print("Calculating mean values and variances...")
        #Input statistics
        for InputsForNetX in AllTemp:
            self.MeansOfDs.append(np.mean(InputsForNetX,axis=0))
            self.VarianceOfDs.append(np.var(InputsForNetX,axis=0))
        #Output statistics
        NormalizedEnergy=np.divide(self.Ds.energies,NrAtoms)
        if self.MinOfOut== None:
            TakeAsReference=True
        else:
            if self.MinOfOut>np.min(NormalizedEnergy):
                TakeAsReference=True
        if TakeAsReference==True:
            self.MinOfOut=np.min(NormalizedEnergy)*2 #factor of two is to make sure that there is room for lower energies


    def read_files(self,TakeAsReference=True):

        Execute=True
        if self.XYZfile==None:
            print("No .xyz-file name specified!")
            Execute=False
        if self.Logfile==None:
            print("No log-file name specified!")
            Execute=False
        if len(self.SymmFunKeys)==0:
            print("No symmetry function keys specified!")
            Execute=False
#        if len(self.Zetas)==0:
#            print("No zetas specified!")
#            Execute=False
#        if len(self.Lambs)==0:
#            print("No lambdas specified!")
#            Execute=False

        if Execute==True:

            self.Ds=DataSet.DataSet()
            self.SymmFunSet=SymmetryFunctionSet.SymmetryFunctionSet(self.SymmFunKeys)
            self.Ds.read_lammps(self.XYZfile,self.Logfile)
            #self.SymmFunSet.add_radial_functions(self.Rs,self.Etas)
            self.SymmFunSet.add_radial_functions_evenly(self.NumberOfRadialFunctions)
            self.SymmFunSet.add_angluar_functions(self.Etas,self.Zetas,self.Lambs)
            AtomicNeuralNetInstance.calculate_statistics_for_dataset(self,TakeAsReference)


    def get_data_batch(self,BatchSize=100,NoBatches=False):

        AllData=list()
        Execute=True
        if self.SymmFunSet==None:
            print("No symmetry function set available!")
            Execute=False
        if self.Ds==None:
            print("No data set available!")
            Execute=False
        if len(self.Ds.geometries)==0:
            print("No geometries available!")
            Execute=False
        if len(self.Ds.energies)==0:
            print("No energies available!")
            Execute=False

        if Execute==True:

            OutputData=np.empty((BatchSize,1))
            if NoBatches==False:
                if BatchSize>len(self.AllGeometries)/10:
                    BatchSize=int(BatchSize/10)
                    print("Shrunk batches to size:"+str(BatchSize))

            #Create a list with all possible random values
            ValuesForDrawingSamples=list(range(0,len(self.Ds.geometries)))

            for i in range(0,BatchSize):
                #Get a new random number
                if NoBatches:
                    MyNr=i
                else:
                    rnd=rand.randint(0,len(ValuesForDrawingSamples)-1)
                    #Get number
                    MyNr=ValuesForDrawingSamples[rnd]
                    #remove number from possible samples
                    ValuesForDrawingSamples.pop(rnd)

                AllData.append(self.AllGeometries[MyNr])
                OutputData[i]=self.Ds.energies[MyNr]

            Inputs=AtomicNeuralNetInstance.sort_and_normalize_data(self,BatchSize,AllData)

            return Inputs,OutputData

    def get_data(self,BatchSize=100,CoverageOfSetInPercent=70,NoBatches=False):

        Execute=True
        if self.SymmFunSet==None:
            print("No symmetry function set available!")
            Execute=False
        if self.Ds==None:
            print("No data set available!")
            Execute=False
        if len(self.Ds.geometries)==0:
            print("No geometries available!")
            Execute=False
        if len(self.Ds.energies)==0:
            print("No energies available!")
            Execute=False

        if Execute==True:
            self.SizeOfInputs=get_size_of_input(self.SymmFunSet.eval_geometry(self.Ds.geometries[0]))
            self.TotalNrOfRadialFuns=self.NumberOfRadialFunctions*len(self.SymmFunKeys)
            AllDataSetLength=len(self.Ds.geometries)
            SetLength=int(AllDataSetLength*CoverageOfSetInPercent/100)

            if NoBatches==False:
                if BatchSize>len(self.AllGeometries)/10:
                    BatchSize=int(BatchSize/10)
                    print("Shrunk batches to size:"+str(BatchSize))
                NrOfBatches=max(1,int(round(SetLength/BatchSize,0)))
            else:
                NrOfBatches=1
                BatchSize=SetLength

            print("Creating and normalizing batches...")
            for i in range(0,NrOfBatches):
                self.Batches.append(AtomicNeuralNetInstance.get_data_batch(self,BatchSize,NoBatches))
                if NoBatches==False:
                    if i % max(int(NrOfBatches/10),1)==0:
                        print(str(100*i/NrOfBatches)+" %")

            return self.Batches

    def make_training_and_validation_data(self,BatchSize=100,TrainingSetInPercent=70,ValidationSetInPercent=30,NoBatches=False):

        if NoBatches==False:
            #Get training data
            self.TrainingBatches=AtomicNeuralNetInstance.get_data(self,BatchSize,TrainingSetInPercent,NoBatches)
            #Get validation data
            self.ValidationBatches=AtomicNeuralNetInstance.get_data(self,BatchSize,ValidationSetInPercent,NoBatches)
        else:
            #Get training data
            temp=AtomicNeuralNetInstance.get_data(self,BatchSize,TrainingSetInPercent,NoBatches)
            self.TrainingInputs=temp[0][0]
            self.TrainingOutputs=temp[0][1]
            #Get validation data
            temp=AtomicNeuralNetInstance.get_data(self,BatchSize,ValidationSetInPercent,NoBatches)
            self.ValidationInputs=temp[0][0]
            self.ValidationOutputs=temp[0][0]


    def sort_and_normalize_data(self,BatchSize,AllData):

        Inputs=list()
        for i in range(0,len(self.SizeOfInputs)):

            Inputs.append(np.zeros((BatchSize,self.SizeOfInputs[i])))
            #exclude nan values
            L=np.nonzero(self.VarianceOfDs[i])
            for j in range(0,len(AllData)):
                Inputs[i][j][L]=np.divide(np.subtract(AllData[j][i][L],self.MeansOfDs[i][L]),np.sqrt(self.VarianceOfDs[i][L]))

        return Inputs

    def make_partitioned_atomic_networks(self):

        AtomicNNs = list()
        AllHiddenLayers=list()
        # Start Session
        if self.Multiple==False:
            self.Session=tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=multiprocessing.cpu_count()))
            

        # make all the networks for the different atom types
        for i in range(0, len(self.Structures)):
            NetworkHiddenLayers=range(3)
            NetworkWeights=range(3)
            NetworkBias=range(3)
            
            RadialHiddenLayers=list()
            AngularHiddenLayers=list()
            CorrectionHiddenLayers=list()
            
            RadialWeights=list()
            AngularWeights=list()
            CorrectionWeights=list()
            
            RadialBias=list()
            AngularBias=list()
            CorrectionBias=list()
            
            RadialNetwork=None
            AngularNetwork=None
            CorrectionNetwork=None
            
            RadialInputLayer=None
            AngularInputLayer=None
            CorrectionInputLayer=None
            
            CreateNewRadial=True
            CreateNewAngular=True
            CreateNewCorrection=True
            
            RadialForceNetworks=None
            AngularForceNetworks=None
            CorrectionForceNetworks=None
            # Read structures for specific network parts
            StructureForAtom=self.Structures[i]
            RadialStructure=StructureForAtom.RadialNetworkStructure
            AngularStructure=StructureForAtom.AngularNetworkStructure
            CorrectionStructure=StructureForAtom.CorrectionNetworkStructure
            #Construct networks out of loaded data
            if len(self.HiddenData) != 0:
                #Load data for atom
                WeightData=self.HiddenData[i]
                BiasData=self.BiasData[i]
                if len(WeightData.RadialNetworkData)>0:
                    
                    CreateNewRadial=False
                    #Recreate radial network             
                    RadialWeights = WeightData.RadialNetworkData
                    RadialBias = BiasData.RadialNetworkData
                    if len(RadialWeights) == len(RadialStructure) - 1:
                        RadialForceNetworks = make_force_networks(RadialStructure, RadialWeights, RadialBias)
                    else:
                        RadialForceNetworks = None
    
                    for j in range(1, len(RadialStructure)):
                        
                        ThisWeightData = RadialWeights[j - 1]
                        ThisBiasData = RadialBias[j - 1]
                        RadialNrIn = ThisWeightData.shape[0]
                        RadialNrHidden = ThisWeightData.shape[1]
                        RadialHiddenLayers.append(construct_hidden_layer(RadialNrIn, RadialNrHidden, self.HiddenType, ThisWeightData, self.BiasType,ThisBiasData,WeightData.RadialVariable))

                    NetworkHiddenLayers[0]=RadialHiddenLayers
                    NetworkWeights[0]=RadialWeights
                    NetworkBias[0]=RadialBias
                    
                if len(WeightData.AngularNetworkData)>0:   
                    
                    CreateNewAngular=False
                    #Recreate angular network             
                    AngularWeights = WeightData.AngularNetworkData
                    AngularBias = BiasData.AngularNetworkData
                    if len(AngularWeights) == len(AngularStructure) - 1:
                        AngularForceNetworks = make_force_networks(AngularStructure, AngularWeights, AngularBias)
                    else:
                        AngularForceNetworks = None
    
                    for j in range(1, len(AngularStructure)):
                        
                        ThisWeightData = AngularWeights[j - 1]
                        ThisBiasData = AngularBias[j - 1]
                        AngularNrIn = ThisWeightData.shape[0]
                        AngularNrHidden = ThisWeightData.shape[1]
                        AngularHiddenLayers.append(construct_hidden_layer(AngularNrIn, AngularNrHidden, self.HiddenType, ThisWeightData, self.BiasType,ThisBiasData,WeightData.AngularVariable))
                    
                    NetworkHiddenLayers[1]=AngularHiddenLayers   
                    NetworkWeights[1]=AngularWeights
                    NetworkBias[1]=AngularBias
                    
                if len(WeightData.CorrectionNetworkData)>0:   
                    CreateNewCorrection=False
                    #Recreate correction network             
                    CorrectionWeights = WeightData.CorrectionNetworkData
                    CorrectionBias = BiasData.CorrectionNetworkData
                    if len(CorrectionWeights) == len(CorrectionStructure) - 1:
                        CorrectionForceNetworks = make_force_networks(CorrectionStructure, CorrectionWeights, CorrectionBias)
                    else:
                        CorrectionForceNetworks = None
    
                    for j in range(1, len(CorrectionStructure)):
                        
                        ThisWeightData = CorrectionWeights[j - 1]
                        ThisBiasData = CorrectionBias[j - 1]
                        CorrectionNrIn = ThisWeightData.shape[0]
                        CorrectionNrHidden = ThisWeightData.shape[1]
                        CorrectionHiddenLayers.append(construct_hidden_layer(CorrectionNrIn, CorrectionNrHidden, self.HiddenType, ThisWeightData, self.BiasType,ThisBiasData,WeightData.CorrectionVariable))
                     
                    NetworkHiddenLayers[2]=CorrectionHiddenLayers
                    NetworkWeights[2]=CorrectionWeights
                    NetworkBias[2]=CorrectionBias
            
            if CreateNewRadial==True:
                #Create radial network
                for j in range(1, len(RadialStructure)):
                    RadialNrIn = RadialStructure[j - 1]
                    RadialNrHidden = RadialStructure[j]
                    RadialHiddenLayers.append(construct_hidden_layer(RadialNrIn, RadialNrHidden, self.HiddenType, [], self.BiasType))

                NetworkHiddenLayers[0]=RadialHiddenLayers
                
            if CreateNewAngular==True:
                #Create angular network
                for j in range(1, len(AngularStructure)):
                    AngularNrIn = AngularStructure[j - 1]
                    AngularNrHidden = AngularStructure[j]
                    AngularHiddenLayers.append(construct_hidden_layer(AngularNrIn, AngularNrHidden, self.HiddenType, [], self.BiasType))
                
                NetworkHiddenLayers[1]=AngularHiddenLayers
                
            if CreateNewCorrection==True:
                #Create correction network
                for j in range(1, len(CorrectionStructure)):
                    CorrectionNrIn = CorrectionStructure[j - 1]
                    CorrectionNrHidden = CorrectionStructure[j]
                    CorrectionHiddenLayers.append(construct_hidden_layer(CorrectionNrIn, CorrectionNrHidden, self.HiddenType, [], self.BiasType))
                
                NetworkHiddenLayers[2]=CorrectionHiddenLayers
                
            AllHiddenLayers.append(NetworkHiddenLayers)

            for k in range(0, self.NumberOfSameNetworks[i]):
                
                
                if len(RadialHiddenLayers)>0:
                    # Make radial input layer
                    if CreateNewRadial==False:
                        RadialHiddenData=self.HiddenData[i].RadialNetworkData
                        RadialNrInputs = RadialHiddenData[0].shape[0]
                    else:
                        RadialNrInputs = RadialStructure[0]
    
                    RadialInputLayer = construct_input_layer(RadialNrInputs)
                    # Connect radial input to first hidden layer
                    RadialFirstWeights = RadialHiddenLayers[0][0]
                    RadialFirstBiases = RadialHiddenLayers[0][1]
                    RadialNetwork = connect_layers(RadialInputLayer, RadialFirstWeights, RadialFirstBiases, self.ActFun, self.ActFunParam)
                    #Connect radial hidden layers
                    for l in range(1, len(RadialHiddenLayers)):
                        RadialTempWeights = RadialHiddenLayers[l][0]
                        RadialTempBiases = RadialHiddenLayers[l][1]
                        if l == len(RadialHiddenLayers) - 1:
                            RadialNetwork = connect_layers(RadialNetwork, RadialTempWeights, RadialTempBiases, "none", self.ActFunParam)
                        else:
                            RadialNetwork = connect_layers(RadialNetwork, RadialTempWeights, RadialTempBiases, self.ActFun, self.ActFunParam)
                    
                if len(AngularHiddenLayers)>0:
                    # Make angular input layer
                    if CreateNewAngular==False:
                        AngularHiddenData=self.HiddenData[i].AngularNetworkData
                        AngularNrInputs = AngularHiddenData[0].shape[0]
                    else:
                        AngularNrInputs = AngularStructure[0]
    
                    AngularInputLayer = construct_input_layer(AngularNrInputs)
                    # Connect angular input to first hidden layer
                    AngularFirstWeights = AngularHiddenLayers[0][0]
                    AngularFirstBiases = AngularHiddenLayers[0][1]
                    AngularNetwork = connect_layers(AngularInputLayer, AngularFirstWeights, AngularFirstBiases, self.ActFun, self.ActFunParam)
                    #Connect angular hidden layers
                    for l in range(1, len(AngularHiddenLayers)):
                        AngularTempWeights = AngularHiddenLayers[l][0]
                        AngularTempBiases = AngularHiddenLayers[l][1]
                        if l == len(AngularHiddenLayers) - 1:
                            AngularNetwork = connect_layers(AngularNetwork, AngularTempWeights, AngularTempBiases, "none", self.ActFunParam)
                        else:
                            AngularNetwork = connect_layers(AngularNetwork, AngularTempWeights, AngularTempBiases, self.ActFun, self.ActFunParam)
               
                if len(CorrectionHiddenLayers)>0:
                    # Make correction input layer
                    if CreateNewCorrection==False:
                        CorrectionHiddenData=self.HiddenData[i].CorrectionNetworkData
                        CorrectionNrInputs = CorrectionHiddenData[0].shape[0]
                    else:
                        CorrectionNrInputs = CorrectionStructure[0]
    
                    CorrectionInputLayer = construct_input_layer(CorrectionNrInputs)
                    # Connect Correction input to first hidden layer
                    CorrectionFirstWeights = CorrectionHiddenLayers[0][0]
                    CorrectionFirstBiases = CorrectionHiddenLayers[0][1]
                    CorrectionNetwork = connect_layers(CorrectionInputLayer, CorrectionFirstWeights, CorrectionFirstBiases, self.ActFun, self.ActFunParam)
                    #Connect Correction hidden layers
                    for l in range(1, len(CorrectionHiddenLayers)):
                        CorrectionTempWeights = CorrectionHiddenLayers[l][0]
                        CorrectionTempBiases = CorrectionHiddenLayers[l][1]
                        if l == len(CorrectionHiddenLayers) - 1:
                            CorrectionNetwork = connect_layers(CorrectionNetwork, CorrectionTempWeights, CorrectionTempBiases, "none", self.ActFunParam)
                        else:
                            CorrectionNetwork = connect_layers(CorrectionNetwork, CorrectionTempWeights, CorrectionTempBiases, self.ActFun, self.ActFunParam)
     
                
                #Store all networks
                Network=range(3)
                if RadialNetwork!=None :
                    Network[0]=RadialNetwork
                if AngularNetwork!=None:
                    Network[1]=AngularNetwork
                if CorrectionNetwork!=None:
                    Network[2]=CorrectionNetwork

                #Store all force networks
                ForceNetworks=list()
                if RadialForceNetworks!=None :
                    ForceNetworks.append(RadialForceNetworks)
                if AngularForceNetworks!=None:
                    ForceNetworks.append(AngularForceNetworks)
                if CorrectionForceNetworks!=None:
                    ForceNetworks.append(CorrectionForceNetworks)
                    
                #Store all weights
                RawWeights=list()
                if len(RadialWeights)>0:
                    RawWeights.append(RadialWeights)
                if len(AngularWeights)>0:
                    RawWeights.append(AngularWeights)
                if len(CorrectionWeights)>0:
                    RawWeights.append(CorrectionWeights)
                    
                #Store all bias
                RawBias=list()
                if len(RadialBias)>0:
                    RawBias.append(RadialBias)
                if len(AngularBias)>0:
                    RawBias.append(AngularBias)
                if len(CorrectionBias)>0:
                    RawBias.append(CorrectionBias)
                    
                #Store all input layers
                InputLayer=range(3)
                if RadialInputLayer!=None :
                    InputLayer[0]=RadialInputLayer
                if AngularInputLayer!=None:
                    InputLayer[1]=AngularInputLayer
                if CorrectionInputLayer!=None:
                    InputLayer[2]=CorrectionInputLayer
                
                #Always none AtomicNNs dont need a specific output layer (removal would cause indexing problems)
                OutputLayer=None
                
                if len(self.Gs) != 0:
                    if len(self.Gs) > 0:
                        AtomicNNs.append(
                            [self.NumberOfSameNetworks[i], Network, InputLayer, OutputLayer, ForceNetworks, RawWeights,
                             RawBias, self.ActFun, self.Gs[i]])
                    else:
                        AtomicNNs.append(
                            [self.NumberOfSameNetworks[i], Network, InputLayer, OutputLayer, ForceNetworks, RawWeights,
                             RawBias, self.ActFun])
                else:
                    AtomicNNs.append(
                        [self.NumberOfSameNetworks[i], Network, InputLayer, OutputLayer, ForceNetworks, RawWeights, RawBias,
                         self.ActFun])

        
        self.AtomicNNs=AtomicNNs
        self.VariablesDictionary=AllHiddenLayers

    def make_atomic_networks(self):

        AllHiddenLayers = list()
        AtomicNNs = list()
        # Start Session
        if self.Multiple==False:
            self.Session=tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=multiprocessing.cpu_count()))
            
        OldBiasNr = 0
        OldShape = None

        # make all the networks for the different atom types
        for i in range(0, len(self.Structures)):
            # Make hidden layers
            HiddenLayers = list()
            Structure = self.Structures[i]
            if len(self.HiddenData) != 0:
                RawWeights = self.HiddenData[i]
                RawBias = self.BiasData[i]
                if len(RawWeights) == len(Structure) - 1:
                    ForceNetworks = make_force_networks(Structure, RawWeights, RawBias)
                else:
                    ForceNetworks = None

                for j in range(1, len(Structure)):
                    NrIn = Structure[j - 1]
                    NrHidden = Structure[j]

                    if j == len(Structure) - 1 and self.MakeLastLayerConstant == True:
                        HiddenLayers.append(construct_not_trainable_layer(NrIn, NrHidden, self.MinOfOut))
                    else:
                        if j >= len(self.HiddenData[i]) and self.MakeLastLayerConstant == True:
                            tempWeights, tempBias = construct_hidden_layer(NrIn, NrHidden, self.HiddenType, [], self.BiasType, [],
                                                                           self.MakeAllVariable, self.InitMean, self.InitStddev)

                            indices = []
                            values = []
                            thisShape = tempWeights.get_shape().as_list()

                            for q in range(0, OldBiasNr):
                                indices.append([q, q])
                                values += [1.0]

                            delta = tf.SparseTensor(indices, values, thisShape)
                            tempWeights = tempWeights + tf.sparse_tensor_to_dense(delta)

                            HiddenLayers.append([tempWeights, tempBias])
                        else:
                            OldBiasNr = len(self.BiasData[i][j - 1])
                            OldShape = self.HiddenData[i][j - 1].shape
                            # fill old weights in new structure
                            if OldBiasNr < NrHidden:
                                ThisWeightData = np.random.normal(loc=0.0, scale=0.01, size=(NrIn, NrHidden))
                                ThisWeightData[0:OldShape[0], 0:OldShape[1]] = self.HiddenData[i][j - 1]
                                ThisBiasData = np.zeros([NrHidden])
                                ThisBiasData[0:OldBiasNr] = self.BiasData[i][j - 1]
                            elif OldBiasNr>NrHidden:
                                ThisWeightData = np.zeros((NrIn, NrHidden))
                                ThisWeightData[0:, 0:] = self.HiddenData[i][j - 1][0:NrIn,0:NrHidden]
                                ThisBiasData = np.zeros([NrHidden])
                                ThisBiasData[0:OldBiasNr] = self.BiasData[i][j - 1][0:NrIn,0:NrHidden]
                            else:
                                ThisWeightData = self.HiddenData[i][j - 1]
                                ThisBiasData = self.BiasData[i][j - 1]

                            HiddenLayers.append(
                                construct_hidden_layer(NrIn, NrHidden, self.HiddenType, ThisWeightData, self.BiasType,
                                                       ThisBiasData, self.MakeAllVariable))

            else:
                RawWeights = None
                RawBias = None
                ForceNetworks = None
                for j in range(1, len(Structure)):
                    NrIn = Structure[j - 1]
                    NrHidden = Structure[j]
                    if j == len(Structure) - 1 and self.MakeLastLayerConstant == True:
                        HiddenLayers.append(construct_not_trainable_layer(NrIn, NrHidden, self.MinOfOut))
                    else:
                        HiddenLayers.append(construct_hidden_layer(NrIn, NrHidden, self.HiddenType, [], self.BiasType))

            AllHiddenLayers.append(HiddenLayers)

            for k in range(0, self.NumberOfSameNetworks[i]):
                # Make input layer
                if len(self.HiddenData) != 0:
                    NrInputs = self.HiddenData[i][0].shape[0]
                else:
                    NrInputs = Structure[0]

                InputLayer = construct_input_layer(NrInputs)
                # Make output layer
                OutputLayer = construct_output_layer(Structure[-1])
                # Connect input to first hidden layer
                FirstWeights = HiddenLayers[0][0]
                FirstBiases = HiddenLayers[0][1]
                Network = connect_layers(InputLayer, FirstWeights, FirstBiases, self.ActFun, self.ActFunParam)

                for l in range(1, len(HiddenLayers)):
                    # Connect ouput of in layer to second hidden layer

                    if l == len(HiddenLayers) - 1:
                        Weights = HiddenLayers[l][0]
                        Biases = HiddenLayers[l][1]
                        Network = connect_layers(Network, Weights, Biases, "none", self.ActFunParam)
                    else:
                        Weights = HiddenLayers[l][0]
                        Biases = HiddenLayers[l][1]
                        Network = connect_layers(Network, Weights, Biases, self.ActFun, self.ActFunParam)

                if len(self.Gs) != 0:
                    if len(self.Gs) > 0:
                        AtomicNNs.append(
                            [self.NumberOfSameNetworks[i], Network, InputLayer, OutputLayer, ForceNetworks, RawWeights,
                             RawBias, self.ActFun, self.Gs[i]])
                    else:
                        AtomicNNs.append(
                            [self.NumberOfSameNetworks[i], Network, InputLayer, OutputLayer, ForceNetworks, RawWeights,
                             RawBias, self.ActFun])
                else:
                    AtomicNNs.append(
                        [self.NumberOfSameNetworks[i], Network, InputLayer, OutputLayer, ForceNetworks, RawWeights, RawBias,
                         self.ActFun])

        
        self.AtomicNNs=AtomicNNs
        self.VariablesDictionary=AllHiddenLayers
        
        
class MultipleInstanceTraining(object):
    
    def __init__(self):
        #Training variables
        self.TrainingInstances=list()
        self.EpochsPerCycle=1
        self.GlobalEpochs=100
        self.GlobalStructures=list()
        self.GlobalLearningRate=0.001
        self.GlobalCostCriterium=0.0001
        self.GlobalRegularization="L2"
        self.GlobalRegularizationParam=0.0001
        self.GlobalOptimizer="Adam"
        self.GlobalTrainingCosts=list()
        self.GlobalValidationCosts=list()
        self.GlobalMinOfOut=0
        self.MakePlots=False
        self.GlobalSession=tf.Session()
        
    def initialize_multiple_instances(self):
        
        Execute=True
        if len(self.TrainingInstances)==0:
            Execute=False
            print("No training instances available!")
            
        if Execute==True:
            #Initialize all instances with same settings
            for Instance in self.TrainingInstances:
                Instance.Multiple=True
                Instance.Epochs=self.EpochsPerCycle
                Instance.MakeAllVariable=True
                Instance.Structures=self.GlobalStructures
                Instance.Session = self.GlobalSession
                Instance.MakePlots=False
                Instance.ActFun="relu"
                Instance.CostCriterium=0
                Instance.HiddenType="truncated_normal"
                Instance.LearningRate=self.GlobalLearningRate
                Instance.OptimizerType=self.GlobalOptimizer
                Instance.Regularization=self.GlobalRegularization
                Instance.RegularizationParam=self.GlobalRegularizationParam
                if Instance.MinOfOut <self.GlobalMinOfOut:
                    self.GlobalMinOfOut=Instance.MinOfOut
                    
                #Clear unnecessary data
                Instance.Ds.geometries=list()
                Instance.Ds.Energies=list()
                Instance.Batches=list()
                Instance.AllGeometries=list()
            #Write global minimum to all instances
            for Instance in self.TrainingInstances:
                Instance.MinOfOut=self.GlobalMinOfOut
                
    def set_session(self):
        for Instance in self.TrainingInstances:
            Instance.Session = self.GlobalSession

    def train_multiple_instances(self,StartModelName=None):
        print("Startet multiple instance training!")
        ct=0
        LastStepsModelData=list()
        for i in range(0,self.GlobalEpochs):
            AllConverged=True
            for Instance in self.TrainingInstances:
                if ct==0:
                    if StartModelName!=None:
                        Instance.expand_existing_net(ModelName=StartModelName)
                    else:
                        Instance.make_and_initialize_network()
                else:
                    Instance.expand_existing_net(ModelData=LastStepsModelData)
                
                LastStepsModelData=Instance.start_batch_training()
                tf.reset_default_graph()
                self.GlobalSession=tf.Session()
                MultipleInstanceTraining.set_session(self)
                self.GlobalTrainingCosts+=Instance.OverallTrainingCosts
                self.GlobalValidationCosts+=Instance.OverallValidationCosts
                if ct % max(int((self.GlobalEpochs*len(self.TrainingInstances))/50),1)==0 or i==(self.GlobalEpochs-1):
                    if self.MakePlots==True:
                        if ct ==0:
                            fig,ax,TrainingCostPlot,ValidationCostPlot=initialize_cost_plot(self.GlobalTrainingCosts,self.GlobalValidationCosts)
                        else:
                            update_cost_plot(fig,ax,TrainingCostPlot,self.GlobalTrainingCosts,ValidationCostPlot,self.GlobalValidationCosts)
                    
                    #Finished percentage output
                    print(str(100*ct/(self.GlobalEpochs*len(self.TrainingInstances)))+" %")
                    np.save("trained_variables",LastStepsModelData)
                ct=ct+1
                #Abort criteria
                if Instance.TrainingCosts<=self.GlobalCostCriterium and Instance.ValidationCosts<=self.GlobalCostCriterium and AllConverged==True:
                    AllConverged=True
                else:
                    AllConverged=False
                
            if AllConverged==True:
                print("Reached cost criterium: "+str(self.GlobalTrainingCosts))
                break     
                    
                
class PartitionedStructure(object):
    
     def __init__(self):
         
         self.RadialNetworkStructure=list()
         self.AngularNetworkStructure=list()
         self.CorrectionNetworkStructure=list()
         
                
class PartitionedNetworkData(object):
    
     def __init__(self):
         
         self.RadialNetworkData=list()
         self.AngularNetworkData=list()
         self.CorrectionNetworkData=list()
         self.RadialVariable=False
         self.AngularVariable=False
         self.CorrectionVariable=False
         
class PartitionedGs(object):
    
     def __init__(self):
         
         self.RadialGs=list()
         self.AngularGs=list()
         self.CorrectionGs=list()
                