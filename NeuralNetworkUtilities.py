#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:30:10 2017

@author: Fuchs Alexander
"""
import numpy as _np
import tensorflow as _tf
import DataSet as _DataSet
import SymmetryFunctionSet as _SymmetryFunctionSet
import random as _rand
import matplotlib.pyplot as _plt
import multiprocessing as _multiprocessing
import time as _time 
import os as _os


_plt.ion()
_tf.reset_default_graph()

#Construct input for the NN.
def _construct_input_layer(InputUnits):
    
    Inputs=_tf.placeholder(_tf.float32, shape=[None, InputUnits])

    return Inputs

#Construct the weights for this layer.
def _construct_hidden_layer(LayerBeforeUnits,HiddenUnits,InitType=None,InitData=[],BiasType=None,BiasData=[],MakeAllVariable=False,Mean=0.0,Stddev=1.0):
    
    if len(InitData)==0:
        if InitType!=None:
            if InitType == "zeros":
                Weights=_tf.Variable(_tf.zeros([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
            elif InitType =="ones":
                Weights=_tf.Variable(_tf.ones([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
            elif InitType == "fill":
                Weights=_tf.Variable(_tf.fill([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
            elif InitType == "random_normal":
                Weights=_tf.Variable(_tf.random_normal([LayerBeforeUnits,HiddenUnits],mean=Mean,stddev=Stddev),dtype=_tf.float32,name="variable")
            elif InitType == "truncated_normal":
                Weights=_tf.Variable(_tf.truncated_normal([LayerBeforeUnits,HiddenUnits],mean=Mean,stddev=Stddev),dtype=_tf.float32,name="variable")
            elif InitType == "random_uniform":
                Weights=_tf.Variable(_tf.random_uniform([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
            elif InitType == "random_shuffle":
                Weights=_tf.Variable(_tf.random_shuffle([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
            elif InitType == "random_crop":
                Weights=_tf.Variable(_tf.random_crop([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
            elif InitType == "random_gamma":
                Weights=_tf.Variable(_tf.random_gamma([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
            else:
                #Assume random weights if no InitType is given
                Weights=_tf.Variable(_tf.random_uniform([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
        else:
            #Assume random weights if no InitType is given
            Weights=_tf.Variable(_tf.random_uniform([LayerBeforeUnits,HiddenUnits]),dtype=_tf.float32,name="variable")
    else:
        if MakeAllVariable==False:
            Weights=_tf.constant(InitData,dtype=_tf.float32,name="constant")
        else:
            Weights=_tf.Variable(InitData,dtype=_tf.float32,name="variable")
    #Construct the bias for this layer
    if len(BiasData)!=0:

        if MakeAllVariable==False:
            Biases=_tf.constant(BiasData,dtype=_tf.float32,name="bias")
        else:
            Biases=_tf.Variable(BiasData,dtype=_tf.float32,name="bias")

    else:
        if InitType == "zeros":
            Biases=_tf.Variable(_tf.zeros([HiddenUnits]),dtype=_tf.float32,name="bias")
        elif InitType =="ones":
            Biases=_tf.Variable(_tf.ones([HiddenUnits]),dtype=_tf.float32,name="bias")
        elif InitType == "fill":
            Biases=_tf.Variable(_tf.fill([HiddenUnits],BiasData),dtype=_tf.float32,name="bias")
        elif InitType == "random_normal":
            Biases=_tf.Variable(_tf.random_normal([HiddenUnits],mean=Mean,stddev=Stddev),dtype=_tf.float32,name="bias")
        elif InitType == "truncated_normal":
            Biases=_tf.Variable(_tf.truncated_normal([HiddenUnits],mean=Mean,stddev=Stddev),dtype=_tf.float32,name="bias")
        elif InitType == "random_uniform":
            Biases=_tf.Variable(_tf.random_uniform([HiddenUnits]),dtype=_tf.float32,name="bias")
        elif InitType == "random_shuffle":
            Biases=_tf.Variable(_tf.random_shuffle([HiddenUnits]),dtype=_tf.float32,name="bias")
        elif InitType == "random_crop":
            Biases=_tf.Variable(_tf.random_crop([HiddenUnits],BiasData),dtype=_tf.float32,name="bias")
        elif InitType == "random_gamma":
            Biases=_tf.Variable(_tf.random_gamma([HiddenUnits],InitData),dtype=_tf.float32,name="bias")
        else:
            Biases = _tf.Variable(_tf.random_uniform([HiddenUnits]),dtype=_tf.float32,name="bias")
    
    return Weights,Biases

#Construct the output layer for the NN.
def _construct_output_layer(OutputUnits):
    
    Outputs = _tf.placeholder(_tf.float32, shape=[None, OutputUnits])

    return Outputs


#Make a not trainable layer with all weights being 1.
#Returns two tensors(weights,biases).
def _construct_not_trainable_layer(NrInputs,NrOutputs,Min):


    Weights=_tf.constant(_np.ones([NrInputs,NrOutputs]),dtype=_tf.float32)#, trainable=False)
    Biases=_tf.constant(_np.zeros([NrOutputs]),dtype=_tf.float32)#,trainable=False)
    if Min!=0:
        Biases=_tf.add(Biases,Min/NrOutputs)

    return Weights,Biases

 #Connect the outputs of the layer before to the current layer using an 
 #activation function.
 #Returns the connected tensor. 
def _connect_layers(InputsForLayer,Layer1Weights,Layer1Bias,ActFun=None,FunParam=None,Dropout=0):
   
    if ActFun!=None:
        if ActFun=="sigmoid":
            Out=_tf.nn.sigmoid(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="tanh":
            Out=_tf.nn.tanh(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="relu":
            Out=_tf.nn.relu(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="relu6":
            Out=_tf.nn.relu6(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="crelu":
            Out=_tf.nn.crelu(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="elu":
            Out=_tf.nn.elu(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="softplus":
            Out=_tf.nn.softplus(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        elif ActFun=="dropout":
            Out=_tf.nn.dropout(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias,FunParam)
        elif ActFun=="bias_add":
            Out=_tf.nn.bias_add(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias,FunParam)
        elif ActFun == "none":
            Out = _tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias
    else:
        Out=_tf.nn.sigmoid(_tf.matmul(InputsForLayer, Layer1Weights) + Layer1Bias)
        
    if Dropout!=0:
        #Apply dropout between layers   
        Out=_tf.nn.dropout(Out,Dropout)

    return Out

#Construct a basic NN.
#Returns a tensor for the NN outout and the feeding placeholders for intput
#and output,
def make_feed_forward_neuralnetwork(Structure,HiddenType=None,HiddenData=None,BiasType=None,BiasData=None,ActFun=None,ActFunParam=None,Dropout=0):

    #Make inputs
    NrInputs=Structure[0]
    InputLayer=_construct_input_layer(NrInputs)
    #Make hidden layers
    HiddenLayers=list()
    for i in range(1,len(Structure)):
        NrIn=Structure[i-1]
        NrHidden=Structure[i]
        HiddenLayers.append(_construct_hidden_layer(NrIn,NrHidden,HiddenType,HiddenData[i-1],BiasType,BiasData[i-1]))


    #Make output layer
    OutputLayer=_construct_output_layer(Structure[-1])

   #Connect input to first hidden layer
    FirstWeights=HiddenLayers[0][0]
    FirstBiases=HiddenLayers[0][1]
    InConnection=_connect_layers(InputLayer,FirstWeights,FirstBiases,ActFun,ActFunParam,Dropout)
    Network=InConnection

    for j in range(1,len(HiddenLayers)):
       #Connect ouput of in layer to second hidden layer
        if j==1 :
            SecondWeights=HiddenLayers[j][0]
            SecondBiases=HiddenLayers[j][1]
            Network=_connect_layers(Network,SecondWeights,SecondBiases,ActFun,ActFunParam,Dropout)
        else:
            Weights=HiddenLayers[j][0]
            Biases=HiddenLayers[j][1]
            Network=_connect_layers(Network,Weights,Biases,ActFun,ActFunParam,Dropout)

    return Network,InputLayer,OutputLayer


 #Returns the cost function as a tensor.
def cost_for_network(Prediction,ReferenceValue,Type):

    if Type=="squared-difference":
        Cost=0.5*_tf.reduce_sum((Prediction-ReferenceValue)**2)
    elif Type=="Adaptive_1":
        epsilon=10e-9
        Cost=0.5*_tf.reduce_sum((Prediction-ReferenceValue)**2\
                                *(_tf.sigmoid(_tf.abs(Prediction-ReferenceValue+epsilon))-0.5)\
                                +(0.5+_tf.sigmoid(_tf.abs(Prediction-ReferenceValue+epsilon)))\
                                *_tf.pow(_tf.abs(Prediction-ReferenceValue+epsilon),1.25))
    elif Type=="Adaptive_2":
        epsilon=10e-9
        Cost=0.5*_tf.reduce_sum((Prediction-ReferenceValue)**2\
                                *(_tf.sigmoid(_tf.abs(Prediction-ReferenceValue+epsilon))-0.5)\
                                +(0.5+_tf.sigmoid(_tf.abs(Prediction-ReferenceValue+epsilon)))\
                                *_tf.abs(Prediction-ReferenceValue+epsilon))

    return Cost



#Calculates the runnung average over N steps.
def running_mean(x,N):
    cumsum=_np.cumsum(_np.insert(x,0,0))
    return (cumsum[N:]-cumsum[:-N])/N

#Initializes the cost plot for the training 
#Returns the figure and the different plots 
def _initialize_cost_plot(TrainingData,ValidationData=[]):

    fig=_plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscaley_on(True)
    TrainingCostPlot, = ax.semilogy(_np.arange(0,len(TrainingData)),TrainingData)
    if len(ValidationData)!=0:
        ValidationCostPlot,=ax.semilogy(_np.arange(0,len(ValidationData)),ValidationData)
    else:
        ValidationCostPlot=None
    #add running average plot 
    running_avg=running_mean(TrainingData,1000)
    RunningMeanPlot,=ax.semilogy(_np.arange(0,len(running_avg)),running_avg)

    #Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel("batches")
    ax.set_ylabel("log(cost)")
    ax.set_title("Normalized cost per batch")

    #We need to draw *and* flush
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig,ax,TrainingCostPlot,ValidationCostPlot,RunningMeanPlot

#Updates the cost plot with new data
def _update_cost_plot(figure,ax,TrainingCostPlot,TrainingCost,ValidationCostPlot=None,ValidationCost=None,RunningMeanPlot=None):

    TrainingCostPlot.set_data(_np.arange(0,len(TrainingCost)),TrainingCost)
    if ValidationCostPlot!=None:
        ValidationCostPlot.set_data(_np.arange(0,len(ValidationCost)),ValidationCost)
    
    if RunningMeanPlot != None:
        running_avg=running_mean(TrainingCost,1000)
        RunningMeanPlot.set_data(_np.arange(0,len(running_avg)),running_avg)
    #Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    #We need to draw *and* flush
    figure.canvas.draw()
    figure.canvas.flush_events()

#Initializes the plot of the absolute value of the weights 
#Can be used to identify redundant symmetry functions
#Returns the figure and the plot
def _initialize_weights_plot(sparse_weights,n_gs):
    fig=_plt.figure()
    ax = fig.add_subplot(111)
    weights_plot=ax.bar(_np.arange(n_gs),sparse_weights)
    ax.set_autoscaley_on(True)
    
    #Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel("Symmetry function")
    ax.set_ylabel("Weights")
    ax.set_title("Weights for symmetry functions")
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig,weights_plot

#Updates the plot of the absolute value of the weights 
#Can be used to identify redundant symmetry functions
#Returns the figure and the plot
def _update_weights_plot(fig,weights_plot,sparse_weights):
    for u,rect in enumerate(weights_plot):
        rect.set_height(sparse_weights[u])
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig,weights_plot

#Converts cartesion to spherical coordinates
#Returns the spherical coordinates
def cartesian_to_spherical(xyz):
    spherical = _np.zeros_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    spherical[:,0] = _np.sqrt(xy + xyz[:,2]**2)
    spherical[:,1] = _np.arctan2(xyz[:,2], _np.sqrt(xy))
    spherical[:,2] = _np.arctan2(xyz[:,1], xyz[:,0])
    return spherical

#Returns a tensor for the learning rate 
#Learning rate can be decayed with different methods.
#Returns the tensors for the global step and the learning reate
def _get_learning_rate(StartLearningRate,LearningRateType,decay_steps,boundaries=[],values=[]):
    
    if LearningRateType=="none":
        global_step = _tf.Variable(0, trainable=False)
        return global_step,StartLearningRate
    elif LearningRateType=="exponential_decay":
        global_step = _tf.Variable(0, trainable=False)
        return global_step,_tf.train.exponential_decay(StartLearningRate, global_step, decay_steps, decay_rate=0.96, staircase=False)
    elif LearningRateType=="inverse_time_decay":
        global_step = _tf.Variable(0, trainable=False)
        return global_step,_tf.train.inverse_time_decay(StartLearningRate, global_step, decay_steps,  decay_rate=0.96, staircase=False)
    elif LearningRateType=="piecewise_constant":
        global_step = _tf.Variable(0, trainable=False)
        return global_step,_tf.train.piecewise_constant(global_step, boundaries, values)
    elif LearningRateType=="polynomial_decay_p1":
        global_step = _tf.Variable(0, trainable=False)
        return global_step,_tf.train.polynomial_decay(StartLearningRate, global_step, decay_steps, end_learning_rate=0.00001, power=1.0, cycle=False)
    elif LearningRateType=="polynomial_decay_p2":
        global_step = _tf.Variable(0, trainable=False)
        return global_step,_tf.train.polynomial_decay(StartLearningRate, global_step, decay_steps, end_learning_rate=0.00001, power=2.0, cycle=False)

#Calculates the distances between all atoms
#Returns a matrix with all distances.
def _calc_distance_to_all_atoms(xyz):
    
    Ngeom=len(xyz[:,0])
    distances=_np.zeros((Ngeom,Ngeom))
    for i in range(Ngeom):
        distances[i,:]=_np.linalg.norm(xyz-xyz[i,:])
    
    return distances

#Gets the maximum and minimum radial distance within the dataset
#Returns a min,max value as floats.
def _get_ds_r_min_r_max(geoms):
    r_min=10e10
    r_max=0
    for geom in geoms:
        np_geom=_np.zeros((len(geom),3))
        np_dist=_np.zeros((len(geom),len(geom)))
        for i,atom in enumerate(geom):
            xyz=atom[1]
            np_geom[i,:]=xyz
        np_dist=_calc_distance_to_all_atoms(np_geom)
        L=np_dist!=0
        r_min_tmp=_np.min(np_dist[L])
        r_max_tmp=_np.max(np_dist)
        if r_min_tmp<r_min:
            r_min= r_min_tmp

        if r_max_tmp>r_max:
            r_max=r_max_tmp
            
    return r_min,r_max


#Creates geometries outside the dataset for a fixed energy
#to prevent the network from performing badly outside the training area.
#The radial coverage area of the training data has to be specified via
#r_min and r_max, the types of atoms as a list: ["species1",species2] and 
#the number of atoms per atom type have to be specified as a list:[1,2]
#Returns N geometries as a list.
def _create_zero_diff_geometries(r_min,r_max,types,N_atoms_per_type,N):
    
    out_geoms=[]
    Natoms=sum(N_atoms_per_type)
    np_geom=_np.zeros((Natoms,3))
    np_dist=_np.zeros((Natoms,Natoms))
    for i in range(N):
        #try to get valid position for atom
        run=True
        while(run):
            #calucalte random position of atom
            geom=[]
            ct=0
            for j in range(len(N_atoms_per_type)):
                for k in range(N_atoms_per_type[j]):
                    r=_rand.uniform(0,5*r_max)
                    phi=_rand.uniform(0,2*_np.pi)
                    theta=_rand.uniform(0,_np.pi)
                    x=r*_np.cos(theta)*_np.cos(phi)
                    y=r*_np.cos(theta)*_np.sin(phi)
                    z=r*_np.sin(theta)
                    xyz=[x,y,z]
                    a_type=types[j]
                    atom=(a_type,_np.asarray(xyz))
                    np_geom[ct,:]=xyz
                    ct+=1
                    geom.append(atom)
                
            np_dist=_calc_distance_to_all_atoms(np_geom)
            L=np_dist!=0
            if _np.all(np_dist[L])>r_max or _np.all(np_dist[L])<r_min:
                out_geoms.append(geom)
                run=False

       
        
    return out_geoms

#Parse q-chem format to NN compatible format
#Returns the geometries in a compatible list
def parse_qchem_geometries(in_geoms):
    out_geoms=[]
    for in_geom in in_geoms:
        atoms_list=in_geom.list_of_atoms
        out_geom=[]
        for atom in atoms_list:
            xyz=[float(atom[1]),float(atom[2]),float(atom[3])]
            my_tuple=(atom[0],_np.asarray(xyz))
            out_geom.append(my_tuple)
        out_geoms.append(out_geom)
                
    return out_geoms


#This class implements all the properties and methods for training and 
#evaluating the network
class AtomicNeuralNetInstance(object):

    def __init__(self):
        #Training variables
        self.Net=None
        self.Structures=[]
        self.NumberOfAtomsPerType=[]
        self.TrainingInputs=[]
        self.TrainingOutputs=[]
        self.Epochs=1000
        self.GlobalStep=None
        self.LearningRate=0.01
        self.LearningRateFun=None
        self.LearningRateType="none"
        self.LearningDecayEpochs=100
        self.LearningRateBounds=[]
        self.LearningRateValues=[]
        self.ValidationInputs=[]
        self.ValidationOutputs=[]
        self.Gs=[]
        self.ForceTrainingInput=[]
        self.ForceValidationInput=[]
        self.ForceTrainingOutput=[]
        self.ForceValidationOutput=[]
        self.HiddenType="truncated_normal"
        self.HiddenData=[]
        self.BiasType="zeros"
        self.BiasData=[]
        self.TrainingBatches=[]
        self.ValidationBatches=[]
        self.ActFun="elu"
        self.ActFunParam=None
        self.CostCriterium=0
        self.dE_Criterium=0
        self.OptimizerType=None
        self.OptimizerProp=None
        self.Session=[]
        self.TrainedNetwork=[]
        self.TrainingCosts=[]
        self.ValidationCosts=[]
        self.OverallTrainingCosts=[]
        self.OverallValidationCosts=[]
        self.TrainedVariables=[]
        self.RegLoss=0
        self.CostFun=None
        self.Optimizer=None
        self.OutputLayer=None
        self.OutputForce=None
        self.TotalEnergy=None
        self.dEi_dGij=[]
        self.MakePlots=False
        self.InitMean=0.0
        self.InitStddev=1.0
        self.MakeLastLayerConstant=False
        self.MakeAllVariable=True
        self.Regularization="none"
        self.RegularizationParam=0.0
        self.Dropout=[0]
        self.DeltaE=0
        self.dE_Fun=None
        self.CurrentEpochNr=0
        self.IsPartitioned=False
        self.CostFunType="squared-difference"
        self.Prediction=None
        self.OutputLayerForce=None
        #Data variables
        self.AllGeometries=[]
        self.AllGDerivatives=[]
        self.SizeOfInputsPerType=[]
        self.SizeOfInputsPerAtom=[]
        
        self.XYZfile=None
        self.Logfile=None
        self.atomtypes=[]
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
        self.InputDerivatives=False
        self.EvalData=[]

        #Other
        self.Multiple=False
        self.UseForce=False
        self.FirstWeights=[]
        self.SavingDirectory="save"


    def initialize_network(self):
        """Initializes the network for training by starting a session and getting 
        the placeholder for the output, the cost function, the learning rate and 
        the optimizer.
        """
        try:
            #Make virtual output layer for feeding the data to the cost function
            self.OutputLayer=_construct_output_layer(1)
            if self.UseForce:
                self.OutputLayerForce=_construct_output_layer(sum(self.NumberOfAtomsPerType)*3)
            #Cost function for whole net
            self.CostFun=self._atomic_cost_function()
            
            #if self.IsPartitioned==True:
            All_Vars=_tf.trainable_variables()
            decay_steps=len(self.TrainingBatches)*self.LearningDecayEpochs
            self.GlobalStep,self.LearningRateFun=_get_learning_rate(self.LearningRate,self.LearningRateType,decay_steps,self.LearningRateBounds,self.LearningRateValues)
            
            #Set optimizer
            if self.OptimizerType==None:
               self.Optimizer=_tf.train.GradientDescentOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
            else:
                if self.OptimizerType=="GradientDescent":
                    self.Optimizer=_tf.train.GradientDescentOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="Adagrad":
                    self.Optimizer=_tf.train.AdagradOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="Adadelta":
                    self.Optimizer=_tf.train.AdadeltaOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="AdagradDA":
                    self.Optimizer=_tf.train.AdagradDAOptimizer(self.LearningRateFun,self.OptimizerProp).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="Momentum":
                    self.Optimizer=_tf.train.MomentumOptimizer(self.LearningRateFun,self.OptimizerProp).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="Adam":
                    self.Optimizer=_tf.train.AdamOptimizer(self.LearningRateFun, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="Ftrl":
                   self.Optimizer=_tf.train.FtrlOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="ProximalGradientDescent":
                    self.Optimizer=_tf.train.ProximalGradientDescentOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="ProximalAdagrad":
                    self.Optimizer=_tf.train.ProximalAdagradOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                elif self.OptimizerType=="RMSProp":
                    self.Optimizer=_tf.train.RMSPropOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
                else:
                    self.Optimizer=_tf.train.GradientDescentOptimizer(self.LearningRateFun).minimize(self.CostFun,var_list=All_Vars,global_step=self.GlobalStep)
        except:
            print("Evaluation only no training supported if all networks are constant!")
        #Initialize session
        self.Session.run(_tf.global_variables_initializer())
        
    
    def load_model(self,ModelName="save/trained_variables"):
        """Loads the model in the specified folder.
        Args:
            ModelName(str):Path to the model.
        Returns:
            1
        """
        if ".npy" not in ModelName:
            ModelName=ModelName+".npy"
            rare_model=_np.load(ModelName)
            self.TrainedVariables=rare_model[0]
            self.MeansOfDs=rare_model[1]
            self.VarianceOfDs=rare_model[2]
            self.MinOfOut=rare_model[3]

        return 1
    

    def expand_existing_net(self,ModelName="save/trained_variables",MakeAllVariable=True,ModelData=None,ConvertToPartitioned=False):
        """Creates a new network out of stored data.
        Args:
            MakeAllVariables(bool): Specifies if all layers can be trained
            ModelData (list): Passes the model directly from a training before. 
            ConvertToPartitioned(bool):Converts a StandardAtomicNetwork to a 
            PartitionedAtomicNetwork network with the StandardAtomicNetwork
            beeing the force network part."""
    
        if ModelData==None:
            Success=self.load_model(ModelName)
        else:
            self.TrainedVariables=ModelData[0]
            self.MinOfOut=ModelData[1]
            Success=1
        if Success==1:                
            print("Model successfully loaded!")
            
            if self.IsPartitioned==False:
                if ConvertToPartitioned:
                    raise(ValueError)
                self.Net=_StandardAtomicNetwork()
            else:
                self.Net=_PartitionedAtomicNetwork() 
                
            if ConvertToPartitioned:
                self.HiddenData,self.BiasData=self._convert_standard_to_partitioned_net()
            else:
                self.HiddenData,self.BiasData=self.Net.get_weights_biases_from_data(self.TrainedVariables,self.Multiple)

            self.HiddenType="truncated_normal"
            self.InitMean=0
            self.InitStddev=0.01
            self.BiasType="zeros"
            self.MakeAllVariable=MakeAllVariable
            #try:
            self.make_and_initialize_network(self)
            #except:
            #    print("Partitioned network loaded, please set IsPartitioned=True")
            
    def _convert_standard_to_partitioned_net(self):
        """Converts a standard (Behler) network to the force-field part of  a
        partitioned network.
        
        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""
            
        WeightData,BiasData=self.Net.get_weights_biases_from_data(self.TrainedVariables,self.Multiple)
        OutWeights=list()
        OutBiases=list()
        for i in range(0,len(self.TrainedVariables)):
            Network=self.TrainedVariables[i]
            DataStructWeights=_PartitionedNetworkData() 
            DataStructBiases=_PartitionedNetworkData() 
            for j in range(0,len(Network)):
                Weights=WeightData[i][j]
                Biases=BiasData[i][j]
                DataStructWeights.ForceFieldNetworkData.append(Weights)
                DataStructBiases.ForceFieldNetworkData.append(Biases)
                        
                
            OutWeights.append(DataStructWeights)
            OutBiases.append(DataStructBiases)
    
        return OutWeights,OutBiases          

    def evaluate(self,Tensor,Layers,Data):
        """Evaluate model for given input data
        Returns:
            The output of the given tensor"""
        if len(Layers)==1:
            return self.Session.run(Tensor, feed_dict={Layers[0]:Data})
        else:
            return self.Session.run(Tensor, feed_dict={i: _np.array(d) for i, d in zip(Layers,Data)})    
        
    def calc_dE(self,Layers,Data):
        """Returns:
            A tensor for the energy error of the dataset."""
        return _np.nan_to_num(_np.mean(self.evaluate(self.dE_Fun,Layers,Data)))     
            
    def _train_step(self,Layers,Data):
        """Does ones training step(one batch).
        Returns:
            Cost(float):The training cost for the step."""
        #Train the network for one step
        _,Cost=self.Session.run([self.Optimizer,self.CostFun],feed_dict={i: _np.array(d) for i, d in zip(Layers,Data)})
        return Cost


    def _validate_step(self,Layers,Data):
        """Calculates the validation cost for this training step,
        without optimizing the net.
        
        Returns:
            Cost (float): The cost for the data"""
        #Evaluate cost function without changing the network
        Cost=self.Session.run(self.CostFun,feed_dict={i: _np.array(d) for i, d in zip(Layers,Data)})
        return Cost

    def _train_atomic_network_batch(self,Layers,TrainingData,ValidationData):
        """Trains one batch for an atomic network.
        
        Returns:
            TrainingCost (float) The cost for the training step
            ValidationCost(float) The cost for the validation data"""
            
        TrainingCost=0
        ValidationCost=0
        #train batch
        #TrainingCost=sum(_train_step(Session,Optimizer,Layers,TrainingData,CostFun))[0]
        TrainingCost=self._train_step(Layers,TrainingData)
    
        #check validation dataset error
        if ValidationData!=None:
            #ValidationCost=sum(_validate_step(Session,Layers,ValidationData,CostFun))[0]
            ValidationCost=self._validate_step(Layers,ValidationData)
    
        return TrainingCost,ValidationCost     
    

    def _train_atomic_networks(self,TrainingInputs,TrainingOutputs,ValidationInputs=None,ValidationOutputs=None):
        """Trains one step for an atomic network.
        First it prepares the input data and the placeholder and then executes a
        training step. 
        Returns:
            the current session,the trained network and the cost for the 
        training step """
    
    
        TrainCost=[]
        ValidationCost=[]
        ValidationCost=0
        TrainCost=0
        #Prepare data environment for training
        Layers,TrainingData=self.Net.prepare_data_environment(TrainingInputs,self.OutputLayer,TrainingOutputs)
        #Make validation input vector
        if len(ValidationInputs)>0:
            ValidationData=self.Net.make_data_for_atomicNNs(ValidationInputs,ValidationOutputs)
        else:
            ValidationData=None

        print("Started training...")
        #Start training of the atomic network
        for i in range(self.Epochs):
            Cost=self._train_step(Layers,TrainingData)
            TrainCost.append(sum(Cost)/len(Cost))
            #check validation dataset error
            if ValidationData!=None:
                temp_val=self._validate_step(Layers,ValidationData)
                ValidationCost.append(sum(temp_val)/len(temp_val))
    
            if i % max(int(self.Epochs/100),1)==0:
                if self.MakePlot:
                    if i==0:
                        figure,ax,TrainPlot,ValPlot,RunningMeanPlot=_initialize_cost_plot(TrainCost,ValidationCost)
                    else:
                        _update_cost_plot(figure,ax,TrainPlot,TrainCost,ValPlot,ValidationCost,RunningMeanPlot)
                print(str(100*i/self.Epochs)+" %")
    
            if TrainCost[-1]<self.CostCriterium:
                print(TrainCost[-1])
                break
            TrainedNetwork=_tf.trainable_variables()
    
        return TrainedNetwork,TrainCost,ValidationCost
  
           
    def make_network(self):
        """Creates the specified network""" 
        
        Execute=True
        if len(self.Structures)==0:
            print("No structures for the specific nets specified!")
            Execute=False
        if self.IsPartitioned==False:
            if len(self.Structures[0])-1<len(self.Dropout):
                print("Dropout can only be between layers so it must be shorter than the structure,\n but is "+str(len(self.Structures[0]))+" and "+str(len(self.Dropout)))
                Execute=False
        if len(self.NumberOfAtomsPerType)==0:
            print("No number of specific nets specified!")
            Execute=False

        if Execute==True:
            if self.Net== None:
                if self.IsPartitioned==False:
                    self.Net=_StandardAtomicNetwork()
                else:
                    self.Net=_PartitionedAtomicNetwork()
                
        self.Net.make_atomic_networks(self)

    
    def make_and_initialize_network(self):
        """Creates and initializes the specified network"""
        
        self.make_network()
        self.initialize_network()


    def start_training(self):
        """Start the training without the use of batch training.
        Returns:
            self.TrainingCost(list):The costs for the training data.
            self.ValidationCost(list):The costs for the validation data."""
        Execute=True
        if len(self.Net.AtomicNNs)==0:
            print("No atomic neural nets available!")
            Execute=False
        if len(self.TrainingInputs)==0:
            print("No training inputs specified!")
            Execute=False
        if len(self.TrainingOutputs)==0:
            print("No training outputs specified!")
            Execute=False

        if Execute==True:
            self.TrainedNetwork,TrainingCosts,ValidationCosts=self._train_atomic_networks(self.TrainingInputs,self.TrainingOutputs,self.ValidationInputs,self.ValidationOutputs)
            self.TrainedVariables=self.Net.get_trained_variables(self.Session)
            #Store variables

            if not _os.path.exists(self.SavingDirectory):
                _os.makedirs(self.SavingDirectory)
            _np.save(self.SavingDirectory+"/trained_variables",self.TrainedVariables)

            self.TrainingCosts=TrainingCosts
            self.ValidationCosts=ValidationCosts
            print("Training finished")

        return self.TrainingCosts,self.ValidationCosts
    

    def expand_trained_net(self, nAtoms,ModelName=None):
        """Expands a stored net for the specified atoms.
        Args:
            nAtoms (list): Is a list of integers:[1,2]
            ModelName (str): Is the path to the stored file"""
            
        self.NumberOfAtomsPerType=nAtoms
        self.expand_existing_net(ModelName)
        

    def _dE_stat(self,Layers):
        """Calculates the energy error for the whole dataset.
        
        Args:
            Layers(list):List of sorted placeholders
            
        Returns:
            train_stat (list) List consisting of the mean value and the
            variance of the dataset for training data.
            vals_stat (list) List consisting of the mean value and the
            variance of the dataset for validation data."""
    
        train_stat=[]
        train_dE=0
        train_var=0
        val_stat=[]
        val_dE=0
        val_var=0
        
        for i in range(0,len(self.TrainingBatches)):
            TrainingInputs=self.TrainingBatches[i][0]
            TrainingOutputs=self.TrainingBatches[i][1]

            TrainingData=self.Net.make_data_for_atomicNNs(TrainingInputs,TrainingOutputs)

            if i==0:
                train_dE=self.evaluate(self.dE_Fun,Layers,TrainingData)
            else:
                temp=self.evaluate(self.dE_Fun,Layers,TrainingData)
                train_dE=_tf.concat([train_dE,temp],0)
    
        for i in range(0,len(self.ValidationBatches)):
            ValidationInputs=self.ValidationBatches[i][0]
            ValidationOutputs=self.ValidationBatches[i][1]
            
            ValidationData=self.Net.make_data_for_atomicNNs(ValidationInputs,ValidationOutputs)
           
            if i==0:
                val_dE=self.evaluate(self.dE_Fun,Layers,ValidationData)
            else:
                temp=self.evaluate(self.dE_Fun,Layers,ValidationData)
                val_dE=_tf.concat([val_dE,temp],0)
        
        with self.Session.as_default():
            train_dE=train_dE.eval().tolist()
            val_dE=val_dE.eval().tolist()
        
        train_mean=_np.mean(train_dE)
        train_var=_np.var(train_dE)
        val_mean=_np.mean(val_dE)
        val_var=_np.var(val_dE)
        
        train_stat=[train_mean,train_var]
        val_stat=[val_mean,val_var]
        
        return train_stat,val_stat
    
    
    def eval_dataset_energy(self,Batches,BatchNr=0):
        """Prepares and evaluates the dataset for the loaded network.
        
        Args:
            Batches (list):List of a list of numpy arrays.
            BatchNr (int): Index of the batch to evaluate.
            
        Returns:
            List of the predicted energies for the dataset
        """
        
        AllData=Batches[BatchNr]
        GData=AllData[0]
        
        Layers,Data=self.Net.prepare_data_environment(GData)
   
        return self.evaluate(self.TotalEnergy,Layers,Data)
    
    def eval_dataset_force(self,Batches,BatchNr=0):
        """Prepares and evaluates the dataset for the loaded network.
                
        Args:
            Batches (list):List of a list of numpy arrays.
            BatchNr (int): Index of the batch to evaluate.
            
        Returns:
            List of the predicted forces for the dataset"""
            
        AllData=Batches[BatchNr]
        GData=AllData[0]
        DerGData=AllData[2]
        if self.IsPartitioned==False:
            Layers,Data=self.Net.prepare_data_environment(GData,None,[],None,DerGData,AppendForce=True)
        else:
           raise(NotImplementedError) 
        
        return self.evaluate(self.OutputForce,Layers,Data)
    
    def start_evaluation(self,nAtoms,ModelName="save/trained_variables"):
        """Recreates a saved network,prepares and evaluates the specified 
        dataset.
        
        Args:
            nAtoms (list): List of number of atoms per type
            ModelName (str):Path of model.
        Returns:
            Out (list):The predicted energies for the dataset."""
            
        Out=0
        self.expand_trained_net(nAtoms,ModelName)
        for i in range(len(self.EvalData)):
            Out=self.eval_dataset_energy(self.EvalData,i)
                
        return Out
    
    
    def _eval_step(self):
        """Evaluates the prepared data.
        Returns:
            Out (list) List of network outputs (energies)."""
        Out=self.Net.evaluate_all_atomicnns(self.TrainingInputs)

        return Out


    def start_batch_training(self,find_best_symmfuns=False):
        """Starts a batch training
        At each epoch a random batch is selected and trained.
        Every 1% the trained variables (weights and biases) are saved to
        the specified folder.
        At the end of the training an error for the whole dataset is calulated.
        If multiple instances are trained the flag Multiple has to be set to 
        true.
        Args:
            find_best_symmfuns(bool):Specifies whether a weight analysis
            in the first hidden layer should be performed or not.
        
        Returns:
            [self.TrainedVariables,self.MinOfOut] (list):
            The trained network in as a list
            (self.MinOfOut is the offset of the last bias node, necessary
             for tanh or sigmoid activation functions in the last layer)."""
                
        #Clear cost array for multi instance training
        self.OverallTrainingCosts=list()
        self.OverallValidationCosts=list()

        start=_time.time()
        Execute=True
        if len(self.Net.AtomicNNs)==0:
            print("No atomic neural nets available!")
            Execute=False
        if len(self.TrainingBatches)==0:
            print("No training batches specified!")
            Execute=False

        if sum(self.NumberOfAtomsPerType)!= len(self.TrainingBatches[0][0]):
            print([self.NumberOfAtomsPerType,len(self.TrainingBatches[0][0])])
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
                    rnd=_rand.randint(0,NrOfTrainingBatches-1)
                    self.TrainingInputs=self.TrainingBatches[rnd][0]
                    self.TrainingOutputs=self.TrainingBatches[rnd][1]
                    
                    BatchSize=self.TrainingInputs[0].shape[0]
                    if self.ValidationBatches:
                        rnd=_rand.randint(0,NrOfValidationBatches-1)
                        self.ValidationInputs=self.ValidationBatches[rnd][0]
                        self.ValidationOutputs=self.ValidationBatches[rnd][1]
                    if self.UseForce:
                        self.ForceTrainingInput=self.TrainingBatches[rnd][2]
                        self.ForceTrainingOutput=self.TrainingBatches[rnd][3]
                        if self.ValidationBatches:
                            self.ForceValidationInput=self.ValidationBatches[rnd][2]
                            self.ForceValidationOutput=self.ValidationBatches[rnd][3]
                            
                    #Prepare data and layers for feeding
                    if i==0:
                        EnergyLayers=self.Net.make_layers_for_atomicNNs(self.OutputLayer,[],False)
                        Layers,TrainingData=self.Net.prepare_data_environment(self.TrainingInputs,self.OutputLayer,self.TrainingOutputs,self.OutputLayerForce,self.ForceTrainingInput,self.ForceTrainingOutput)

                    else:
                        TrainingData=self.Net.make_data_for_atomicNNs(self.TrainingInputs,self.TrainingOutputs,self.ForceTrainingInput,self.ForceTrainingOutput)
                    #Make validation input vector
                    if len(self.ValidationInputs)>0:
                        ValidationData=self.Net.make_data_for_atomicNNs(self.ValidationInputs,self.ValidationOutputs,self.ForceValidationInput,self.ForceValidationOutput)
                    else:
                        ValidationData=None
                        
                    #Train one batch
                    TrainingCosts,ValidationCosts=self._train_atomic_network_batch(Layers,TrainingData,ValidationData)
                                        
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
                    self.DeltaE=(self.calc_dE(Layers,TrainingData)+self.calc_dE(Layers,TrainingData))/2
                else:
                    self.DeltaE=self.calc_dE(Layers,TrainingData)
                    
                if self.Multiple==False:
                    if i % max(int(self.Epochs/20),1)==0 or i==(self.Epochs-1):
                        #Cost plot
                        if self.MakePlots==True:
                            if i ==0:
                                if find_best_symmfuns:
                                    sparse_tensor=_np.abs(self.Session.run(self.FirstWeights[0]))#only supports force field at the moment
                                    sparse_weights=_np.sum(sparse_tensor,axis=1)
                                    fig_weights,weights_plot=_initialize_weights_plot(sparse_weights,self.SizeOfInputsPerType[0])
                                else:
                                    fig,ax,TrainingCostPlot,ValidationCostPlot,RunningMeanPlot=_initialize_cost_plot(self.OverallTrainingCosts,self.OverallValidationCosts)
                            else:
                                if find_best_symmfuns:
                                    sparse_tensor=_np.abs(self.Session.run(self.FirstWeights[0]))#only supports force field at the moment
                                    sparse_weights=_np.sum(sparse_tensor,axis=1)
                                    fig_weights,weights_plot=_update_weights_plot(fig_weights,weights_plot,sparse_weights)
                                else:
                                    _update_cost_plot(fig,ax,TrainingCostPlot,self.OverallTrainingCosts,ValidationCostPlot,self.OverallValidationCosts,RunningMeanPlot)
                        #Finished percentage output
                        print([str(100*i/self.Epochs)+" %","deltaE = "+str(self.DeltaE)+" ev","Cost = "+str(self.TrainingCosts),"t = "+str(_time.time()-start)+" s","global step: "+str(self.Session.run(self.GlobalStep))])
                        Prediction=self.eval_dataset_energy([[self.TrainingInputs]])
                        print("Data:")
                        print("Ei="+str(self.TrainingOutputs[0:max(int(len(self.TrainingOutputs)/20),1)]))
                        if self.UseForce:
                            Force=self.eval_dataset_force(self.TrainingInputs,self.ForceTrainingInput)
                            print("F1_x="+str(self.ForceTrainingOutput[0:max(int(len(self.TrainingOutputs)/20),1),0]))
                            #print("Net gradient = "+str(self.Session.run(self.dEi_dGij,feed_dict={i: _np.array(d) for i, d in zip(EnergyLayers,self.TrainingInputs)})))
                        print("Prediction:")
                        print("Ei="+str(Prediction[0:max(int(len(Prediction)/20),1)]))
                        if self.UseForce:
                            print("F1_x="+str(Force[0:max(int(len(Prediction)/20),1),0]))
                        #Store variables
                        self.TrainedVariables=self.Net.get_trained_variables(self.Session)



                        if not _os.path.exists(self.SavingDirectory):
                            _os.makedirs(self.SavingDirectory)
                        _np.save(self.SavingDirectory+"/trained_variables",[self.TrainedVariables,self.MeansOfDs,self.VarianceOfDs,self.MinOfOut])
                    

                    #Abort criteria
                    if self.TrainingCosts<=self.CostCriterium and self.ValidationCosts<=self.CostCriterium or self.DeltaE<self.dE_Criterium:
                        
                        if self.ValidationCosts!=0:
                            print("Reached criterium!")
                            print("Cost= "+str((self.TrainingCosts+self.ValidationCosts)/2))
                            print("delta E = "+str(self.DeltaE)+" ev")
                            print("t = "+str(_time.time()-start)+" s")
                            print("Epoch = "+str(i))
                            print("")

                        else:
                            print("Reached criterium!")
                            print("Cost= "+str(self.TrainingCosts))
                            print("delta E = "+str(self.DeltaE)+" ev")
                            print("t = "+str(_time.time()-start)+" s")
                            print("Epoch = "+str(i))
                            print("")
                        

                        print("Calculation of whole dataset energy difference ...")
                        train_stat,val_stat=self._dE_stat(EnergyLayers)
                        print("Training dataset error= "+str(train_stat[0])+"+-"+str(_np.sqrt(train_stat[1]))+" ev")
                        print("Validation dataset error= "+str(val_stat[0])+"+-"+str(_np.sqrt(val_stat[1]))+" ev")
                        print("Training finished")
                        break   
                            
                    if i==(self.Epochs-1):
                        print("Training finished")
                        print("delta E = "+str(self.DeltaE)+" ev")
                        print("t = "+str(_time.time()-start)+" s")
                        print("")
                        
                        train_stat,val_stat=self._dE_stat(EnergyLayers)
                        print("Training dataset error= "+str(train_stat[0])+"+-"+str(_np.sqrt(train_stat[1]))+" ev")
                        print("Validation dataset error= "+str(val_stat[0])+"+-"+str(_np.sqrt(val_stat[1]))+" ev")
                        
                        
            if self.Multiple==True:
                
                return [self.TrainedVariables,self.MinOfOut]
                    
    def _convert_dataset(self,TakeAsReference):
        """Converts the cartesian coordinates to a symmetry function vector and  
        calculates the mean value and the variance of the symmetry function 
        vector.
        
        Args:
            TakeAsReference(bool): Specifies if the MinOfOut Parameter should be 
                                set according to this dataset.
        """

        print("Converting data to neural net input format...")
        NrGeom=len(self.Ds.geometries)
        AllTemp=list()
        #Get G vectors

        for i in range(0,NrGeom):
     
            temp=_np.asarray(self.SymmFunSet.eval_geometry(self.Ds.geometries[i]))

            self.AllGeometries.append(temp)
            self.AllGDerivatives.append(_np.asarray(self.SymmFunSet.eval_geometry_derivatives(self.Ds.geometries[i])))
            if i % max(int(NrGeom/25),1)==0:
                print(str(100*i/NrGeom)+" %")
            for j in range(0,len(temp)):
                if i==0:
                    AllTemp.append(_np.empty((NrGeom,temp[j].shape[0])))
                    AllTemp[j][i]=temp[j]
                else:
                    AllTemp[j][i]=temp[j]
            
        if TakeAsReference:
            self._calculate_statistics_for_dataset(AllTemp)
                    
        
                    
    def _calculate_statistics_for_dataset(self,AllTemp):
        """To be documented..."""
        
        NrAtoms=sum(self.NumberOfAtomsPerType)
        #calculate mean and sigmas for all Gs
        print("Calculating mean values and variances...")
        #Input statistics
        self.MeansOfDs=[0]*len(self.NumberOfAtomsPerType)
        self.VarianceOfDs=[0]*len(self.NumberOfAtomsPerType)
        ct=0
        InputsForTypeX=[]
        for i,InputsForNetX in enumerate(AllTemp):
            if len(InputsForTypeX)==0:
                InputsForTypeX=list(InputsForNetX)
            else:
                InputsForTypeX+=list(InputsForNetX)
                
            if self.NumberOfAtomsPerType[ct]==i+1: 
                self.MeansOfDs[ct]=_np.mean(InputsForTypeX,axis=0)
                self.VarianceOfDs[ct]=_np.var(InputsForTypeX,axis=0)
                InputsForTypeX=[]
                ct+=1
        #Output statistics
        if len(self.Ds.energies)>0:
            NormalizedEnergy=_np.divide(self.Ds.energies,NrAtoms)
            self.MinOfOut=_np.min(NormalizedEnergy)*2 #factor of two is to make sure that there is room for lower energies
    

    def read_lammps_files(self,TakeAsReference=True,LoadGeometries=True):
        """Reads lammps files,adds symmetry functions to the symmetry function
        basis and converts the cartesian corrdinates to symmetry function vectors.
        
        Args:
            TakeAsReference(bool): Specifies if the MinOfOut Parameter should be 
                                set according to this dataset.
            LoadGeometries(bool): Specifies if the conversion of the geometry 
                                coordinates should be performed."""
                                
        Execute=True
        if self.XYZfile==None:
            print("No .xyz-file name specified!")
            Execute=False
        if self.Logfile==None:
            print("No log-file name specified!")
            Execute=False
        if len(self.atomtypes)==0:
            print("No atom types specified!")
            Execute=False
#        if len(self.Zetas)==0:
#            print("No zetas specified!")
#            Execute=False
#        if len(self.Lambs)==0:
#            print("No lambdas specified!")
#            Execute=False

        if Execute==True:

            self.Ds=_DataSet.DataSet()
            self.SymmFunSet=_SymmetryFunctionSet.SymmetryFunctionSet(self.atomtypes)
            self.Ds.read_lammps(self.XYZfile,self.Logfile)
            print("Added dataset!")
            #self.SymmFunSet.add_radial_functions(self.Rs,self.Etas)
            self.SymmFunSet.add_radial_functions_evenly(self.NumberOfRadialFunctions)
            self.SymmFunSet.add_angular_functions(self.Etas,self.Zetas,self.Lambs)
            self.SizeOfInputsPerType=self.SymmFunSet.num_Gs
            for i,a_type in enumerate(self.NumberOfAtomsPerType):
                for j in range(0,a_type):
                    self.SizeOfInputsPerAtom.append(self.SizeOfInputsPerType[i])
        if LoadGeometries:
            self._convert_dataset(TakeAsReference)
    

    def init_dataset(self,geometries,energies,g_derivatives=[],TakeAsReference=True):
        """Initializes a loaded dataset.
        
        Args:
            geometries (list): List of geomtries
            energies (list) : List of energies
            g_derivaties (list): List of G-vector derivatives
            TakeAsReference (bool): Specifies if the MinOfOut Parameter 
                                    should be set according to this dataset"""
    
        if len(geometries)==len(energies):
            self.Ds=_DataSet.DataSet()
            self.VarianceOfDs=[]
            self.MeansOfDs=[]
            self.SizeOfInputsPerType=[]
            self.SymmFunSet=_SymmetryFunctionSet.SymmetryFunctionSet(self.atomtypes)
            self.Ds.energies=energies
            self.Ds.geometries=geometries
            self.Ds.g_derivaties=g_derivatives
            self.SymmFunSet.add_radial_functions_evenly(self.NumberOfRadialFunctions)
            self.SymmFunSet.add_angular_functions(self.Etas,self.Zetas,self.Lambs)
            self.SizeOfInputsPerType=self.SymmFunSet.num_Gs
            for i,a_type in enumerate(self.NumberOfAtomsPerType):
                for j in range(0,a_type):
                    self.SizeOfInputsPerAtom.append(self.SizeOfInputsPerType[i])
                
            self._convert_dataset(TakeAsReference)
        else:
            print("Number of energies: "+str(len(energies))+" does not match number of geometries: "+str(len(geometries)))
        

    def create_eval_data(self,geometries,NoBatches=True):
        """Converts the geometries in compatible format and prepares the data
        for evaluation.
        
        Args:
            geometries (list): List of geometries
            NoBatches (bool): Specifies if the data is split into differnt 
                batches or only consits of a single not randomized batch.
        """
        dummy_energies=[0]*len(geometries)
        if len(self.MeansOfDs)==0:
            IsReference=True
        else:
            IsReference=False
            
        self.init_dataset(geometries,dummy_energies,TakeAsReference=IsReference)
        
        self.EvalData=self.get_data(NoBatches=True)
             
        return self.EvalData
        

    def _get_data_batch(self,BatchSize=100,NoBatches=False):
        """Creates a data batch by drawing a random sample out of the dataset
        The symmetry function vector is then normlized.
        
        Args:
            BatchSize (int):Number of elements per training cycle
            NoBatches (bool): Specifies if the data is split into differnt 
                batches or only consits of a single not randomized batch.
                
        Returns:
            Input (list): list of G vectors
            EnergyData (list): list of energies
            GDerivativesInput (optional list): list of G-vector derivatives
            ForceData (optional list): list of forces per atom"""

        GeomData=[]
        GDerivativesInput=[]
        ForceData=[]
        
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
            if NoBatches:
                BatchSize=len(self.Ds.geometries)

            EnergyData=_np.empty((BatchSize,1))
            ForceData=_np.empty((BatchSize,sum(self.NumberOfAtomsPerType)*3))
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
                    rnd=_rand.randint(0,len(ValuesForDrawingSamples)-1)
                    #Get number
                    MyNr=ValuesForDrawingSamples[rnd]
                    #remove number from possible samples
                    ValuesForDrawingSamples.pop(rnd)

                GeomData.append(self.AllGeometries[MyNr])
                EnergyData[i]=self.Ds.energies[MyNr]
                if len(self.Ds.g_derivaties)>0:
                    ForceData[i]=[f for atom in self.Ds.g_derivaties[MyNr] for f in atom]
                if self.UseForce:
                    GDerivativesInput.append(self.AllGDerivatives[MyNr])
            
            Inputs,GDerivativesInput=self._sort_and_normalize_data(BatchSize,GeomData,GDerivativesInput)
            
            if self.UseForce:
                return Inputs,EnergyData,GDerivativesInput,ForceData
            else:
                return Inputs,EnergyData
                
    
    def get_data(self,BatchSize=100,CoverageOfSetInPercent=70,NoBatches=False):
        """Creates a batch collection.
        
        Args:
            CoverageOfSetInPercent(int):discribes how many data points are 
                included in the batch (value from 0-100).
            NoBatches (bool): Specifies if the data is split into differnt 
                batches or only consits of a single not randomized batch.
                
        Returns:
            Batches: List of numpy arrays"""
            
        Batches=[]
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
           
            EnergyDataSetLength=len(self.Ds.geometries)
            SetLength=int(EnergyDataSetLength*CoverageOfSetInPercent/100)

            if NoBatches==False:
                if BatchSize>len(self.AllGeometries)/10:
                    BatchSize=int(BatchSize/10)
                    print("Shrunk batches to size:"+str(BatchSize))
                NrOfBatches=max(1,int(round(SetLength/BatchSize,0)))
            else:
                NrOfBatches=1
            print("Creating and normalizing "+str(NrOfBatches)+" batches...")
            for i in range(0,NrOfBatches):
                Batches.append(self._get_data_batch(BatchSize,NoBatches))
                if NoBatches==False:
                    if i % max(int(NrOfBatches/10),1)==0:
                        print(str(100*i/NrOfBatches)+" %")

            return Batches
        

    def make_training_and_validation_data(self,BatchSize=100,TrainingSetInPercent=70,ValidationSetInPercent=30,NoBatches=False):
        """Creates training and validation data.
        
        Args:
            
            BatchSize (int): Specifies the number of data points per batch.
            TrainingSetInPercent (int): Discribes the coverage of the training dataset.(value from 0-100)
            TrainingSetInPercent (int): Discribes the coverage of the validation dataset.(value from 0-100)
            NoBatches (bool): Specifies if the data is split into differnt batches or only """
            
        if NoBatches==False:
            #Get training data
            self.TrainingBatches=self.get_data(BatchSize,TrainingSetInPercent,NoBatches)
            #Get validation data
            self.ValidationBatches=self.get_data(BatchSize,ValidationSetInPercent,NoBatches)
        else:
            #Get training data
            temp=self.get_data(BatchSize,TrainingSetInPercent,NoBatches)
            self.TrainingInputs=temp[0][0]
            self.TrainingOutputs=temp[0][1]
            #Get validation data
            temp=self.get_data(BatchSize,ValidationSetInPercent,NoBatches)
            self.ValidationInputs=temp[0][0]
            self.ValidationOutputs=temp[0][0]
            
    

    def _atomic_cost_function(self):
        """The atomic cost function consists of multiple parts which are each
        represented by a tensor.
        The main part is the energy cost.
        The reqularization and the force cost is optional.
        
        Returns:
            A tensor which is the sum of all costs"""
        
        self.TotalEnergy,AllEnergies=self.Net.energy_of_all_atomic_networks()
  
        Cost=cost_for_network(self.TotalEnergy,self.OutputLayer,self.CostFunType)
        #add force cost
        if self.UseForce==True:
            if self.IsPartitioned==True:
                raise(NotImplementedError)
            else:
                self.OutputForce,AllForces=self.force_of_all_atomic_networks()  
                #Cost+=cost_for_network(self.OutputForce,self.OutputLayerForce,self.CostFunType)
                
        #Create tensor for energy difference calculation
        self.dE_Fun=_tf.abs(self.TotalEnergy-self.OutputLayer)
        
        if self.Regularization=="L1":
            trainableVars=_tf.trainable_variables()
            l1_regularizer = _tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
            Cost += _tf.contrib.layers.apply_regularization(l1_regularizer, trainableVars)
        elif self.Regularization=="L2":
            trainableVars=_tf.trainable_variables()
            self.RegLoss=_tf.add_n([_tf.nn.l2_loss(v) for v in trainableVars
                               if 'bias' not in v.name]) * self.RegularizationParam
            Cost += self.RegLoss
    
        return Cost


    def _sort_and_normalize_data(self,BatchSize,GData,GDerivativesData=[]):
        """Normalizes the input data.
        
        Args:
            BatchSize (int): Specifies the number of data points per batch.
            GeomData (list): Raw geometry data
            ForceData (list): (Optional) Raw derivatives of input vector
            
        Returns:
            Inputs: Normalized inputs
            DerInputs: If GDerivativesData is available a list of numpy array is returned
                    ,else an empty list is returned."""

        Inputs=[]
        DerInputs=[]
        ct=0
        for VarianceOfDs,MeanOfDs,NrAtoms in zip(self.VarianceOfDs,self.MeansOfDs,self.NumberOfAtomsPerType):
            for i in range(NrAtoms):
                Inputs.append(_np.zeros((BatchSize, self.SizeOfInputsPerAtom[ct])))
                if len(GDerivativesData)>0:
                    DerInputs.append(_np.zeros((BatchSize, self.SizeOfInputsPerAtom[ct],3*sum(self.NumberOfAtomsPerType))))
                #exclude nan values
                L=_np.nonzero(VarianceOfDs)
    
                if L[0].size>0:
                    for j in range(0,len(GData)):
                        Inputs[ct][j][L]=_np.divide(_np.subtract(GData[j][ct][L],MeanOfDs[L]),_np.sqrt(VarianceOfDs[L]))
                        if len(GDerivativesData)>0:
                            DerInputs[ct][j]=GDerivativesData[j][ct]
                    
                ct+=1
                

        return Inputs,DerInputs
    

    def _init_correction_network_data(self,network,geoms,energies,N_zero_geoms=10000):
        """Creates data outside the area covered by the dataset and adds them 
        to the training data.
        
        Args:
            network (Tensor): Pre trained network
            geoms (list): List of geometries
            energies (list): List of energies
            N_zero_geom (int): Number of geometries to create
        """
        #get coverage of dataset
        ds_r_min,ds_r_max=_get_ds_r_min_r_max(geoms)
        #create geometries outside of dataset

        zero_ds_geoms=_create_zero_diff_geometries(ds_r_min,ds_r_max,self.atomtypes,self.NumberOfAtomsPerType,N_zero_geoms)
        all_geoms=geoms+zero_ds_geoms
        #eval geometries with FF network to make energy difference zero
        network.create_eval_data(geoms)
        zero_ds_energies=[energy for element in network.eval_dataset_energy(network.EvalData) for energy in element]
        
        all_energies=energies+zero_ds_energies
        self.init_dataset(all_geoms,all_energies)




class _StandardAtomicNetwork(object):   
    
    
    def __init__(self):
        self.AtomicNNs=[]
        self.VariablesDictionary={}
        
        

    def get_structure_from_data(self,TrainedData):
        """Calculates the structure based on the stored varibales
        
        Args:
            TrainedData (list): Trained network data
        Returns:
            Structues (list):A list of structures
            (one structure per atom species)"""
        
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
     
    def force_of_all_atomic_networks(self):
        """This function constructs the force expression for the atomic networks.
        
        Returns:
            A tensor which represents the force output of the network""" 
        
        F=[]
        Fi=[]
        for i in range(0,len(self.AtomicNNs)):
            AtomicNet=self.AtomicNNs[i]
            Type=AtomicNet[0]
            G_Input=AtomicNet[2]
            dGij_dxk=AtomicNet[3]
            dGij_dxk_t=_tf.transpose(dGij_dxk,perm=[0,2,1])
            temp=_tf.gradients(self.TotalEnergy,G_Input)
            dEi_dGij=_tf.reshape(temp,[-1,self.SizeOfInputsPerType[Type],1])
            mul=_tf.matmul(dGij_dxk_t,dEi_dGij)
            dim_red=_tf.reshape(mul,[-1,sum(self.NumberOfAtomsPerType)*3])
            if i==0:
                F=dim_red
            else:
                F=_tf.add(F,dim_red)
            Fi.append(dim_red)
        
        return F,Fi
        
    def energy_of_all_atomic_networks(self):    
        """This function constructs the energy expression for 
        the atomic networks.
        
        Returns:
            Prediction: A tensor which represents the energy output of 
                        the partitioned network.
            AllEnergies: A list of tensors which represent the single Network
                        energy contributions."""

        Prediction=0
        AllEnergies=list()
    
        for i in range(0,len(self.AtomicNNs)):
            #Get network data
            AtomicNetwork=self.AtomicNNs[i]
            Network=AtomicNetwork[1]
            #Get input data for network
            AllEnergies.append(Network)
        
        Prediction=_tf.add_n(AllEnergies)
    
        return Prediction,AllEnergies
        
    def get_trained_variables(self,Session):
        """Prepares the data for saving.
        It gets the weights and biases from the session.
        
        Returns:
            NetworkData (list): All the network parameters as a list"""
            
        NetworkData=list()
        for HiddenLayers in self.VariablesDictionary:
            Layers=list()
            for i in range(0,len(HiddenLayers)):
                Weights=Session.run(HiddenLayers[i][0])
                Biases=Session.run(HiddenLayers[i][1])
                Layers.append([Weights,Biases])
            NetworkData.append(Layers)
        return NetworkData 
    
    def make_layers_for_atomicNNs(self,OutputLayer=None,OutputLayerForce=None,AppendForce=True):
        """Sorts the input placeholders in the correct order for feeding.
        Each atom has a seperate placeholder which must be feed at each step.
        The placeholders have to match the symmetry function input.
        For training the output placeholder also has to be feed.
        
        Returns:
            Layers (list):All placeholders as a list."""
        #Create list of placeholders for feeding in correct order
        Layers=list()
        ForceLayer=False
        for AtomicNetwork in self.AtomicNNs:
            Layers.append(AtomicNetwork[2])
            if len(AtomicNetwork)>3 and AppendForce:
                Layers.append(AtomicNetwork[3])
                ForceLayer=True
        if OutputLayer!=None:
            Layers.append(OutputLayer)
            if ForceLayer and AppendForce:
                Layers.append(OutputLayerForce)
    
        return Layers
    

    def make_data_for_atomicNNs(self,GData,OutData=[],GDerivatives=[],ForceOutput=[],AppendForce=True):
        """Sorts the symmetry function data for feeding.
            For training the output data also has to be added.
        Returns:
            CombinedData(list): Sorted data for the batch as a list."""
        #Sort data matching the placeholders
        CombinedData=list()
        if len(GDerivatives)!=0:
            for e,f in zip(GData,GDerivatives):
                CombinedData.append(e)
                CombinedData.append(f)
        else:
            for Data in GData:
                CombinedData.append(Data)
        if len(OutData)!=0:
            CombinedData.append(OutData)
            if len(ForceOutput)!=0:
                CombinedData.append(ForceOutput)
    
        return CombinedData
    
    
    

    def prepare_data_environment(self,GData,OutputLayer=None,OutData=[],OutputLayerForce=None,GDerivatives=[],ForceOutput=[],AppendForce=True):
        """Prepares the data and the input placeholders for the training in a NN.
        Returns:
            Layers (list):List of sorted placeholders
            Data (list): List of sorted data"""
        
        #Put data and placeholders in correct order for feeding
        Layers=self.make_layers_for_atomicNNs(OutputLayer,OutputLayerForce,AppendForce)
        
        Data=self.make_data_for_atomicNNs(GData,OutData,GDerivatives,ForceOutput,AppendForce)
        
        return Layers,Data
    
    def evaluate_all_atomicnns(self,GData,AppendForce=False):
        """Evaluates the networks and calculates the energy as a sum of all network
        outputs.
        
        Returns:
            Energy(float): Predicted energy as a float."""
        Energy=0
        Layers,Data=self.prepare_data_environment(GData,OutputLayer=None,OutData=[],OutputLayerForce=None,GDerivativesInput=[],ForceOutput=[],AppendForce=False)
        for i in range(0,len(self.AtomicNNs)):
            AtomicNetwork=self.AtomicNNs[i]
            Energy+=self.evaluate(AtomicNetwork[1],[Layers[i]],Data[i])
    
        return Energy
    

    def get_weights_biases_from_data(self,TrainedVariables,Multi=False):
        """Reads out the saved network data and sorts them into 
        weights and biases.
        
        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""
            
        Weights=list()
        Biases=list()
        for i in range(0,len(TrainedVariables)):
            NetworkData=TrainedVariables[i]
            ThisWeights=list()
            ThisBiases=list()
            for j in range(0,len(NetworkData)):
                ThisWeights.append(NetworkData[j][0])
                ThisBiases.append(NetworkData[j][1])
            Weights.append(ThisWeights)
            Biases.append(ThisBiases)
    
        return Weights,Biases

    def make_parallel_atomic_networks(self,NetInstance):
        """Creates the specified network with separate varibale tensors
            for each atoms.(Only for evaluation)"""

        AtomicNNs = list()
        # Start Session
        NetInstance.Session=_tf.Session()    
        # make all layers
        if len(NetInstance.Structures)!= len(self.NumberOfAtomsPerType):
            raise ValueError("Length of Structures does not match length of NumberOfAtomsPerType")
        if len(NetInstance.HiddenData) != 0:
            for i in range(0, len(NetInstance.Structures)):
                if len(NetInstance.Dropout)>i:
                    Dropout=NetInstance.Dropout[i]
                else:
                    Dropout=NetInstance.Dropout[-1]
                    
                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):
               
                    # Make hidden layers
                    HiddenLayers = list()
                    Structure = NetInstance.Structures[i]
                    
                    RawWeights = NetInstance.HiddenData[i]
                    RawBias = NetInstance.BiasData[i]
                        
                    for j in range(1, len(Structure)):
                        NrIn = Structure[j - 1]
                        NrHidden = Structure[j]
                        # fill old weights in new structure
                        ThisWeightData = RawWeights[j - 1]
                        ThisBiasData = RawBias[j - 1]
        
                        HiddenLayers.append(
                            _construct_hidden_layer(NrIn, NrHidden, NetInstance.HiddenType, ThisWeightData, NetInstance.BiasType,
                                                   ThisBiasData, NetInstance.MakeAllVariable))
                                   
                    # Make input layer
                    if len(NetInstance.HiddenData) != 0:
                        NrInputs = NetInstance.HiddenData[i][0].shape[0]
                    else:
                        NrInputs = Structure[0]
        
                    InputLayer = _construct_input_layer(NrInputs)
                    # Connect input to first hidden layer
                    FirstWeights = HiddenLayers[0][0]
                    FirstBiases = HiddenLayers[0][1]
                    Network = _connect_layers(InputLayer, FirstWeights, FirstBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
        
                    for l in range(1, len(HiddenLayers)):
                        # Connect ouput of in layer to second hidden layer
        
                        if l == len(HiddenLayers) - 1:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(Network, Weights, Biases, "none", NetInstance.ActFunParam,Dropout)
                        else:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(Network, Weights, Biases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
                    
                    if NetInstance.UseForce:
                        InputForce=_tf.placeholder(_tf.float32, shape=[None, NrInputs,3*sum(NetInstance.NumberOfAtomsPerType)])
                        Normalization=_tf.placeholder(_tf.float32, shape=[None, NrInputs])
                        AtomicNNs.append([i, Network, InputLayer,InputForce,Normalization])
                    else:
                        AtomicNNs.append(
                            [i, Network, InputLayer])
        else:
            print("No network data found!")
            
        self.AtomicNNs=AtomicNNs
    

    def make_atomic_networks(self,NetInstance):
        """Creates the specified network."""
        
        AllHiddenLayers = list()
        AtomicNNs = list()
        # Start Session
        if NetInstance.Multiple==False:
            NetInstance.Session=_tf.Session(config=_tf.ConfigProto(
  intra_op_parallelism_threads=_multiprocessing.cpu_count()))
            
        OldBiasNr = 0
        OldShape = None
        if len(NetInstance.Structures)!= len(NetInstance.NumberOfAtomsPerType):
            raise ValueError("Length of Structures does not match length of NumberOfAtomsPerType")
        if isinstance(NetInstance.Structures[0],_PartitionedStructure):
            raise ValueError("Please set IsPartitioned = True !")
        else:
            # create layers for the different atom types
            for i in range(0, len(NetInstance.Structures)):
                if len(NetInstance.Dropout)>i:
                    Dropout=NetInstance.Dropout[i]
                else:
                    Dropout=NetInstance.Dropout[-1]
                    
                # Make hidden layers
                HiddenLayers = list()
                Structure = NetInstance.Structures[i]
                if len(NetInstance.HiddenData) != 0:
                    
                    RawBias = NetInstance.BiasData[i]
    
                    for j in range(1, len(Structure)):
                        NrIn = Structure[j - 1]
                        NrHidden = Structure[j]
    
                        if j == len(Structure) - 1 and NetInstance.MakeLastLayerConstant == True:
                            HiddenLayers.append(_construct_not_trainable_layer(NrIn, NrHidden, NetInstance.MinOfOut))
                        else:
                            if j >= len(NetInstance.HiddenData[i]) and NetInstance.MakeLastLayerConstant == True:
                                tempWeights, tempBias = _construct_hidden_layer(NrIn, NrHidden, NetInstance.HiddenType, [], NetInstance.BiasType, [],
                                                                               True, NetInstance.InitMean, NetInstance.InitStddev)
                                
                                indices = []
                                values = []
                                thisShape = tempWeights.get_shape().as_list()
                                if thisShape[0]==thisShape[1]:
                                    for q in range(0, OldBiasNr):
                                        indices.append([q, q])
                                        values += [1.0]
                                        
                                    delta = _tf.SparseTensor(indices, values, thisShape)
                                    tempWeights = tempWeights + _tf.sparse_tensor_to_dense(delta)
                                
                                HiddenLayers.append([tempWeights, tempBias])
                            else:
                                if len(RawBias) >= j:
                                    OldBiasNr = len(NetInstance.BiasData[i][j - 1])
                                    OldShape = NetInstance.HiddenData[i][j - 1].shape
                                    # fill old weights in new structure
                                    if OldBiasNr < NrHidden:
                                        ThisWeightData = _np.random.normal(loc=0.0, scale=0.01, size=(NrIn, NrHidden))
                                        ThisWeightData[0:OldShape[0], 0:OldShape[1]] = NetInstance.HiddenData[i][j - 1]
                                        ThisBiasData = _np.zeros([NrHidden])
                                        ThisBiasData[0:OldBiasNr] = NetInstance.BiasData[i][j - 1]
                                    elif OldBiasNr>NrHidden:
                                        ThisWeightData = _np.zeros((NrIn, NrHidden))
                                        ThisWeightData[0:, 0:] = NetInstance.HiddenData[i][j - 1][0:NrIn,0:NrHidden]
                                        ThisBiasData = _np.zeros([NrHidden])
                                        ThisBiasData[0:OldBiasNr] = NetInstance.BiasData[i][j - 1][0:NrIn,0:NrHidden]
                                    else:
                                        ThisWeightData = NetInstance.HiddenData[i][j - 1]
                                        ThisBiasData = NetInstance.BiasData[i][j - 1]
        
                                    HiddenLayers.append(
                                        _construct_hidden_layer(NrIn, NrHidden, NetInstance.HiddenType, ThisWeightData, NetInstance.BiasType,
                                                               ThisBiasData, NetInstance.MakeAllVariable))
                                else:
                                    raise ValueError("Number of layers doesn't match["+str(len(RawBias))+str(len(Structure))+"], MakeLastLayerConstant has to be set to True!")
    
    
                else:
                    for j in range(1, len(Structure)):
                        NrIn = Structure[j - 1]
                        NrHidden = Structure[j]
                        if j == len(Structure) - 1 and NetInstance.MakeLastLayerConstant == True:
                            HiddenLayers.append(_construct_not_trainable_layer(NrIn, NrHidden, NetInstance.MinOfOut))
                        else:
                            HiddenLayers.append(_construct_hidden_layer(NrIn, NrHidden, NetInstance.HiddenType, [], NetInstance.BiasType))
    
                AllHiddenLayers.append(HiddenLayers)
                #create network for each atom
                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):
                    # Make input layer
                    if len(NetInstance.HiddenData) != 0:
                        NrInputs = NetInstance.HiddenData[i][0].shape[0]
                    else:
                        NrInputs = Structure[0]
    
                    InputLayer = _construct_input_layer(NrInputs)
                    # Connect input to first hidden layer
                    FirstWeights = HiddenLayers[0][0]
                    NetInstance.FirstWeights.append(FirstWeights)
                    FirstBiases = HiddenLayers[0][1]
                    Network = _connect_layers(InputLayer, FirstWeights, FirstBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
    
                    for l in range(1, len(HiddenLayers)):
                        # Connect layers
    
                        if l == len(HiddenLayers) - 1:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(Network, Weights, Biases, "none", NetInstance.ActFunParam,Dropout)
                        else:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(Network, Weights, Biases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
    
                    if NetInstance.UseForce:
                        InputForce=_tf.placeholder(_tf.float32, shape=[None, NrInputs,3*sum(NetInstance.NumberOfAtomsPerType)])
                        Normalization=_tf.placeholder(_tf.float32, shape=[None, NrInputs])
                        AtomicNNs.append([i, Network, InputLayer,InputForce,Normalization])
                    else:
                        AtomicNNs.append([i, Network, InputLayer])
    
            
            self.AtomicNNs=AtomicNNs
            self.VariablesDictionary=AllHiddenLayers
            
class _PartitionedAtomicNetwork(object):  
    
    def __init__(self):
        self.AtomicNNs=[]
        self.VariablesDictionary={}    
      
    def get_structure_from_data(self,TrainedData):
        raise(NotImplemented)
        
    def force_of_all_atomic_networks(self):
        raise(NotImplemented)
    
    def energy_of_all_atomic_networks(self):
        """This function constructs the energy expression for 
        the partitioned atomic networks.
        
        Returns:
            Prediction: A tensor which represents the energy output of 
                        the partitioned network.
            AllEnergies: A list of tensors which represent the single Network
                        energy contributions."""
            
        Prediction=0
        AllEnergies=list()
        for i in range(0,len(self.AtomicNNs)):
            #Get network data
            AtomicNetwork=self.AtomicNNs[i]
            Networks=AtomicNetwork[1]
            for j in range(0,2):
                SubNet=Networks[j]
                if SubNet!=j:
                    #Get input data for network
                    AllEnergies.append(SubNet)
    
        Prediction=_tf.add_n(AllEnergies)
    
    
        return Prediction,AllEnergies
         
    def get_trained_variables(self,Session):
        """Prepares the data for saving.
        It gets the weights and biases from the session.
        
        Returns:
            NetworkData (list): All the network parameters as a list"""
            
        NetworkData=list()
        
        for Network in self.VariablesDictionary:
            NetLayers=[]
            for i in range(0,len(Network)):
                SubNetLayers=[]
                if Network[i]!=i: #if SNetwork[i]==i means no net data
                    SubNet=Network[i]
                    for j in range(0,len(SubNet)):
                        Weights=Session.run(SubNet[j][0])
                        Biases=Session.run(SubNet[j][1])
                        SubNetLayers.append([Weights,Biases])
                        
                NetLayers.append(SubNetLayers)
                
            NetworkData.append(NetLayers)
    
        return NetworkData   
 
    def make_layers_for_atomicNNs(self,OutputLayer=[],OutputLayerForce=None,AppendForce=True):
        """Sorts the input placeholders in the correct order for feeding.
        Each atom has a seperate placeholder which must be feed at each step.
        The placeholders have to match the symmetry function input.
        For training the output placeholder also has to be feed.
        
        Returns:
            Layers (list):All placeholders as a list."""
            
        Layers=[]
        for i in range(0,len(self.AtomicNNs)):
            Layer_parts=self.AtomicNNs[i][2]
            #Append layers for each part
            for j in range(0,2):
                if Layer_parts[j]!=j: #Layer_parts is range(2) if empty
                    Layers.append(Layer_parts[j])
        if not(isinstance(OutputLayer,(list, tuple))):
            Layers.append(OutputLayer)
                    
        return Layers

    def make_data_for_atomicNNs(self,GData,OutData=[],GDerivatives=[],ForceOutput=[],AppendForce=True):
        """Sorts the symmetry function data for feeding.
            For training the output data also has to be added.
        Returns:
            CombinedData(list): Sorted data for the batch as a list."""
        CombinedData=[]
        for i in range(0,len(self.AtomicNNs)):
            Data=GData[i]
            #Append layers for each part
            for j in range(0,2):
                if j==0: #force field data
                    CombinedData.append(Data)
                else:
                    CombinedData.append(Data)
                    
        if len(OutData)!=0:
            CombinedData.append(OutData)
            
        return CombinedData
    
    def prepare_data_environment(self,GData,OutputLayer=None,OutData=[],OutputLayerForce=None,GDerivatives=[],ForceOutput=[],AppendForce=True):
        """Prepares the data and the input placeholders for the training in a 
        partitioned NN. 
        Returns:
            Layers (list):Sorted placeholders
            CombinedData (list):Sorted data as lists"""
        Layers=self.make_layers_for_atomicNNs(self,OutputLayer)
        CombinedData=self.make_data_for_atomicNNs(self,GData,OutData)
            
        return Layers,CombinedData
        

    def evaluate_all_atomicnns(self,GData):
        """Evaluates the partitioned networks and calculates the energy as a sum of all 
        network outputs.
        Returns:
            Energy (float):The predicted energy as a float."""
        Energy=0
        Layers,Data=self.prepare_data_environment(GData,list(),list())
        ct=0
        for i in range(0,len(self.AtomicNNs)):
            AllAtomicNetworks=self.AtomicNNs[i][1]
            for j in range(0,2):
                SubNet=AllAtomicNetworks[j]
                if SubNet!=j:
                    Energy+=self.evaluate(SubNet,[Layers[ct]],Data[ct])
                    ct=ct+1
    
        return Energy
    
    def get_weights_biases_from_data(self,TrainedVariables,Multi=False):
        """Reads out the saved network data for partitioned nets and sorts them into 
        weights and biases
        
        Returns:
            Weights (list) List of numpy arrays
            Biases (list) List of numpy arrays"""
            
        Weights=list()
        Biases=list()
        for i in range(0,len(TrainedVariables)):
            NetworkData=TrainedVariables[i]
            ThisWeights=_PartitionedNetworkData()
            ThisBiases=_PartitionedNetworkData()
            for j in range(0,len(NetworkData)):
                SubNetData=NetworkData[j]
                for k in range(0,len(SubNetData)):
                    if j==0:
                        ThisWeights.ForceFieldNetworkData.append(SubNetData[k][0])
                        if Multi==False:
                            ThisWeights.ForceFieldVariable=False
                        else:
                            ThisWeights.ForceFieldVariable=True
                            
                        ThisBiases.ForceFieldNetworkData.append(SubNetData[k][1])
                    elif j==1:
                        ThisWeights.CorrectionNetworkData.append(SubNetData[k][0])
                        if Multi==False:
                            ThisWeights.CorretionVariable=False
                        else:
                            ThisWeights.CorretionVariable=True
                        
                        ThisBiases.CorrectionNetworkData.append(SubNetData[k][1])
            Weights.append(ThisWeights)
            Biases.append(ThisBiases)
    
        return Weights,Biases

    def make_parallel_atomic_networks(self,NetInstance):
        """Creates the specified partitioned network with separate varibale
        tensors for each atoms.(Only for evaluation)"""
    
        AtomicNNs = list()
        # Start Session
        NetInstance.Session=_tf.Session()
        if len(NetInstance.Structures)!= len(NetInstance.NumberOfAtomsPerType):
            raise ValueError("Length of Structures does not match length of NumberOfAtomsPerType")
        if len(NetInstance.HiddenData) != 0:    
            # make all the networks for the different atom types
            for i in range(0, len(NetInstance.Structures)):
                if len(NetInstance.Dropout)>i:
                    Dropout=NetInstance.Dropout[i]
                else:
                    Dropout=NetInstance.Dropout[-1]
                    
                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):
                    
                    ForceFieldHiddenLayers=list()
                    CorrectionHiddenLayers=list()
                    
                    ForceFieldWeights=list()
                    CorrectionWeights=list()
                    
                    ForceFieldBias=list()
                    CorrectionBias=list()
                    
                    ForceFieldNetwork=None
                    CorrectionNetwork=None
                    
                    ForceFieldInputLayer=None
                    CorrectionInputLayer=None
                    
                    # Read structures for specific network parts
                    StructureForAtom=NetInstance.Structures[i]
                    ForceFieldStructure=StructureForAtom.ForceFieldNetworkStructure
                    CorrectionStructure=StructureForAtom.CorrectionNetworkStructure
                    #Construct networks out of loaded data
                    
                    #Load data for atom
                    WeightData=NetInstance.HiddenData[i]
                    BiasData=NetInstance.BiasData[i]
                    if len(WeightData.ForceFieldNetworkData)>0:
                        
                        #Recreate force field network             
                        ForceFieldWeights = WeightData.ForceFieldNetworkData
                        ForceFieldBias = BiasData.ForceFieldNetworkData
        
                        for j in range(1, len(ForceFieldStructure)):
                            
                            ThisWeightData = ForceFieldWeights[j - 1]
                            ThisBiasData = ForceFieldBias[j - 1]
                            ForceFieldNrIn = ThisWeightData.shape[0]
                            ForceFieldNrHidden = ThisWeightData.shape[1]
                            ForceFieldHiddenLayers.append(_construct_hidden_layer(ForceFieldNrIn, ForceFieldNrHidden, NetInstance.HiddenType, ThisWeightData, NetInstance.BiasType,ThisBiasData,WeightData.ForceFieldVariable,NetInstance.InitMean,NetInstance.InitStddev))
    
        
                    if len(WeightData.CorrectionNetworkData)>0:   
                        #Recreate correction network             
                        CorrectionWeights = WeightData.CorrectionNetworkData
                        CorrectionBias = BiasData.CorrectionNetworkData
        
                        for j in range(1, len(CorrectionStructure)):
                            
                            ThisWeightData = CorrectionWeights[j - 1]
                            ThisBiasData = CorrectionBias[j - 1]
                            CorrectionNrIn = ThisWeightData.shape[0]
                            CorrectionNrHidden = ThisWeightData.shape[1]
                            CorrectionHiddenLayers.append(_construct_hidden_layer(CorrectionNrIn, CorrectionNrHidden, NetInstance.HiddenType, ThisWeightData, NetInstance.BiasType,ThisBiasData,WeightData.CorrectionVariable,NetInstance.InitMean,NetInstance.InitStddev))
    
                        
                    if len(ForceFieldHiddenLayers)>0:
                        # Make force field input layer
                        ForceFieldHiddenData=NetInstance.HiddenData[i].ForceFieldNetworkData
                        ForceFieldNrInputs = ForceFieldHiddenData[0].shape[0]
        
                        ForceFieldInputLayer = _construct_input_layer(ForceFieldNrInputs)
                        # Connect force field input to first hidden layer
                        ForceFieldFirstWeights = ForceFieldHiddenLayers[0][0]
                        ForceFieldFirstBiases = ForceFieldHiddenLayers[0][1]
                        ForceFieldNetwork = _connect_layers(ForceFieldInputLayer, ForceFieldFirstWeights, ForceFieldFirstBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
                        #Connect force field hidden layers
                        for l in range(1, len(ForceFieldHiddenLayers)):
                            ForceFieldTempWeights = ForceFieldHiddenLayers[l][0]
                            ForceFieldTempBiases = ForceFieldHiddenLayers[l][1]
                            if l == len(ForceFieldHiddenLayers) - 1:
                                ForceFieldNetwork = _connect_layers(ForceFieldNetwork, ForceFieldTempWeights, ForceFieldTempBiases, "none", NetInstance.ActFunParam,Dropout)
                            else:
                                ForceFieldNetwork = _connect_layers(ForceFieldNetwork, ForceFieldTempWeights, ForceFieldTempBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
                        
        
                    if len(CorrectionHiddenLayers)>0:
                        # Make correction input layer
                        CorrectionHiddenData=NetInstance.HiddenData[i].CorrectionNetworkData
                        CorrectionNrInputs = CorrectionHiddenData[0].shape[0]
        
                        CorrectionInputLayer = _construct_input_layer(CorrectionNrInputs)
                        # Connect Correction input to first hidden layer
                        CorrectionFirstWeights = CorrectionHiddenLayers[0][0]
                        CorrectionFirstBiases = CorrectionHiddenLayers[0][1]
                        CorrectionNetwork = _connect_layers(CorrectionInputLayer, CorrectionFirstWeights, CorrectionFirstBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
                        #Connect Correction hidden layers
                        for l in range(1, len(CorrectionHiddenLayers)):
                            CorrectionTempWeights = CorrectionHiddenLayers[l][0]
                            CorrectionTempBiases = CorrectionHiddenLayers[l][1]
                            if l == len(CorrectionHiddenLayers) - 1:
                                CorrectionNetwork = _connect_layers(CorrectionNetwork, CorrectionTempWeights, CorrectionTempBiases, "none", NetInstance.ActFunParam,Dropout)
                            else:
                                CorrectionNetwork = _connect_layers(CorrectionNetwork, CorrectionTempWeights, CorrectionTempBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
         
                        
                    #Store all networks
                    Network=range(2)
                    if ForceFieldNetwork!=None :
                        Network[0]=ForceFieldNetwork
                    if CorrectionNetwork!=None:
                        Network[1]=CorrectionNetwork
                                      
                    #Store all input layers
                    InputLayer=range(2)
                    if ForceFieldInputLayer!=None :
                        InputLayer[0]=ForceFieldInputLayer
                    if CorrectionInputLayer!=None:
                        InputLayer[1]=CorrectionInputLayer

                    if len(NetInstance.Gs) != 0:
                        if len(NetInstance.Gs) > 0:
                            AtomicNNs.append(
                                [i, Network, InputLayer,NetInstance.Gs[i]])
                        else:
                            AtomicNNs.append(
                                [i, Network, InputLayer])
                    else:
                        AtomicNNs.append(
                            [i, Network, InputLayer])
        else:
            print("No network data found!")
        
        self.AtomicNNs=AtomicNNs
        
   
    def make_atomic_networks(self,NetInstance):
        """Creates the specified partitioned network."""

        AtomicNNs = list()
        AllHiddenLayers=list()
        # Start Session
        if NetInstance.Multiple==False:
            NetInstance.Session=_tf.Session(config=_tf.ConfigProto(
  intra_op_parallelism_threads=_multiprocessing.cpu_count()))
        if len(NetInstance.Structures)!= len(NetInstance.NumberOfAtomsPerType):
            raise ValueError("Length of Structures does not match length of NumberOfAtomsPerType")
        if not(isinstance(NetInstance.Structures[0],_PartitionedStructure)):
            raise ValueError("Please set IsPartitioned = False !")
            
        else:
            # make all the networks for the different atom types
            for i in range(0, len(NetInstance.Structures)):
                
                if len(NetInstance.Dropout)>i:
                    Dropout=NetInstance.Dropout[i]
                else:
                    Dropout=NetInstance.Dropout[-1]
                    
                
                NetworkHiddenLayers=range(2)
                
                ForceFieldHiddenLayers=list()
                CorrectionHiddenLayers=list()
                
                ForceFieldWeights=list()
                CorrectionWeights=list()
                
                ForceFieldBias=list()
                CorrectionBias=list()
                
                ForceFieldNetwork=None
                CorrectionNetwork=None
                
                ForceFieldInputLayer=None
                CorrectionInputLayer=None
                
                CreateNewForceField=True
                CreateNewCorrection=True
                
                # Read structures for specific network parts
                StructureForAtom=NetInstance.Structures[i]
                ForceFieldStructure=StructureForAtom.ForceFieldNetworkStructure
                CorrectionStructure=StructureForAtom.CorrectionNetworkStructure
                #Construct networks out of loaded data
                if len(NetInstance.HiddenData) != 0:
                    #Load data for atom
                    WeightData=NetInstance.HiddenData[i]
                    BiasData=NetInstance.BiasData[i]
                    if len(WeightData.ForceFieldNetworkData)>0:
                        
                        CreateNewForceField=False
                        #Recreate force field network             
                        ForceFieldWeights = WeightData.ForceFieldNetworkData
                        ForceFieldBias = BiasData.ForceFieldNetworkData
        
                        for j in range(1, len(ForceFieldStructure)):
                            
                            ThisWeightData = ForceFieldWeights[j - 1]
                            ThisBiasData = ForceFieldBias[j - 1]
                            ForceFieldNrIn = ThisWeightData.shape[0]
                            ForceFieldNrHidden = ThisWeightData.shape[1]
                            ForceFieldHiddenLayers.append(_construct_hidden_layer(ForceFieldNrIn, ForceFieldNrHidden, NetInstance.HiddenType, ThisWeightData, NetInstance.BiasType,ThisBiasData,WeightData.ForceFieldVariable,NetInstance.InitMean,NetInstance.InitStddev))
    
                        NetworkHiddenLayers[0]=ForceFieldHiddenLayers
                        
                    if len(WeightData.CorrectionNetworkData)>0:   
                        CreateNewCorrection=False
                        #Recreate correction network             
                        CorrectionWeights = WeightData.CorrectionNetworkData
                        CorrectionBias = BiasData.CorrectionNetworkData
        
                        for j in range(1, len(CorrectionStructure)):
                            
                            ThisWeightData = CorrectionWeights[j - 1]
                            ThisBiasData = CorrectionBias[j - 1]
                            CorrectionNrIn = ThisWeightData.shape[0]
                            CorrectionNrHidden = ThisWeightData.shape[1]
                            CorrectionHiddenLayers.append(_construct_hidden_layer(CorrectionNrIn, CorrectionNrHidden, NetInstance.HiddenType, ThisWeightData, NetInstance.BiasType,ThisBiasData,WeightData.CorrectionVariable,NetInstance.InitMean,NetInstance.InitStddev))
                         
                        NetworkHiddenLayers[1]=CorrectionHiddenLayers
                
                if CreateNewForceField==True:
                    #Create force field network
                    for j in range(1, len(ForceFieldStructure)):
                        ForceFieldNrIn = ForceFieldStructure[j - 1]
                        ForceFieldNrHidden = ForceFieldStructure[j]
                        ForceFieldHiddenLayers.append(_construct_hidden_layer(ForceFieldNrIn, ForceFieldNrHidden, NetInstance.HiddenType, [], NetInstance.BiasType,[],True,NetInstance.InitMean,NetInstance.InitStddev))
    
                    NetworkHiddenLayers[0]=ForceFieldHiddenLayers
    
                if CreateNewCorrection==True:
                    #Create correction network
                    for j in range(1, len(CorrectionStructure)):
                        CorrectionNrIn = CorrectionStructure[j - 1]
                        CorrectionNrHidden = CorrectionStructure[j]
                        CorrectionHiddenLayers.append(_construct_hidden_layer(CorrectionNrIn, CorrectionNrHidden, NetInstance.HiddenType, [], NetInstance.BiasType,[],True,NetInstance.InitMean,NetInstance.InitStddev))
                    
                    NetworkHiddenLayers[1]=CorrectionHiddenLayers
                    
                AllHiddenLayers.append(NetworkHiddenLayers)
    
                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):
                    
                    
                    if len(ForceFieldHiddenLayers)>0:
                        # Make force field input layer
                        if CreateNewForceField==False:
                            ForceFieldHiddenData=NetInstance.HiddenData[i].ForceFieldNetworkData
                            ForceFieldNrInputs = ForceFieldHiddenData[0].shape[0]
                        else:
                            ForceFieldNrInputs = ForceFieldStructure[0]
        
                        ForceFieldInputLayer = _construct_input_layer(ForceFieldNrInputs)
                        # Connect force field input to first hidden layer
                        ForceFieldFirstWeights = ForceFieldHiddenLayers[0][0]
                        NetInstance.FirstWeights.append(ForceFieldFirstWeights)
                        ForceFieldFirstBiases = ForceFieldHiddenLayers[0][1]
                        ForceFieldNetwork = _connect_layers(ForceFieldInputLayer, ForceFieldFirstWeights, ForceFieldFirstBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
                        #Connect force field hidden layers
                        for l in range(1, len(ForceFieldHiddenLayers)):
                            ForceFieldTempWeights = ForceFieldHiddenLayers[l][0]
                            ForceFieldTempBiases = ForceFieldHiddenLayers[l][1]
                            if l == len(ForceFieldHiddenLayers) - 1:
                                ForceFieldNetwork = _connect_layers(ForceFieldNetwork, ForceFieldTempWeights, ForceFieldTempBiases, "none", NetInstance.ActFunParam,Dropout)
                            else:
                                ForceFieldNetwork = _connect_layers(ForceFieldNetwork, ForceFieldTempWeights, ForceFieldTempBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
                        
    
                    if len(CorrectionHiddenLayers)>0:
                        # Make correction input layer
                        if CreateNewCorrection==False:
                            CorrectionHiddenData=NetInstance.HiddenData[i].CorrectionNetworkData
                            CorrectionNrInputs = CorrectionHiddenData[0].shape[0]
                        else:
                            CorrectionNrInputs = CorrectionStructure[0]
        
                        CorrectionInputLayer = _construct_input_layer(CorrectionNrInputs)
                        # Connect Correction input to first hidden layer
                        CorrectionFirstWeights = CorrectionHiddenLayers[0][0]
                        NetInstance.FirstWeights.append(CorrectionFirstWeights)
                        CorrectionFirstBiases = CorrectionHiddenLayers[0][1]
                        CorrectionNetwork = _connect_layers(CorrectionInputLayer, CorrectionFirstWeights, CorrectionFirstBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
                        #Connect Correction hidden layers
                        for l in range(1, len(CorrectionHiddenLayers)):
                            CorrectionTempWeights = CorrectionHiddenLayers[l][0]
                            CorrectionTempBiases = CorrectionHiddenLayers[l][1]
                            if l == len(CorrectionHiddenLayers) - 1:
                                CorrectionNetwork = _connect_layers(CorrectionNetwork, CorrectionTempWeights, CorrectionTempBiases, "none", NetInstance.ActFunParam,Dropout)
                            else:
                                CorrectionNetwork = _connect_layers(CorrectionNetwork, CorrectionTempWeights, CorrectionTempBiases, NetInstance.ActFun, NetInstance.ActFunParam,Dropout)
         
                    
                    #Store all networks
                    Network=range(2)
                    if ForceFieldNetwork!=None :
                        Network[0]=ForceFieldNetwork
                    if CorrectionNetwork!=None:
                        Network[1]=CorrectionNetwork
                    
                    #Store all input layers
                    InputLayer=range(2)
                    if ForceFieldInputLayer!=None :
                        InputLayer[0]=ForceFieldInputLayer
                    if CorrectionInputLayer!=None:
                        InputLayer[1]=CorrectionInputLayer
                               
                    if len(NetInstance.Gs) != 0:
                        if len(NetInstance.Gs) > 0:
                            AtomicNNs.append(
                                [i, Network, InputLayer,NetInstance.Gs[i]])
                        else:
                            AtomicNNs.append(
                                [i, Network, InputLayer])
                    else:
                        AtomicNNs.append(
                            [i, Network, InputLayer])
    
            
            self.AtomicNNs=AtomicNNs
            self.VariablesDictionary=AllHiddenLayers    

            
          
class MultipleInstanceTraining(object):
    """This class implements the possibillities to train multiple training
    instances at once. This is neccessary if the datasets have a different
    number of atoms per species. """
    
    def __init__(self):
        #Training variables
        self.TrainingInstances=list()
        self.EpochsPerCycle=1
        self.GlobalEpochs=100
        self.GlobalStructures=list()
        self.GlobalLearningRate=0.001
        self.GlobalCostCriterium=0
        self.Global_dE_Criterium=0
        self.GlobalRegularization="L2"
        self.GlobalRegularizationParam=0.0001
        self.GlobalOptimizer="Adam"
        self.GlobalTrainingCosts=list()
        self.GlobalValidationCosts=list()
        self.GlobalMinOfOut=0
        self.MakePlots=False
        self.IsPartitioned=False
        self.GlobalSession=_tf.Session()
    
    
    def initialize_multiple_instances(self):
        """Initializes all instances with the same parameters."""
        
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
                Instance.dE_Criterium=0
                Instance.IsPartitioned=self.IsPartitioned
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
        """Sets the session of the currently trained instance to the
        global session"""
        
        for Instance in self.TrainingInstances:
            Instance.Session = self.GlobalSession

    def train_multiple_instances(self,StartModelName=None):
        """Trains each instance for EpochsPerCylce epochs then uses the resulting network
        as a basis for the next training instance.
        Args:
            StartModelName (str): Path to a saved model."""
        
        print("Startet multiple instance training!")
        ct=0
        LastStepsModelData=list()
        for i in range(0,self.GlobalEpochs):
            for Instance in self.TrainingInstances:
                if ct==0:
                    if StartModelName!=None:
                        Instance.expand_existing_net(ModelName=StartModelName)
                    else:
                        Instance.make_and_initialize_network()
                else:
                    Instance.expand_existing_net(ModelData=LastStepsModelData)
                
                LastStepsModelData=Instance.start_batch_training()
                _tf.reset_default_graph()
                self.GlobalSession=_tf.Session()
                MultipleInstanceTraining.set_session(self)
                self.GlobalTrainingCosts+=Instance.OverallTrainingCosts
                self.GlobalValidationCosts+=Instance.OverallValidationCosts
                if ct % max(int((self.GlobalEpochs*len(self.TrainingInstances))/50),1)==0 or i==(self.GlobalEpochs-1):
                    if self.MakePlots==True:
                        if ct ==0:
                            fig,ax,TrainingCostPlot,ValidationCostPlot,RunningMeanPlot=_initialize_cost_plot(self.GlobalTrainingCosts,self.GlobalValidationCosts)
                        else:
                            _update_cost_plot(fig,ax,TrainingCostPlot,self.GlobalTrainingCosts,ValidationCostPlot,self.GlobalValidationCosts,RunningMeanPlot)
                    
                    #Finished percentage output
                    print(str(100*ct/(self.GlobalEpochs*len(self.TrainingInstances)))+" %")
                    _np.save("trained_variables",LastStepsModelData)
                ct=ct+1
                #Abort criteria
                if self.GlobalTrainingCosts<=self.GlobalCostCriterium and self.GlobalValidationCosts<=self.GloablCostCriterium or Instance.DeltaE<self.Global_dE_Criterium:
                    
                    if self.GlobalValidationCosts!=0:
                        print("Reached criterium!")
                        print("Cost= "+str((self.GlobalTrainingCosts+self.GlobalValidationCosts)/2))
                        print("delta E = "+str(Instance.DeltaE)+" ev")
                        print("Epoch = "+str(i))
                        print("")
    
                    else:
                        print("Reached criterium!")
                        print("Cost= "+str(self.GlobalTrainingCosts))
                        print("delta E = "+str(Instance.DeltaE)+" ev")
                        print("Epoch = "+str(i))
                        print("")
                        
                    print("Training finished")
                    break
                        
                if i==(self.GlobalEpochs-1):
                    print("Training finished")
                    print("delta E = "+str(Instance.DeltaE)+" ev")
                    print("Epoch = "+str(i))
                    print("")                               
                                           
class _PartitionedStructure(object):
     """This class is a container for the partitioned network structures."""
     
     def __init__(self):
     
         self.ForceFieldNetworkStructure=list()
         self.CorrectionNetworkStructure=list()
         
          
class _PartitionedNetworkData(object):
     """This class is a container for the partitioned network data."""   
     
     def __init__(self):
         
         self.ForceFieldNetworkData=list()
         self.CorrectionNetworkData=list()
         self.ForceFieldVariable=False
         self.CorrectionVariable=False
         
                