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
import ReadLammpsData as _ReaderLammps
import ReadQEData as _ReaderQE


_plt.ion()
_tf.reset_default_graph()


def _construct_input_layer(InputUnits):
    """Construct input for the NN.

    Args:
        InputUnits (int):Number of input units

    Returns:
        Inputs (tensor):The input placeholder
    """

    Inputs = _tf.placeholder(_tf.float32, shape=[None, InputUnits])

    return Inputs


def _construct_hidden_layer(
        PreviousLayerUnits,
        ThisHiddenUnits,
        WeightType=None,
        WeightData=[],
        BiasType=None,
        BiasData=[],
        MakeAllVariable=False,
        Mean=0.0,
        Stddev=1.0):
    """Constructs the weights and biases for this layer with the specified
    initialization.

    Args:
        PreviousLayerUnits (int): Number of units of the previous layer
        ThisHiddenUnits (int): Number of units in this layer
        WeightType (str): Initialization of the weights
            Possible values are:
                zeros,ones,fill,random_normal,truncated_normal,
                random_uniform,random_shuffle,random_crop,
                random_gamma
        WeightData (numpy array): Values for the weight initialization
        BiasType (str): Initialization of the biases
            Possible values are:
                zeros,ones,fill,random_normal,truncated_normal,
                random_uniform,random_shuffle,random_crop,
                random_gamma
        MakeAllVariable (bool): Flag indicating whether the weights and biases
        should be trainable or not.
        Mean (float): Mean value for the normal distribution of the
        initialization.
        Stddev (float):Standard deviation for the normal distribution of the
        initialization.

    Returns:
        Weights (tensor): Weights tensor
        Biases (tensor):Biases tensor
    """
    if len(WeightData) == 0:
        if WeightType is not None:
            if WeightType == "zeros":
                Weights = _tf.Variable(_tf.zeros(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32,
                    name="variable")
            elif WeightType == "ones":
                Weights = _tf.Variable(_tf.ones(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32,
                    name="variable")
            elif WeightType == "fill":
                Weights = _tf.Variable(_tf.fill(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32,
                    name="variable")
            elif WeightType == "random_normal":
                Weights = _tf.Variable(
                    _tf.random_normal(
                        [
                            PreviousLayerUnits,
                            ThisHiddenUnits],
                        mean=Mean,
                        stddev=Stddev),
                    dtype=_tf.float32,
                    name="variable")
            elif WeightType == "truncated_normal":
                Weights = _tf.Variable(
                    _tf.truncated_normal(
                        [
                            PreviousLayerUnits,
                            ThisHiddenUnits],
                        mean=Mean,
                        stddev=Stddev),
                    dtype=_tf.float32,
                    name="variable")
            elif WeightType == "random_uniform":
                Weights = _tf.Variable(_tf.random_uniform(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32, 
                        name="variable")
            elif WeightType == "random_shuffle":
                Weights = _tf.Variable(_tf.random_shuffle(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32, 
                        name="variable")
            elif WeightType == "random_crop":
                Weights = _tf.Variable(_tf.random_crop(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32, 
                        name="variable")
            elif WeightType == "random_gamma":
                Weights = _tf.Variable(_tf.random_gamma(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32, 
                        name="variable")
            else:
                # Assume random weights if no WeightType is given
                Weights = _tf.Variable(_tf.random_uniform(
                    [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32, 
                        name="variable")
        else:
            # Assume random weights if no WeightType is given
            Weights = _tf.Variable(_tf.random_uniform(
                [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32, 
                name="variable")
    else:
        if not MakeAllVariable:
            Weights = _tf.constant(
                WeightData, dtype=_tf.float32, name="constant")
        else:
            Weights = _tf.Variable(
                WeightData, dtype=_tf.float32, name="variable")
    # Construct the bias for this layer
    if len(BiasData) != 0:

        if not MakeAllVariable:
            Biases = _tf.constant(BiasData, dtype=_tf.float32, name="bias")
        else:
            Biases = _tf.Variable(BiasData, dtype=_tf.float32, name="bias")

    else:
        if BiasType == "zeros":
            Biases = _tf.Variable(
                _tf.zeros([ThisHiddenUnits]), dtype=_tf.float32, name="bias")
        elif BiasType == "ones":
            Biases = _tf.Variable(
                _tf.ones([ThisHiddenUnits]), dtype=_tf.float32, name="bias")
        elif BiasType == "fill":
            Biases = _tf.Variable(
                _tf.fill(
                    [ThisHiddenUnits],
                    BiasData),
                dtype=_tf.float32,
                name="bias")
        elif BiasType == "random_normal":
            Biases = _tf.Variable(
                _tf.random_normal(
                    [ThisHiddenUnits],
                    mean=Mean,
                    stddev=Stddev),
                dtype=_tf.float32,
                name="bias")
        elif BiasType == "truncated_normal":
            Biases = _tf.Variable(
                _tf.truncated_normal(
                    [ThisHiddenUnits],
                    mean=Mean,
                    stddev=Stddev),
                dtype=_tf.float32,
                name="bias")
        elif BiasType == "random_uniform":
            Biases = _tf.Variable(_tf.random_uniform(
                [ThisHiddenUnits]), dtype=_tf.float32, name="bias")
        elif BiasType == "random_shuffle":
            Biases = _tf.Variable(_tf.random_shuffle(
                [ThisHiddenUnits]), dtype=_tf.float32, name="bias")
        elif BiasType == "random_crop":
            Biases = _tf.Variable(_tf.random_crop(
                [ThisHiddenUnits], BiasData), dtype=_tf.float32, name="bias")
        elif BiasType == "random_gamma":
            Biases = _tf.Variable(_tf.random_gamma(
                [ThisHiddenUnits], BiasData), dtype=_tf.float32, name="bias")
        else:
            Biases = _tf.Variable(_tf.random_uniform(
                [ThisHiddenUnits]), dtype=_tf.float32, name="bias")

    return Weights, Biases


def _construct_output_layer(OutputUnits):
    """#Constructs the output layer for the NN.

    Args:
        OutputUnits(int): Number of output units

    Returns:
        Outputs (tensor): Output placeholder
    """
    Outputs = _tf.placeholder(_tf.float32, shape=[None, OutputUnits])

    return Outputs


def _construct_not_trainable_layer(NrInputs, NrOutputs, Min):
    """Make a not trainable layer with all weights being 1.

    Args:
        NrInputs (int): Number of units in the previous layer
        NrOutputs (int): Number of units in this layer
        Min (float): Minimum of output shifts network output by
        a constant value

    Returns:
        Weights (tensor): Weights tensor
        Biases (tensor): Biases tensor"""

    # , trainable=False)
    Weights = _tf.constant(_np.ones([NrInputs, NrOutputs]), dtype=_tf.float32)
    # ,trainable=False)
    Biases = _tf.constant(_np.zeros([NrOutputs]), dtype=_tf.float32)
    if Min != 0:
        Biases = _tf.add(Biases, Min / NrOutputs)

    return Weights, Biases


def _connect_layers(InputsForLayer, ThisLayerWeights,
                    ThisLayerBias, ActFun=None, FunParam=None, Dropout=0):
    """Connect the outputs of the layer before with the current layer using an
    activation function and matrix multiplication.

    Args:
        InputsForLayer (tensor): Tensor of the previous layer
        ThisLayerWeights (tensor): Weight tensor for this layer
        ThisLayerBias (tensor): Bias tensor for this layer
        ActFun (str): A value specifying the activation function for the layer
            Possible values are:
                sigmoid,tanh,relu,relu6,crelu,elu,softplus,dropout,bias_add,
                none
        FunParam(parameter): Parameter for the specified activation function
        Dropout (float):Dropout (0-1 =>0%-100%) applied after this layer.

    Returns:
        Out (tensor): The connected tensor."""
    if ActFun is not None:
        if ActFun == "sigmoid":
            Out = _tf.nn.sigmoid(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias)
        elif ActFun == "tanh":
            Out = _tf.nn.tanh(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias)
        elif ActFun == "relu":
            Out = _tf.nn.relu(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias)
        elif ActFun == "relu6":
            Out = _tf.nn.relu6(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias)
        elif ActFun == "crelu":
            Out = _tf.nn.crelu(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias)
        elif ActFun == "elu":
            Out = _tf.nn.elu(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias)
        elif ActFun == "softplus":
            Out = _tf.nn.softplus(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias)
        elif ActFun == "dropout":
            Out = _tf.nn.dropout(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias, FunParam)
        elif ActFun == "bias_add":
            Out = _tf.nn.bias_add(_tf.matmul(
                InputsForLayer, ThisLayerWeights) + ThisLayerBias, FunParam)
        elif ActFun == "none":
            Out = _tf.matmul(InputsForLayer, ThisLayerWeights) + ThisLayerBias
    else:
        Out = _tf.nn.sigmoid(_tf.matmul(
            InputsForLayer, ThisLayerWeights) + ThisLayerBias)

    if Dropout != 0:
        # Apply dropout between layers
        Out = _tf.nn.dropout(Out, Dropout)

    return Out


def make_feed_forward_neuralnetwork(
        Structure,
        WeightType=None,
        WeightData=None,
        BiasType=None,
        BiasData=None,
        ActFun=None,
        ActFunParam=None,
        Dropout=0):
    """Creates a standard NN with the specified structure

    Args:
        Structure (list): List of number of units per layer
        WeightType (str):Specifies the type of weight initialization.
            Possible values are:
                zeros,ones,fill,random_normal,truncated_normal,
                random_uniform,random_shuffle,random_crop,
                random_gamma
        WeightData (numpy array):Values for the weight initialization
        BiasType (str):Specifies the type of bias initilization.
            Possible values are:
                zeros,ones,fill,random_normal,truncated_normal,
                random_uniform,random_shuffle,random_crop,
                random_gamma
        BiasData (numpy array):Values for the bias initialization
        ActFun (str):Specifies the activation function.
            Possible values are:
                sigmoid,tanh,relu,relu6,crelu,elu,softplus,dropout,bias_add,
                none
        ActFunParam(parameter): Parameter for the specified activation function
        Dropout (float):Dropout (0-1 =>0%-100%) applied after this layer.

    Returns:
        Network(tensor):Output of the NN
        InputLayer(tensor):Input placeholders
        OutputLayer (tensor): Output placeholders
    """
    # Make inputs
    NrInputs = Structure[0]
    InputLayer = _construct_input_layer(NrInputs)
    # Make hidden layers
    HiddenLayers = list()
    for i in range(1, len(Structure)):
        NrIn = Structure[i - 1]
        NrHidden = Structure[i]
        HiddenLayers.append(_construct_hidden_layer(
            NrIn, NrHidden, WeightType, WeightData[i - 1], BiasType,
            BiasData[i - 1]))

    # Make output layer
    OutputLayer = _construct_output_layer(Structure[-1])

   # Connect input to first hidden layer
    FirstWeights = HiddenLayers[0][0]
    FirstBiases = HiddenLayers[0][1]
    InConnection = _connect_layers(
        InputLayer, FirstWeights, FirstBiases, ActFun, ActFunParam, Dropout)
    Network = InConnection

    for j in range(1, len(HiddenLayers)):
       # Connect ouput of in layer to second hidden layer
        if j == 1:
            SecondWeights = HiddenLayers[j][0]
            SecondBiases = HiddenLayers[j][1]
            Network = _connect_layers(
                Network,
                SecondWeights,
                SecondBiases,
                ActFun,
                ActFunParam,
                Dropout)
        else:
            Weights = HiddenLayers[j][0]
            Biases = HiddenLayers[j][1]
            Network = _connect_layers(
                Network, Weights, Biases, ActFun, ActFunParam, Dropout)

    return Network, InputLayer, OutputLayer


def cost_for_network(Prediction, ReferenceValue, Type):
    """
    Creates the specified cost function.

    Args:
        Prediction (tensor):Prediction tensor of the network
        ReferenceValue (tensor):Placeholder for the reference values
        Type (str): Type of cost function
            Possible values:
                squared-difference,Adaptive_1,Adaptive_2

    Returns:
        Cost(tensor): The cost function as a tensor."""

    if Type == "squared-difference":
        Cost = 0.5 * _tf.reduce_sum((Prediction - ReferenceValue)**2)
    elif Type == "Adaptive_1":
        epsilon = 10e-9
        Cost = 0.5 * _tf.reduce_sum((Prediction - ReferenceValue)**2
                                    * (_tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)) - 0.5)
                                    + (0.5 + _tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)))
                                    * _tf.pow(_tf.abs(Prediction - ReferenceValue + epsilon), 1.25))
    elif Type == "Adaptive_2":
        epsilon = 10e-9
        Cost = 0.5 * _tf.reduce_sum((Prediction - ReferenceValue)**2
                                    * (_tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)) - 0.5)
                                    + (0.5 + _tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)))
                                    * _tf.abs(Prediction - ReferenceValue + epsilon))

    return Cost


def running_mean(x, N):
    """Calculates the runnung average over N steps.

    Args:
        x(numpy array): Vector for the calculation
        N (int): Number of values to average over
    Returns:
        A vector containing the running average of the input vector

    """
    cumsum = _np.cumsum(_np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def _initialize_cost_plot(TrainingData, ValidationData=[]):
    """Initializes the cost plot for the training
    Returns the figure and the different plots"""
    fig = _plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscaley_on(True)
    TrainingCostPlot, = ax.semilogy(
        _np.arange(0, len(TrainingData)), TrainingData)
    if len(ValidationData) != 0:
        ValidationCostPlot, = ax.semilogy(
            _np.arange(0, len(ValidationData)), ValidationData)
    else:
        ValidationCostPlot = None
    # add running average plot
    running_avg = running_mean(TrainingData, 1000)
    RunningMeanPlot, = ax.semilogy(
        _np.arange(0, len(running_avg)), running_avg)

    # Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel("batches")
    ax.set_ylabel("log(cost)")
    ax.set_title("Normalized cost per batch")
    if len(ValidationData) != 0:
        fig.legend(handles=[TrainingCostPlot,ValidationCostPlot,RunningMeanPlot],labels=["Training cost","Valiation cost","Running avg"])
    else:
        fig.legend(handles=[TrainingCostPlot,RunningMeanPlot],labels=["Training cost","Running avg"])
        
    # We need to draw *and* flush
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax, TrainingCostPlot, ValidationCostPlot, RunningMeanPlot


def _update_cost_plot(
        figure,
        ax,
        TrainingCostPlot,
        TrainingCost,
        ValidationCostPlot=None,
        ValidationCost=None,
        RunningMeanPlot=None):
    """Updates the cost plot with new data"""

    TrainingCostPlot.set_data(_np.arange(0, len(TrainingCost)), TrainingCost)
    if ValidationCostPlot is not None:
        ValidationCostPlot.set_data(_np.arange(
            0, len(ValidationCost)), ValidationCost)

    if RunningMeanPlot is not None:
        running_avg = running_mean(TrainingCost, 1000)
        RunningMeanPlot.set_data(_np.arange(0, len(running_avg)), running_avg)
    # Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    # We need to draw *and* flush
    figure.canvas.draw()
    figure.canvas.flush_events()


def _initialize_weights_plot(sparse_weights, n_gs):
    """Initializes the plot of the absolute value of the weights
    Can be used to identify redundant symmetry functions
    Returns the figure and the plot"""

    fig = _plt.figure()
    ax = fig.add_subplot(111)
    weights_plot = ax.bar(_np.arange(n_gs), sparse_weights)
    ax.set_autoscaley_on(True)

    # Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel("Symmetry function")
    ax.set_ylabel("Weights")
    ax.set_title("Weights for symmetry functions")
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, weights_plot


def _update_weights_plot(fig, weights_plot, sparse_weights):
    """Updates the plot of the absolute value of the weights
    Can be used to identify redundant symmetry functions
    Returns the figure and the plot"""

    for u, rect in enumerate(weights_plot):
        rect.set_height(sparse_weights[u])
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, weights_plot


def cartesian_to_spherical(xyz):
    """Converts cartesion to spherical coordinates
    Returns the spherical coordinates"""

    spherical = _np.zeros_like(xyz)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    spherical[:, 0] = _np.sqrt(xy + xyz[:, 2]**2)
    spherical[:, 1] = _np.arctan2(xyz[:, 2], _np.sqrt(xy))
    spherical[:, 2] = _np.arctan2(xyz[:, 1], xyz[:, 0])
    return spherical


def _get_learning_rate(StartLearningRate, LearningRateType,
                       decay_steps, boundaries=[], values=[]):
    """Returns a tensor for the learning rate
    Learning rate can be decayed with different methods.
    Returns the tensors for the global step and the learning rate"""

    if LearningRateType == "none":
        global_step = _tf.Variable(0, trainable=False)
        return global_step, StartLearningRate
    elif LearningRateType == "exponential_decay":
        global_step = _tf.Variable(0, trainable=False)
        return global_step, _tf.train.exponential_decay(
            StartLearningRate, global_step, decay_steps, decay_rate=0.96,
            staircase=False)
    elif LearningRateType == "inverse_time_decay":
        global_step = _tf.Variable(0, trainable=False)
        return global_step, _tf.train.inverse_time_decay(
            StartLearningRate, global_step, decay_steps, decay_rate=0.96, 
            staircase=False)
    elif LearningRateType == "piecewise_constant":
        global_step = _tf.Variable(0, trainable=False)
        return global_step, _tf.train.piecewise_constant(
            global_step, boundaries, values)
    elif LearningRateType == "polynomial_decay_p1":
        global_step = _tf.Variable(0, trainable=False)
        return global_step, _tf.train.polynomial_decay(
            StartLearningRate, global_step, decay_steps, 
            end_learning_rate=0.00001, power=1.0, cycle=False)
    elif LearningRateType == "polynomial_decay_p2":
        global_step = _tf.Variable(0, trainable=False)
        return global_step, _tf.train.polynomial_decay(
            StartLearningRate, global_step, decay_steps, 
            end_learning_rate=0.00001, power=2.0, cycle=False)


def _calc_distance_to_all_atoms(xyz):
    """Calculates the distances between all atoms
    Returns a matrix with all distances."""

    Ngeom = len(xyz[:, 0])
    distances = _np.zeros((Ngeom, Ngeom))
    for i in range(Ngeom):
        distances[i, :] = _np.linalg.norm(xyz - xyz[i, :])

    return distances


def _get_ds_r_min_r_max(geoms):
    """Gets the maximum and minimum radial distance within the dataset
    Returns a min,max value as floats."""

    r_min = 10e10
    r_max = 0
    for geom in geoms:
        np_geom = _np.zeros((len(geom), 3))
        np_dist = _np.zeros((len(geom), len(geom)))
        for i, atom in enumerate(geom):
            xyz = atom[1]
            np_geom[i, :] = xyz
        np_dist = _calc_distance_to_all_atoms(np_geom)
        L = np_dist != 0
        r_min_tmp = _np.min(np_dist[L])
        r_max_tmp = _np.max(np_dist)
        if r_min_tmp < r_min:
            r_min = r_min_tmp

        if r_max_tmp > r_max:
            r_max = r_max_tmp

    return r_min, r_max


def _create_zero_diff_geometries(r_min, r_max, types, N_atoms_per_type, N):
    """Creates geometries outside the dataset for a fixed energy
    to prevent the network from performing badly outside the training area.
    The radial coverage area of the training data has to be specified via
    r_min and r_max, the types of atoms as a list: ["species1",species2] and
    the number of atoms per atom type have to be specified as a list:[1,2]
    Returns N geometries as a list."""

    out_geoms = []
    Natoms = sum(N_atoms_per_type)
    np_geom = _np.zeros((Natoms, 3))
    np_dist = _np.zeros((Natoms, Natoms))
    for i in range(N):
        # try to get valid position for atom
        run = True
        while(run):
            # calucalte random position of atom
            geom = []
            ct = 0
            for j in range(len(N_atoms_per_type)):
                for k in range(N_atoms_per_type[j]):
                    r = _rand.uniform(0, 5 * r_max)
                    phi = _rand.uniform(0, 2 * _np.pi)
                    theta = _rand.uniform(0, _np.pi)
                    x = r * _np.cos(theta) * _np.cos(phi)
                    y = r * _np.cos(theta) * _np.sin(phi)
                    z = r * _np.sin(theta)
                    xyz = [x, y, z]
                    a_type = types[j]
                    atom = (a_type, _np.asarray(xyz))
                    np_geom[ct, :] = xyz
                    ct += 1
                    geom.append(atom)

            np_dist = _calc_distance_to_all_atoms(np_geom)
            L = np_dist != 0
            if _np.all(np_dist[L]) > r_max or _np.all(np_dist[L]) < r_min:
                out_geoms.append(geom)
                run = False

    return out_geoms


def parse_qchem_geometries(in_geoms):
    """Parse q-chem format to NN compatible format
    Returns the geometries in a compatible list"""

    out_geoms = []
    for in_geom in in_geoms:
        atoms_list = in_geom.list_of_atoms
        out_geom = []
        for atom in atoms_list:
            xyz = [float(atom[1]), float(atom[2]), float(atom[3])]
            my_tuple = (atom[0], _np.asarray(xyz))
            out_geom.append(my_tuple)
        out_geoms.append(out_geom)

    return out_geoms


class AtomicNeuralNetInstance(object):
    """This class implements all the properties and methods for training and
    evaluating the network"""

    def __init__(self):

        # Public variables

        # Net settings
        self.Structures = []
        self.Atomtypes = []
        self.SizeOfInputsPerType = []
        self.SizeOfInputsPerAtom = []
        self.NumberOfAtomsPerType = []
        self.WeightType = "truncated_normal"
        self.BiasType = "truncated_normal"
        self.ActFun = "elu"
        self.ActFunParam = None
        self.MakeLastLayerConstant = False
        self.Dropout = [0]
        self.IsPartitioned = False
        self.ClippingValue = 100
        # Data
        self.EvalData = []
        self.TrainingBatches = []
        self.ValidationBatches = []

        # Training variables
        self.Epochs = 1000
        self.GlobalStep = None
        self.LearningRate = 0.01
        self.LearningRateFun = None
        self.LearningRateType = "none"
        self.LearningDecayEpochs = 100
        self.LearningRateBounds = []
        self.LearningRateValues = []
        self.CostFun = None
        self.MakePlots = False
        self.InitMean = 0.0
        self.InitStddev = 1.0
        self.MakeAllVariable = True
        self.Regularization = "none"
        self.RegularizationParam = 0.0
        self.ForceCostParam = 0.00001
        self.InputDerivatives = False
        self.Multiple = False
        self.UseForce = False
        self.CostCriterium = 0
        self.dE_Criterium = 0
        self.OptimizerType = None
        self.OptimizerProp = None
        self.DeltaE = 0
        self.TrainingCosts = []
        self.ValidationCosts = []
        self.OverallTrainingCosts = []
        self.OverallValidationCosts = []
        self.TrainedVariables = []
        self.CostFunType = "squared-difference"
        self.SavingDirectory = "save"
        # Symmetry function set settings
        self.NumberOfRadialFunctions = 20
        self.Rs = []
        self.Etas = []
        self.Zetas = []
        self.Lambs = []

        # Private variables

        # Class instances
        self._Session = []
        self._Net = None
        self._SymmFunSet = None
        self._DataSet = None
        # Inputs
        self._TrainingInputs = []
        self._TrainingOutputs = []
        self._ValidationInputs = []
        self._ValidationOutputs = []
        self._ForceTrainingInput = []
        self._ForceValidationInput = []
        self._ForceTrainingOutput = []
        self._ForceValidationOutput = []
        # Net parameters
        self._MeansOfDs = []
        self._MinOfOut = None
        self._VarianceOfDs = []
        self._BiasData = []
        self._WeightData = []
        self._FirstWeights = []
        self._dE_Fun = None
        self._CurrentEpochNr = 0
        # Output tensors
        self._OutputLayerForce = None
        self._Optimizer = None
        self._ForceOptimizer = None
        self._OutputLayer = None
        self._OutputForce = None
        self._TotalEnergy = None
        self._ForceCost = None
        self._EnergyCost = None
        self._RegLoss = None
        # Dataset
        self._Reader = None
        self._AllGeometries = []
        self._AllGDerivatives = []

    def get_optimizer(self, CostFun):

        # Set optimizer
        Optimizer = _tf.train.GradientDescentOptimizer(self.LearningRateFun)
        if self.OptimizerType == "GradientDescent":
            Optimizer = _tf.train.GradientDescentOptimizer(
                self.LearningRateFun)
        elif self.OptimizerType == "Adagrad":
            self._Optimizer = _tf.train.AdagradOptimizer(self.LearningRateFun)
        elif self.OptimizerType == "Adadelta":
            Optimizer = _tf.train.AdadeltaOptimizer(self.LearningRateFun)
        elif self.OptimizerType == "AdagradDA":
            Optimizer = _tf.train.AdagradDAOptimizer(
                self.LearningRateFun, self.OptimizerProp)
        elif self.OptimizerType == "Momentum":
            Optimizer = _tf.train.MomentumOptimizer(
                self.LearningRateFun, self.OptimizerProp)
        elif self.OptimizerType == "Adam":
            Optimizer = _tf.train.AdamOptimizer(
                self.LearningRateFun, beta1=0.9, beta2=0.999, epsilon=1e-08)
        elif self.OptimizerType == "Ftrl":
            Optimizer = _tf.train.FtrlOptimizer(self.LearningRateFun)
        elif self.OptimizerType == "ProximalGradientDescent":
            Optimizer = _tf.train.ProximalGradientDescentOptimizer(
                self.LearningRateFun)
        elif self.OptimizerType == "ProximalAdagrad":
            Optimizer = _tf.train.ProximalAdagradOptimizer(
                self.LearningRateFun)
        elif self.OptimizerType == "RMSProp":
            Optimizer = _tf.train.RMSPropOptimizer(self.LearningRateFun)
        else:
            Optimizer = _tf.train.GradientDescentOptimizer(
                self.LearningRateFun)

        # clipped minimization
        gvs = Optimizer.compute_gradients(self.CostFun)
        capped_gvs = [(_tf.clip_by_value(_tf.where(_tf.is_finite(grad), grad, 
                                                   _tf.zeros_like(grad)),
                        -self.ClippingValue, self.ClippingValue), var) 
                        for grad, var in gvs]
        Optimizer = Optimizer.apply_gradients(
            capped_gvs, global_step=self.GlobalStep)

        return Optimizer

    def initialize_network(self):
        """Initializes the network for training by starting a session and
        getting the placeholder for the output, the cost function, 
        the learning rate and the optimizer.
        """
        try:
            # Make virtual output layer for feeding the data to the cost
            # function
            self._OutputLayer = _construct_output_layer(1)
            if self.UseForce:
                self._OutputLayerForce = _construct_output_layer(
                    sum(self.NumberOfAtomsPerType) * 3)
            # Cost function for whole net
            self.CostFun = self._atomic_cost_function()
            # if self.IsPartitioned==True:

            decay_steps = len(self.TrainingBatches) * self.LearningDecayEpochs
            self.GlobalStep, self.LearningRateFun = _get_learning_rate(
                self.LearningRate, self.LearningRateType, decay_steps, 
                self.LearningRateBounds, self.LearningRateValues)

            # Set optimizer
            self._Optimizer = self.get_optimizer(self.CostFun)
        except BaseException:
            print("Evaluation only no training\
                  supported if all networks are constant!")
            # Initialize session
        self._Session.run(_tf.global_variables_initializer())

    def load_model(self, ModelName="save/trained_variables"):
        """Loads the model in the specified folder.
        Args:
            ModelName(str):Path to the model.
        Returns:
            1
        """
        if ".npy" not in ModelName:
            ModelName = ModelName + ".npy"
            rare_model = _np.load(ModelName)
            self.TrainedVariables = rare_model[0]
            self._MeansOfDs = rare_model[1]
            self._VarianceOfDs = rare_model[2]
            self._MinOfOut = rare_model[3]

        return 1

    def expand_existing_net(
            self,
            ModelName="save/trained_variables",
            MakeAllVariable=True,
            ModelData=None,
            ConvertToPartitioned=False):
        """Creates a new network out of stored data.
        Args:
            MakeAllVariables(bool): Specifies if all layers can be trained
            ModelData (list): Passes the model directly from a training before.
            ConvertToPartitioned(bool):Converts a StandardAtomicNetwork to a
            PartitionedAtomicNetwork network with the StandardAtomicNetwork
            beeing the force network part."""

        if ModelData is None:
            Success = self.load_model(ModelName)
        else:
            self.TrainedVariables = ModelData[0]
            self._MinOfOut = ModelData[1]
            Success = 1
        if Success == 1:
            print("Model successfully loaded!")

            if not self.IsPartitioned:
                if ConvertToPartitioned:
                    raise(ValueError)
                self._Net = _StandardAtomicNetwork()
            else:
                self._Net = _PartitionedAtomicNetwork()

            if ConvertToPartitioned:
                self._WeightData, self._BiasData = \
                self._convert_standard_to_partitioned_net()
            else:
                self._WeightData, self._BiasData = \
                self._Net.get_weights_biases_from_data(
                    self.TrainedVariables, self.Multiple)

            self.WeightType = "truncated_normal"
            self.InitMean = 0
            self.InitStddev = 0.01
            self.BiasType = "zeros"
            self.MakeAllVariable = MakeAllVariable
            # try:
            self.make_and_initialize_network()
            # except:
            #    print("Partitioned network loaded, please set IsPartitioned=True")

    def _convert_standard_to_partitioned_net(self):
        """Converts a standard (Behler) network to the force-field part of  a
        partitioned network.

        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""

        WeightData, BiasData = self._Net.get_weights_biases_from_data(
            self.TrainedVariables, self.Multiple)
        OutWeights = list()
        OutBiases = list()
        for i in range(0, len(self.TrainedVariables)):
            Network = self.TrainedVariables[i]
            DataStructWeights = _PartitionedNetworkData()
            DataStructBiases = _PartitionedNetworkData()
            for j in range(0, len(Network)):
                Weights = WeightData[i][j]
                Biases = BiasData[i][j]
                DataStructWeights.ForceFieldNetworkData.append(Weights)
                DataStructBiases.ForceFieldNetworkData.append(Biases)

            OutWeights.append(DataStructWeights)
            OutBiases.append(DataStructBiases)

        return OutWeights, OutBiases

    def evaluate(self, Tensor, Layers, Data):
        """Evaluate model for given input data
        Returns:
            The output of the given tensor"""
        if len(Layers) == 1:
            return self._Session.run(Tensor, feed_dict={Layers[0]: Data})
        else:
            return self._Session.run(
                Tensor, feed_dict={
                    i: _np.array(d) for i, d in zip(
                        Layers, Data)})

    def calc_dE(self, Layers, Data):
        """Returns:
            A tensor for the energy error of the dataset."""
        return _np.nan_to_num(
            _np.mean(self.evaluate(self._dE_Fun, Layers, Data)))

    def _train_step(self, Layers, Data):
        """Does ones training step(one batch).
        Returns:
            Cost(float):The training cost for the step."""
        # Train the network for one step
        _, Cost = self._Session.run([self._Optimizer, self.CostFun], 
                feed_dict={i: _np.array(d) for i, d in zip(Layers, Data)})

        return Cost

    def _validate_step(self, Layers, Data):
        """Calculates the validation cost for this training step,
        without optimizing the net.

        Returns:
            Cost (float): The cost for the data"""
        # Evaluate cost function without changing the network
        Cost = self._Session.run(
            self.CostFun, feed_dict={
                i: _np.array(d) for i, d in zip(
                    Layers, Data)})

        return Cost

    def _train_atomic_network_batch(
            self, Layers, TrainingData, ValidationData):
        """Trains one batch for an atomic network.

        Returns:
            TrainingCost (float) The cost for the training step
            ValidationCost(float) The cost for the validation data"""

        TrainingCost = 0
        ValidationCost = 0
        # train batch
        TrainingCost = self._train_step(Layers, TrainingData)

        # check validation dataset error
        if ValidationData is not None:
            ValidationCost = self._validate_step(Layers, ValidationData)

        return TrainingCost, ValidationCost

    def _train_atomic_networks(
            self,
            TrainingInputs,
            TrainingOutputs,
            ValidationInputs=None,
            ValidationOutputs=None):
        """Trains one step for an atomic network.
        First it prepares the input data and the placeholder and then executes 
        a training step.
        Returns:
            the current session,the trained network and the cost for the
        training step """

        TrainCost = []
        ValidationCost = []
        ValidationCost = 0
        TrainCost = 0
        # Prepare data environment for training
        Layers, TrainingData = self._Net.prepare_data_environment(
            TrainingInputs, self._OutputLayer, TrainingOutputs)
        # Make validation input vector
        if len(ValidationInputs) > 0:
            ValidationData = self._Net.make_data_for_atomicNNs(
                ValidationInputs, ValidationOutputs)
        else:
            ValidationData = None

        print("Started training...")
        # Start training of the atomic network
        for i in range(self.Epochs):
            Cost = self._train_step(Layers, TrainingData)
            TrainCost.append(sum(Cost) / len(Cost))
            # check validation dataset error
            if ValidationData is not None:
                temp_val = self._validate_step(Layers, ValidationData)
                ValidationCost.append(sum(temp_val) / len(temp_val))

            if i % max(int(self.Epochs / 100), 1) == 0:
                if self.MakePlot:
                    if i == 0:
                        figure, ax, TrainPlot, ValPlot, RunningMeanPlot = \
                        _initialize_cost_plot(
                            TrainCost, ValidationCost)
                    else:
                        _update_cost_plot(
                            figure,
                            ax,
                            TrainPlot,
                            TrainCost,
                            ValPlot,
                            ValidationCost,
                            RunningMeanPlot)
                print(str(100 * i / self.Epochs) + " %")

            if TrainCost[-1] < self.CostCriterium:
                print(TrainCost[-1])
                break

        return TrainCost, ValidationCost

    def make_network(self):
        """Creates the specified network"""

        Execute = True
        if len(self.Structures) == 0:
            print("No structures for the specific nets specified!")
            Execute = False
        if not self.IsPartitioned:
            if len(self.Structures[0]) - 1 < len(self.Dropout):
                print("Dropout can only be between layers so it must be\
                      shorter than the structure,\n but is " +
                      str(len(self.Structures[0])) + " and "
                      + str(len(self.Dropout)))
                Execute = False
        if len(self.NumberOfAtomsPerType) == 0:
            print("No number of specific nets specified!")
            Execute = False

        if Execute:
            if self._Net is None:
                if not self.IsPartitioned:
                    self._Net = _StandardAtomicNetwork()
                else:
                    self._Net = _PartitionedAtomicNetwork()

        self._Net.make_atomic_networks(self)

    def make_and_initialize_network(self):
        """Creates and initializes the specified network"""

        self.make_network()
        self.initialize_network()

    def start_training(self):
        """Start the training without the use of batch training.
        Returns:
            self.TrainingCost(list):The costs for the training data.
            self.ValidationCost(list):The costs for the validation data."""
        Execute = True
        if len(self._Net.AtomicNNs) == 0:
            print("No atomic neural nets available!")
            Execute = False
        if len(self._TrainingInputs) == 0:
            print("No training inputs specified!")
            Execute = False
        if len(self._TrainingOutputs) == 0:
            print("No training outputs specified!")
            Execute = False

        if Execute:
            TrainingCosts, ValidationCosts = self._train_atomic_networks(
                self._TrainingInputs, self._TrainingOutputs, 
                self._ValidationInputs, self._ValidationOutputs)
            self.TrainedVariables = self._Net.get_trained_variables(
                self._Session)
            # Store variables

            if not _os.path.exists(self.SavingDirectory):
                _os.makedirs(self.SavingDirectory)
            _np.save(self.SavingDirectory +
                     "/trained_variables", self.TrainedVariables)

            self.TrainingCosts = TrainingCosts
            self.ValidationCosts = ValidationCosts
            print("Training finished")

        return self.TrainingCosts, self.ValidationCosts

    def expand_trained_net(self, nAtoms, ModelName=None):
        """Expands a stored net for the specified atoms.
        Args:
            nAtoms (list): Is a list of integers:[1,2]
            ModelName (str): Is the path to the stored file"""

        self.NumberOfAtomsPerType = nAtoms
        self.expand_existing_net(ModelName)

    def _dE_stat(self, Layers):
        """Calculates the energy error for the whole dataset.

        Args:
            Layers(list):List of sorted placeholders

        Returns:
            train_stat (list) List consisting of the mean value and the
            variance of the dataset for training data.
            vals_stat (list) List consisting of the mean value and the
            variance of the dataset for validation data."""

        train_stat = []
        train_dE = 0
        train_var = 0
        val_stat = []
        val_dE = 0
        val_var = 0

        for i in range(0, len(self.TrainingBatches)):
            TrainingInputs = self.TrainingBatches[i][0]
            TrainingOutputs = self.TrainingBatches[i][1]

            TrainingData = self._Net.make_data_for_atomicNNs(
                TrainingInputs, TrainingOutputs, AppendForce=False)

            if i == 0:
                train_dE = self.evaluate(self._dE_Fun, Layers, TrainingData)
            else:
                temp = self.evaluate(self._dE_Fun, Layers, TrainingData)
                train_dE = _tf.concat([train_dE, temp], 0)

        for i in range(0, len(self.ValidationBatches)):
            ValidationInputs = self.ValidationBatches[i][0]
            ValidationOutputs = self.ValidationBatches[i][1]

            ValidationData = self._Net.make_data_for_atomicNNs(
                ValidationInputs, ValidationOutputs, AppendForce=False)

            if i == 0:
                val_dE = self.evaluate(self._dE_Fun, Layers, ValidationData)
            else:
                temp = self.evaluate(self._dE_Fun, Layers, ValidationData)
                val_dE = _tf.concat([val_dE, temp], 0)

        with self._Session.as_default():
            train_dE = train_dE.eval().tolist()
            val_dE = val_dE.eval().tolist()

        train_mean = _np.mean(train_dE)
        train_var = _np.var(train_dE)
        val_mean = _np.mean(val_dE)
        val_var = _np.var(val_dE)

        train_stat = [train_mean, train_var]
        val_stat = [val_mean, val_var]

        return train_stat, val_stat

    def energy_for_geometry(self, geometry):
        self.create_eval_data(geometry)

        return self.eval_dataset_energy(self.EvalData)

    def force_for_geometry(self, geometry):
        self.create_eval_data(geometry)

        return self.eval_dataset_force(self.EvalData)

    def eval_dataset_energy(self, Batches, BatchNr=0):
        """Prepares and evaluates the dataset for the loaded network.

        Args:
            Batches (list):List of a list of numpy arrays.
            BatchNr (int): Index of the batch to evaluate.

        Returns:
            List of the predicted energies for the dataset
        """
        AllData = Batches[BatchNr]
        GData = AllData[0]
        Layers, Data = self._Net.prepare_data_environment(
            GData, AppendForce=False)

        return self.evaluate(self._TotalEnergy, Layers, Data)

    def eval_dataset_force(self, Batches, BatchNr=0):
        """Prepares and evaluates the dataset for the loaded network.

        Args:
            Batches (list):List of a list of numpy arrays.
            BatchNr (int): Index of the batch to evaluate.

        Returns:
            List of the predicted forces for the dataset"""

        AllData = Batches[BatchNr]
        GData = AllData[0]
        DerGData = AllData[2]
        norm = AllData[3]
        if not self.IsPartitioned:
            Layers, Data = self._Net.prepare_data_environment(
                GData, None, [], None, DerGData, Normalization=norm,
                AppendForce=True)
        else:
            raise(NotImplementedError)

        return self.evaluate(self._OutputForce, Layers, Data)

    def start_evaluation(self, nAtoms, ModelName="save/trained_variables"):
        """Recreates a saved network,prepares and evaluates the specified
        dataset.

        Args:
            nAtoms (list): List of number of atoms per type
            ModelName (str):Path of model.
        Returns:
            Out (list):The predicted energies for the dataset."""

        Out = 0
        self.expand_trained_net(nAtoms, ModelName)
        for i in range(len(self.EvalData)):
            Out = self.eval_dataset_energy(self.EvalData, i)

        return Out

    def _eval_step(self):
        """Evaluates the prepared data.
        Returns:
            Out (list) List of network outputs (energies)."""
        Out = self._Net.evaluate_all_atomicnns(self._TrainingInputs)

        return Out

    def start_batch_training(self, find_best_symmfuns=False):
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
            [self.TrainedVariables,self._MinOfOut] (list):
            The trained network in as a list
            (self._MinOfOut is the offset of the last bias node, necessary
             for tanh or sigmoid activation functions in the last layer)."""

        # Clear cost array for multi instance training
        self.OverallTrainingCosts = []
        self.OverallValidationCosts = []
        NormalizationTraining = []
        NormalizationValidation = []

        start = _time.time()
        Execute = True
        if len(self._Net.AtomicNNs) == 0:
            print("No atomic neural nets available!")
            Execute = False
        if len(self.TrainingBatches) == 0:
            print("No training batches specified!")
            Execute = False

        if sum(self.NumberOfAtomsPerType) != len(self.TrainingBatches[0][0]):
            print([self.NumberOfAtomsPerType, len(self.TrainingBatches[0][0])])
            print("Input does not match number of specified networks!")
            Execute = False

        if Execute:
            if not self.Multiple:
                print("Started batch training...")
            NrOfTrainingBatches = len(self.TrainingBatches)
            if self.ValidationBatches:
                NrOfValidationBatches = len(self.ValidationBatches)

            for i in range(0, self.Epochs):
                self._CurrentEpochNr = i
                for j in range(0, NrOfTrainingBatches):

                    tempTrainingCost = []
                    tempValidationCost = []
                    rnd = _rand.randint(0, NrOfTrainingBatches - 1)
                    self._TrainingInputs = self.TrainingBatches[rnd][0]
                    self._TrainingOutputs = self.TrainingBatches[rnd][1]

                    BatchSize = self._TrainingInputs[0].shape[0]
                    if self.ValidationBatches:
                        rnd = _rand.randint(0, NrOfValidationBatches - 1)
                        self._ValidationInputs = self.ValidationBatches[rnd][0]
                        self._ValidationOutputs = self.ValidationBatches[rnd][1]
                    if self.UseForce:
                        self._ForceTrainingInput = self.TrainingBatches[rnd][2]
                        NormalizationTraining = self.TrainingBatches[rnd][3]
                        self._ForceTrainingOutput = self.TrainingBatches[rnd][4]
                        if self.ValidationBatches:
                            self._ForceValidationInput = self.ValidationBatches[rnd][2]
                            NormalizationValidation = self.ValidationBatches[rnd][3]
                            self._ForceValidationOutput = self.ValidationBatches[rnd][4]

                    # Prepare data and layers for feeding
                    if i == 0:
                        EnergyLayers = self._Net.make_layers_for_atomicNNs(
                            self._OutputLayer, [], False)
                        Layers, TrainingData = self._Net.prepare_data_environment(
                            self._TrainingInputs, self._OutputLayer, self._TrainingOutputs, self._OutputLayerForce, self._ForceTrainingInput, self._ForceTrainingOutput, NormalizationTraining, self.UseForce)
                    else:
                        TrainingData = self._Net.make_data_for_atomicNNs(
                            self._TrainingInputs,
                            self._TrainingOutputs,
                            self._ForceTrainingInput,
                            self._ForceTrainingOutput,
                            NormalizationTraining,
                            self.UseForce)
                    # Make validation input vector
                    if len(self._ValidationInputs) > 0:
                        ValidationData = self._Net.make_data_for_atomicNNs(
                            self._ValidationInputs,
                            self._ValidationOutputs,
                            self._ForceValidationInput,
                            self._ForceValidationOutput,
                            NormalizationValidation,
                            self.UseForce)
                    else:
                        ValidationData = None

                    # Train one batch
                    TrainingCosts, ValidationCosts = self._train_atomic_network_batch(
                        Layers, TrainingData, ValidationData)

                    tempTrainingCost.append(TrainingCosts)
                    tempValidationCost.append(ValidationCosts)

                    self.OverallTrainingCosts.append(TrainingCosts / BatchSize)
                    self.OverallValidationCosts.append(
                        ValidationCosts / BatchSize)

                if len(tempTrainingCost) > 0:
                    self.TrainingCosts = sum(
                        tempTrainingCost) / (len(tempTrainingCost) * BatchSize)
                    self.ValidationCosts = sum(
                        tempValidationCost) / (len(tempValidationCost) * BatchSize)
                else:
                    self.TrainingCosts = 1e10
                    self.ValidationCosts = 1e10

                if self.ValidationCosts != 0:
                    self.DeltaE = (self.calc_dE(Layers, TrainingData) +
                                   self.calc_dE(Layers, TrainingData)) / 2
                else:
                    self.DeltaE = self.calc_dE(Layers, TrainingData)

                if not self.Multiple:
                    if i % max(int(self.Epochs / 20),
                               1) == 0 or i == (self.Epochs - 1):
                        # Cost plot
                        if self.MakePlots:
                            if i == 0:
                                if find_best_symmfuns:
                                    # only supports force field at the moment
                                    sparse_tensor = _np.abs(
                                        self._Session.run(self._FirstWeights[0]))
                                    sparse_weights = _np.sum(
                                        sparse_tensor, axis=1)
                                    fig_weights, weights_plot = _initialize_weights_plot(
                                        sparse_weights, self.SizeOfInputsPerType[0])
                                else:
                                    fig, ax, TrainingCostPlot, ValidationCostPlot, RunningMeanPlot = _initialize_cost_plot(
                                        self.OverallTrainingCosts, self.OverallValidationCosts)
                            else:
                                if find_best_symmfuns:
                                    # only supports force field at the moment
                                    sparse_tensor = _np.abs(
                                        self._Session.run(self._FirstWeights[0]))
                                    sparse_weights = _np.sum(
                                        sparse_tensor, axis=1)
                                    fig_weights, weights_plot = _update_weights_plot(
                                        fig_weights, weights_plot, sparse_weights)
                                else:
                                    _update_cost_plot(
                                        fig,
                                        ax,
                                        TrainingCostPlot,
                                        self.OverallTrainingCosts,
                                        ValidationCostPlot,
                                        self.OverallValidationCosts,
                                        RunningMeanPlot)
                        # Finished percentage output
                        print([str(100 * i / self.Epochs) + " %",
                               "deltaE = " + str(self.DeltaE) + " ev",
                               "Cost = " + str(self.TrainingCosts),
                               "t = " + str(_time.time() - start) + " s",
                               "global step: " + str(self._Session.run(self.GlobalStep))])
                        Prediction = self.eval_dataset_energy(
                            [[self._TrainingInputs]])
                        print("Data:")
                        print(
                            "Ei=" + str(self._TrainingOutputs[0:max(int(len(self._TrainingOutputs) / 20), 1)]))
                        if self.UseForce:
                            Force = self.eval_dataset_force(
                                self.TrainingBatches, rnd)
                            print(
                                "F1_x=" + str(self._ForceTrainingOutput[0:max(int(len(self._TrainingOutputs) / 20), 1), 0]))
                        print("Prediction:")
                        print(
                            "Ei=" + str(Prediction[0:max(int(len(Prediction) / 20), 1)]))
                        if self.UseForce:
                            print(
                                "F1_x=" + str(Force[0:max(int(len(Prediction) / 20), 1), 0]))
                        # Store variables
                        self.TrainedVariables = self._Net.get_trained_variables(
                            self._Session)

                        if not _os.path.exists(self.SavingDirectory):
                            _os.makedirs(self.SavingDirectory)
                        _np.save(self.SavingDirectory + "/trained_variables",
                                 [self.TrainedVariables,
                                  self._MeansOfDs,
                                  self._VarianceOfDs,
                                  self._MinOfOut])

                    # Abort criteria
                    if self.TrainingCosts != 0 and self.TrainingCosts <= self.CostCriterium and self.ValidationCosts <= self.CostCriterium or self.DeltaE < self.dE_Criterium:

                        if self.ValidationCosts != 0:
                            print("Reached criterium!")
                            print(
                                "Cost= " + str((self.TrainingCosts + self.ValidationCosts) / 2))
                            print("delta E = " + str(self.DeltaE) + " ev")
                            print("t = " + str(_time.time() - start) + " s")
                            print("Epoch = " + str(i))
                            print("")

                        else:
                            print("Reached criterium!")
                            print("Cost= " + str(self.TrainingCosts))
                            print("delta E = " + str(self.DeltaE) + " ev")
                            print("t = " + str(_time.time() - start) + " s")
                            print("Epoch = " + str(i))
                            print("")

                        print("Calculation of whole dataset energy difference ...")
                        train_stat, val_stat = self._dE_stat(EnergyLayers)
                        print("Training dataset error= " +
                              str(train_stat[0]) +
                              "+-" +
                              str(_np.sqrt(train_stat[1])) +
                              " ev")
                        print("Validation dataset error= " +
                              str(val_stat[0]) +
                              "+-" +
                              str(_np.sqrt(val_stat[1])) +
                              " ev")
                        print("Training finished")
                        break

                    if i == (self.Epochs - 1):
                        print("Training finished")
                        print("delta E = " + str(self.DeltaE) + " ev")
                        print("t = " + str(_time.time() - start) + " s")
                        print("")

                        train_stat, val_stat = self._dE_stat(EnergyLayers)
                        print("Training dataset error= " +
                              str(train_stat[0]) +
                              "+-" +
                              str(_np.sqrt(train_stat[1])) +
                              " ev")
                        print("Validation dataset error= " +
                              str(val_stat[0]) +
                              "+-" +
                              str(_np.sqrt(val_stat[1])) +
                              " ev")

            if self.Multiple:

                return [self.TrainedVariables, self._MinOfOut]

    def create_symmetry_functions(self):

        self.SizeOfInputsPerType = []
        self._SymmFunSet = _SymmetryFunctionSet.SymmetryFunctionSet(
            self.Atomtypes)
        self._SymmFunSet.add_radial_functions_evenly(
            self.NumberOfRadialFunctions)
        self._SymmFunSet.add_angular_functions(
            self.Etas, self.Zetas, self.Lambs)
        self.SizeOfInputsPerType = self._SymmFunSet.num_Gs

        for i, a_type in enumerate(self.NumberOfAtomsPerType):
            for j in range(0, a_type):
                self.SizeOfInputsPerAtom.append(self.SizeOfInputsPerType[i])

    def _convert_dataset(self, TakeAsReference):
        """Converts the cartesian coordinates to a symmetry function vector and
        calculates the mean value and the variance of the symmetry function
        vector.

        Args:
            TakeAsReference(bool): Specifies if the MinOfOut Parameter should be
                                set according to this dataset.
        """

        print("Converting data to neural net input format...")
        NrGeom = len(self._DataSet.geometries)
        AllTemp = list()
        # Get G vectors

        for i in range(0, NrGeom):
            
            temp = _np.asarray(self._SymmFunSet.eval_geometry(
                self._DataSet.geometries[i]))

            self._AllGeometries.append(temp)
            self._AllGDerivatives.append(
                _np.asarray(
                    self._SymmFunSet.eval_geometry_derivatives(
                        self._DataSet.geometries[i])))
            if i % max(int(NrGeom / 25), 1) == 0:
                print(str(100 * i / NrGeom) + " %")
            for j in range(0, len(temp)):
                if i == 0:
                    AllTemp.append(_np.empty((NrGeom, temp[j].shape[0])))
                    AllTemp[j][i] = temp[j]
                else:
                    AllTemp[j][i] = temp[j]

        if TakeAsReference:
            self._calculate_statistics_for_dataset(AllTemp)

    def _calculate_statistics_for_dataset(self, AllTemp):
        """To be documented..."""

        NrAtoms = sum(self.NumberOfAtomsPerType)
        # calculate mean and sigmas for all Gs
        print("Calculating mean values and variances...")
        # Input statistics
        self._MeansOfDs = [0] * len(self.NumberOfAtomsPerType)
        self._VarianceOfDs = [0] * len(self.NumberOfAtomsPerType)
        ct = 0
        InputsForTypeX = []
        for i, InputsForNetX in enumerate(AllTemp):
            if len(InputsForTypeX) == 0:
                InputsForTypeX = list(InputsForNetX)
            else:
                InputsForTypeX += list(InputsForNetX)
            try:
                if self.NumberOfAtomsPerType[ct] == i + 1:
                    self._MeansOfDs[ct] = _np.mean(InputsForTypeX, axis=0)
                    self._VarianceOfDs[ct] = _np.var(InputsForTypeX, axis=0)
                    InputsForTypeX = []
                    ct += 1
            except:
                raise ValueError(
                        "Number of atoms per type does not match input!")
        # Output statistics
        if len(self._DataSet.energies) > 0:
            NormalizedEnergy = _np.divide(self._DataSet.energies, NrAtoms)
            # factor of two is to make sure that there is room for lower
            # energies
            self._MinOfOut = _np.min(NormalizedEnergy) * 2

    def read_qe_md_files(
            self,
            path,
            energy_unit="eV",
            dist_unit="A",
            TakeAsReference=True,
            LoadGeometries=True):
        """Reads lammps files,adds symmetry functions to the symmetry function
        basis and converts the cartesian corrdinates to symmetry function vectors.

        Args:
            TakeAsReference(bool): Specifies if the MinOfOut Parameter should be
                                set according to this dataset.
            LoadGeometries(bool): Specifies if the conversion of the geometry
                                coordinates should be performed."""

        Execute = True
        if len(self.Atomtypes) == 0:
            print("No atom types specified!")
            Execute = False

        if Execute:
            self._DataSet = _DataSet.DataSet()
            self._Reader = _ReaderQE.QE_MD_Reader()
            if energy_unit == "Ry":
                self._Reader.E_conv_factor = 13.605698066
            elif energy_unit == "H":
                self._Reader.E_conv_factor = 27.211396132
            elif energy_unit == "kcal/mol":
                self._Reader.E_conv_factor = 0.043
            elif energy_unit == "kJ/mol":
                self._Reader.E_conv_factor = 0.01
            else:
                self._Reader.E_conv_factor = 1

            if dist_unit == "Bohr" or dist_unit == "au":
                self._Reader.Geom_conv_factor = 0.529177249
            else:
                self._Reader.Geom_conv_factor = 1

            if len(self.Atomtypes) != 0:
                self._Reader.atom_types = self.Atomtypes
            else:
                self.Atomtypes = self._Reader.atom_types

            self._Reader.get_files(path)
            self._Reader.read_all_files()
            self._Reader.calibrate_energy()

            self.NumberOfAtomsPerType = self._Reader.nr_atoms_per_type
            self._DataSet.geometries = self._Reader.geometries
            self._DataSet.energies = self._Reader.energies
            if self.UseForce:
                self._DataSet.forces = self._Reader.forces

            self.create_symmetry_functions()
            print("Added dataset!")

        if LoadGeometries:
            self._convert_dataset(TakeAsReference)

    def read_lammps_files(
            self,
            XYZFile,
            LogFile,
            energy_unit="eV",
            dist_unit="A",
            TakeAsReference=True,
            LoadGeometries=True):
        """Reads lammps files,adds symmetry functions to the symmetry function
        basis and converts the cartesian corrdinates to symmetry function vectors.

        Args:
            TakeAsReference(bool): Specifies if the MinOfOut Parameter should be
                                set according to this dataset.
            LoadGeometries(bool): Specifies if the conversion of the geometry
                                coordinates should be performed."""

        Execute = True
        if len(self.Atomtypes) == 0:
            print("No atom types specified!")
            Execute = False

        if Execute:

            self._DataSet = _DataSet.DataSet()
            self._Reader = _ReaderLammps.LammpsReader()
            if energy_unit == "Ry":
                self._Reader.E_conv_factor = 13.605698066
            elif energy_unit == "H":
                self._Reader.E_conv_factor = 27.211396132
            elif energy_unit == "kcal/mol":
                self._Reader.E_conv_factor = 0.043
            elif energy_unit == "kJ/mol":
                self._Reader.E_conv_factor = 0.01
            else:
                self._Reader.E_conv_factor = 1

            if dist_unit == "Bohr" or dist_unit == "au":
                self._Reader.Geom_conv_factor = 0.529177249
            else:
                self._Reader.Geom_conv_factor = 1

            if len(self.Atomtypes) != 0:
                self._Reader.atom_types = self.Atomtypes
            else:
                self.Atomtypes = self._Reader.atom_types

            self._Reader.read_lammps(XYZFile, LogFile)
            self._DataSet.geometries = self._Reader.geometries
            self._DataSet.energies = self._Reader.energies
            if self.UseForce:
                self._DataSet.forces = self._Reader.forces

            self.create_symmetry_functions()
            print("Added dataset!")

        if LoadGeometries:
            self._convert_dataset(TakeAsReference)

    def init_dataset(self, geometries, energies,
                     forces=[], TakeAsReference=True):
        """Initializes a loaded dataset.

        Args:
            geometries (list): List of geomtries
            energies (list) : List of energies
            forces (list): List of G-vector derivatives
            TakeAsReference (bool): Specifies if the MinOfOut Parameter
                                    should be set according to this dataset"""

        if len(geometries) == len(energies):
            self._DataSet = _DataSet.DataSet()
            self._DataSet.energies = energies
            self._DataSet.geometries = geometries
            self._DataSet.forces = forces
            if TakeAsReference:
                self._VarianceOfDs = []
                self._MeansOfDs = []
            self.create_symmetry_functions()

            self._convert_dataset(TakeAsReference)
        else:
            print("Number of energies: " +
                  str(len(energies)) +
                  " does not match number of geometries: " +
                  str(len(geometries)))

    def create_eval_data(self, geometries, NoBatches=True):
        """Converts the geometries in compatible format and prepares the data
        for evaluation.

        Args:
            geometries (list): List of geometries
            NoBatches (bool): Specifies if the data is split into differnt
                batches or only consits of a single not randomized batch.
        """
        dummy_energies = [0] * len(geometries)
        if len(self._MeansOfDs) == 0:
            IsReference = True
        else:
            IsReference = False

        self.init_dataset(geometries, dummy_energies,
                          TakeAsReference=IsReference)

        self.EvalData = self.get_data(NoBatches=True)

        return self.EvalData

    def _get_data_batch(self, BatchSize=100, NoBatches=False):
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

        GeomData = []
        GDerivativesInput = []
        ForceData = []

        Execute = True
        if self._SymmFunSet is None:
            print("No symmetry function set available!")
            Execute = False
        if self._DataSet is None:
            print("No data set available!")
            Execute = False
        if len(self._DataSet.geometries) == 0:
            print("No geometries available!")
            Execute = False
        if len(self._DataSet.energies) == 0:
            print("No energies available!")
            Execute = False

        if Execute:
            if NoBatches:
                BatchSize = len(self._DataSet.geometries)

            EnergyData = _np.empty((BatchSize, 1))
            ForceData = _np.empty(
                (BatchSize, sum(self.NumberOfAtomsPerType) * 3))
            if not NoBatches:
                if BatchSize > len(self._AllGeometries) / 10:
                    BatchSize = int(BatchSize / 10)
                    print("Shrunk batches to size:" + str(BatchSize))

            # Create a list with all possible random values
            ValuesForDrawingSamples = list(
                range(0, len(self._DataSet.geometries)))

            for i in range(0, BatchSize):
                # Get a new random number
                if NoBatches:
                    MyNr = i
                else:
                    rnd = _rand.randint(0, len(ValuesForDrawingSamples) - 1)
                    # Get number
                    MyNr = ValuesForDrawingSamples[rnd]
                    # remove number from possible samples
                    ValuesForDrawingSamples.pop(rnd)

                GeomData.append(self._AllGeometries[MyNr])
                EnergyData[i] = self._DataSet.energies[MyNr]
                if len(self._DataSet.forces) > 0:
                    ForceData[i] = [f for atom in self._DataSet.forces[MyNr]
                                    for f in atom]
                if self.UseForce:
                    GDerivativesInput.append(self._AllGDerivatives[MyNr])

            Inputs, GDerivativesInput, Normalization = self._sort_and_normalize_data(
                BatchSize, GeomData, GDerivativesInput)

            if self.UseForce:
                return Inputs, EnergyData, GDerivativesInput, Normalization, ForceData
            else:
                return Inputs, EnergyData

    def get_data(self, BatchSize=100, CoverageOfSetInPercent=70,
                 NoBatches=False):
        """Creates a batch collection.

        Args:
            CoverageOfSetInPercent(int):discribes how many data points are
                included in the batch (value from 0-100).
            NoBatches (bool): Specifies if the data is split into differnt
                batches or only consits of a single not randomized batch.

        Returns:
            Batches: List of numpy arrays"""

        Batches = []
        Execute = True
        if self._SymmFunSet is None:
            print("No symmetry function set available!")
            Execute = False
        if self._DataSet is None:
            print("No data set available!")
            Execute = False
        if len(self._DataSet.geometries) == 0:
            print("No geometries available!")
            Execute = False
        if len(self._DataSet.energies) == 0:
            print("No energies available!")
            Execute = False

        if Execute:

            EnergyDataSetLength = len(self._DataSet.geometries)
            SetLength = int(EnergyDataSetLength * CoverageOfSetInPercent / 100)

            if not NoBatches:
                if BatchSize > len(self._AllGeometries) / 10:
                    BatchSize = int(BatchSize / 10)
                    print("Shrunk batches to size:" + str(BatchSize))
                NrOfBatches = max(1, int(round(SetLength / BatchSize, 0)))
            else:
                NrOfBatches = 1
            print("Creating and normalizing " +
                  str(NrOfBatches) + " batches...")
            for i in range(0, NrOfBatches):
                Batches.append(self._get_data_batch(BatchSize, NoBatches))
                if not NoBatches:
                    if i % max(int(NrOfBatches / 10), 1) == 0:
                        print(str(100 * i / NrOfBatches) + " %")

            return Batches

    def make_training_and_validation_data(
            self,
            BatchSize=100,
            TrainingSetInPercent=70,
            ValidationSetInPercent=30,
            NoBatches=False):
        """Creates training and validation data.

        Args:

            BatchSize (int): Specifies the number of data points per batch.
            TrainingSetInPercent (int): Discribes the coverage of the training dataset.(value from 0-100)
            TrainingSetInPercent (int): Discribes the coverage of the validation dataset.(value from 0-100)
            NoBatches (bool): Specifies if the data is split into differnt batches or only """

        if not NoBatches:
            # Get training data
            self.TrainingBatches = self.get_data(
                BatchSize, TrainingSetInPercent, NoBatches)
            # Get validation data
            self.ValidationBatches = self.get_data(
                BatchSize, ValidationSetInPercent, NoBatches)
        else:
            # Get training data
            temp = self.get_data(BatchSize, TrainingSetInPercent, NoBatches)
            self._TrainingInputs = temp[0][0]
            self._TrainingOutputs = temp[0][1]
            # Get validation data
            temp = self.get_data(BatchSize, ValidationSetInPercent, NoBatches)
            self._ValidationInputs = temp[0][0]
            self._ValidationOutputs = temp[0][0]

    def _atomic_cost_function(self):
        """The atomic cost function consists of multiple parts which are each
        represented by a tensor.
        The main part is the energy cost.
        The reqularization and the force cost is optional.

        Returns:
            A tensor which is the sum of all costs"""

        self._TotalEnergy, AllEnergies = self._Net.energy_of_all_atomic_networks()

        self._EnergyCost = cost_for_network(
            self._TotalEnergy, self._OutputLayer, self.CostFunType)
        Cost = self._EnergyCost

        # add force cost
        if self.UseForce:
            if self.IsPartitioned:
                raise(NotImplementedError)
            else:
                self._OutputForce, AllForces = self._Net.force_of_all_atomic_networks(
                    self)
                self._ForceCost = self.ForceCostParam * _tf.divide(
                    cost_for_network(
                        self._OutputForce, self._OutputLayerForce, self.CostFunType), sum(
                        self.NumberOfAtomsPerType))
            Cost += self._ForceCost

        if self.Regularization == "L1":
            trainableVars = _tf.trainable_variables()
            l1_regularizer = _tf.contrib.layers.l1_regularizer(
                scale=0.005, scope=None)
            Cost += _tf.contrib.layers.apply_regularization(
                l1_regularizer, trainableVars)
        elif self.Regularization == "L2":
            trainableVars = _tf.trainable_variables()
            self._RegLoss = _tf.add_n([_tf.nn.l2_loss(v) for v in trainableVars
                                       if 'bias' not in v.name]) * self.RegularizationParam
            Cost += self._RegLoss

        # Create tensor for energy difference calculation
        self._dE_Fun = _tf.abs(self._TotalEnergy - self._OutputLayer)

        return Cost

    def _sort_and_normalize_data(self, BatchSize, GData, GDerivativesData=[]):
        """Normalizes the input data.

        Args:
            BatchSize (int): Specifies the number of data points per batch.
            GeomData (list): Raw geometry data
            ForceData (list): (Optional) Raw derivatives of input vector

        Returns:
            Inputs: Normalized inputs
            DerInputs: If GDerivativesData is available a list of numpy array is returned
                    ,else an empty list is returned."""

        Inputs = []
        DerInputs = []
        Norm = []
        ct = 0
        for VarianceOfDs, MeanOfDs, NrAtoms in zip(
                self._VarianceOfDs, self._MeansOfDs, self.NumberOfAtomsPerType):
            for i in range(NrAtoms):
                Inputs.append(
                    _np.zeros((BatchSize, self.SizeOfInputsPerAtom[ct])))
                if len(GDerivativesData) > 0:
                    DerInputs.append(_np.zeros(
                        (BatchSize, self.SizeOfInputsPerAtom[ct], 3 * sum(self.NumberOfAtomsPerType))))
                    Norm.append(
                        _np.zeros((BatchSize, self.SizeOfInputsPerAtom[ct])))
                # exclude nan values
                L = _np.nonzero(VarianceOfDs)

                if L[0].size > 0:
                    for j in range(0, len(GData)):
                        temp = _np.divide(
                            _np.subtract(
                                GData[j][ct][L], MeanOfDs[L]), _np.sqrt(
                                VarianceOfDs[L]))
                        temp[_np.isinf(temp)] = 0
                        temp[_np.isneginf(temp)] = 0
                        temp[_np.isnan(temp)] = 0
                        Inputs[ct][j][L] = temp
                        if len(GDerivativesData) > 0:
                            DerInputs[ct][j] = GDerivativesData[j][ct]
                            Norm[ct] = _np.tile(_np.divide(
                                1, _np.sqrt(VarianceOfDs)), (BatchSize, 1))

                ct += 1

        return Inputs, DerInputs, Norm

    def _init_correction_network_data(
            self, network, geoms, energies, N_zero_geoms=10000):
        """Creates data outside the area covered by the dataset and adds them
        to the training data.

        Args:
            network (Tensor): Pre trained network
            geoms (list): List of geometries
            energies (list): List of energies
            N_zero_geom (int): Number of geometries to create
        """
        # get coverage of dataset
        ds_r_min, ds_r_max = _get_ds_r_min_r_max(geoms)
        # create geometries outside of dataset

        zero_ds_geoms = _create_zero_diff_geometries(
            ds_r_min, ds_r_max, self.Atomtypes, self.NumberOfAtomsPerType, N_zero_geoms)
        all_geoms = geoms + zero_ds_geoms
        # eval geometries with FF network to make energy difference zero
        network.create_eval_data(geoms)
        zero_ds_energies = [energy for element in network.eval_dataset_energy(
            network.EvalData) for energy in element]

        all_energies = energies + zero_ds_energies
        self.init_dataset(all_geoms, all_energies)


class _StandardAtomicNetwork(object):

    def __init__(self):
        self.AtomicNNs = []
        self.VariablesDictionary = {}

    def get_structure_from_data(self, TrainedData):
        """Calculates the structure based on the stored varibales

        Args:
            TrainedData (list): Trained network data
        Returns:
            Structues (list):A list of structures
            (one structure per atom species)"""

        Structures = list()
        for i in range(0, len(TrainedData)):
            ThisNetwork = TrainedData[i]
            Structure = []
            for j in range(0, len(ThisNetwork)):
                Weights = ThisNetwork[j][0]
                Structure += [Weights.shape[0]]
            Structure += [1]
            Structures.append(Structure)

        return Structures

    def force_of_all_atomic_networks(self, NetInstance):
        """This function constructs the force expression for the atomic networks.

        Returns:
            A tensor which represents the force output of the network"""

        F = []
        Fi = []
        for i in range(0, len(self.AtomicNNs)):
            AtomicNet = self.AtomicNNs[i]
            Type = AtomicNet[0]
            G_Input = AtomicNet[2]
            dGij_dxk = AtomicNet[3]
            norm = AtomicNet[4]
            dGij_dxk_t = _tf.transpose(dGij_dxk, perm=[0, 2, 1])
            Gradient = _tf.gradients(NetInstance._TotalEnergy, G_Input)
            # nan-workaround(may be fixed in later tensorflow versions)
            # finite_values=_tf.is_finite(temp)
            #temp = _tf.boolean_mask(temp,finite_values)
            #norm= _tf.boolean_mask(norm,finite_values)
            dEi_dGij_n = _tf.multiply(Gradient, norm)
            # idx=_tf.to_int32(_tf.where(finite_values))
            # dEi_dGij_n=_tf.scatter_nd(idx,dEi_dGij_n,_tf.shape(finite_values))
            dEi_dGij_n = _tf.where(
                _tf.logical_or(
                    _tf.is_inf(dEi_dGij_n),
                    _tf.is_nan(dEi_dGij_n)),
                _tf.zeros_like(dEi_dGij_n),
                dEi_dGij_n)
            dEi_dGij = _tf.reshape(
                dEi_dGij_n, [-1, NetInstance.SizeOfInputsPerType[Type], 1])
            mul = _tf.matmul(dGij_dxk_t, dEi_dGij)
            dim_red = _tf.reshape(
                mul, [-1, sum(NetInstance.NumberOfAtomsPerType) * 3])

            # no_nan=_tf.logical_and(_tf.logical_not(_tf.is_nan(dim_red)),_tf.is_finite(dim_red))
            #F_no_nan= _tf.boolean_mask(temp,finite_values)
            # idx=_tf.to_int32(_tf.where(no_nan))
            # F_no_nan=_tf.scatter_nd(idx,dim_red,_tf.shape(no_nan))
            # F_no_nan=_tf.where(_tf.is_inf(dim_red),_tf.zeros_like(dim_red),dim_red)
            # F_no_nan=_tf.where(_tf.is_nan(F_no_nan),_tf.zeros_like(F_no_nan),F_no_nan)
            if i == 0:
                F = dim_red
            else:
                F = _tf.add(F, dim_red)
            Fi.append(dim_red)

        return F, Fi

    def energy_of_all_atomic_networks(self):
        """This function constructs the energy expression for
        the atomic networks.

        Returns:
            Prediction: A tensor which represents the energy output of
                        the partitioned network.
            AllEnergies: A list of tensors which represent the single Network
                        energy contributions."""

        Prediction = 0
        AllEnergies = list()

        for i in range(0, len(self.AtomicNNs)):
            # Get network data
            AtomicNetwork = self.AtomicNNs[i]
            Network = AtomicNetwork[1]
            E_no_nan = _tf.where(_tf.is_nan(Network),
                                 _tf.zeros_like(Network), Network)
            # Get input data for network
            AllEnergies.append(E_no_nan)

        Prediction = _tf.add_n(AllEnergies)

        return Prediction, AllEnergies

    def get_trained_variables(self, Session):
        """Prepares the data for saving.
        It gets the weights and biases from the session.

        Returns:
            NetworkData (list): All the network parameters as a list"""

        NetworkData = list()
        for HiddenLayers in self.VariablesDictionary:
            Layers = list()
            for i in range(0, len(HiddenLayers)):
                Weights = Session.run(HiddenLayers[i][0])
                Biases = Session.run(HiddenLayers[i][1])
                Layers.append([Weights, Biases])
            NetworkData.append(Layers)
        return NetworkData

    def make_layers_for_atomicNNs(
            self, OutputLayer=None, OutputLayerForce=None, AppendForce=True):
        """Sorts the input placeholders in the correct order for feeding.
        Each atom has a seperate placeholder which must be feed at each step.
        The placeholders have to match the symmetry function input.
        For training the output placeholder also has to be feed.

        Returns:
            Layers (list):All placeholders as a list."""
        # Create list of placeholders for feeding in correct order
        Layers = []
        ForceLayer = False
        for AtomicNetwork in self.AtomicNNs:
            Layers.append(AtomicNetwork[2])
            if len(AtomicNetwork) > 3 and AppendForce:
                Layers.append(AtomicNetwork[3])  # G-derivatives
                Layers.append(AtomicNetwork[4])  # Normalization
                ForceLayer = True
        if OutputLayer is not None:
            Layers.append(OutputLayer)
            if ForceLayer and AppendForce:
                Layers.append(OutputLayerForce)

        return Layers

    def make_data_for_atomicNNs(self, GData, OutData=[], GDerivatives=[
    ], ForceOutput=[], Normalization=[], AppendForce=True):
        """Sorts the symmetry function data for feeding.
            For training the output data also has to be added.
        Returns:
            CombinedData(list): Sorted data for the batch as a list."""
        # Sort data matching the placeholders
        CombinedData = []
        if AppendForce:
            for e, f, n in zip(GData, GDerivatives, Normalization):
                CombinedData.append(e)
                CombinedData.append(f)
                CombinedData.append(n)
        else:
            for Data in GData:
                CombinedData.append(Data)
        if len(OutData) != 0:
            CombinedData.append(OutData)
            if AppendForce:
                CombinedData.append(ForceOutput)

        return CombinedData

    def prepare_data_environment(
            self,
            GData,
            OutputLayer=None,
            OutData=[],
            OutputLayerForce=None,
            GDerivatives=[],
            ForceOutput=[],
            Normalization=[],
            AppendForce=True):
        """Prepares the data and the input placeholders for the training in a NN.
        Returns:
            Layers (list):List of sorted placeholders
            Data (list): List of sorted data"""

        # Put data and placeholders in correct order for feeding
        Layers = self.make_layers_for_atomicNNs(
            OutputLayer, OutputLayerForce, AppendForce)

        Data = self.make_data_for_atomicNNs(
            GData,
            OutData,
            GDerivatives,
            ForceOutput,
            Normalization,
            AppendForce)

        return Layers, Data

    def evaluate_all_atomicnns(self, GData, AppendForce=False):
        """Evaluates the networks and calculates the energy as a sum of all network
        outputs.

        Returns:
            Energy(float): Predicted energy as a float."""
        Energy = 0
        Layers, Data = self.prepare_data_environment(GData, OutputLayer=None, OutData=[
        ], OutputLayerForce=None, GDerivativesInput=[], ForceOutput=[], AppendForce=False)
        for i in range(0, len(self.AtomicNNs)):
            AtomicNetwork = self.AtomicNNs[i]
            Energy += self.evaluate(AtomicNetwork[1], [Layers[i]], Data[i])

        return Energy

    def get_weights_biases_from_data(self, TrainedVariables, Multi=False):
        """Reads out the saved network data and sorts them into
        weights and biases.

        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""

        Weights = list()
        Biases = list()
        for i in range(0, len(TrainedVariables)):
            NetworkData = TrainedVariables[i]
            ThisWeights = list()
            ThisBiases = list()
            for j in range(0, len(NetworkData)):
                ThisWeights.append(NetworkData[j][0])
                ThisBiases.append(NetworkData[j][1])
            Weights.append(ThisWeights)
            Biases.append(ThisBiases)

        return Weights, Biases

    def make_parallel_atomic_networks(self, NetInstance):
        """Creates the specified network with separate varibale tensors
            for each atoms.(Only for evaluation)"""

        AtomicNNs = list()
        # Start Session
        NetInstance._Session = _tf.Session()
        # make all layers
        if len(NetInstance.Structures) != len(self.NumberOfAtomsPerType):
            raise ValueError(
                "Length of Structures does not match length of NumberOfAtomsPerType")
        if len(NetInstance._WeightData) != 0:
            for i in range(0, len(NetInstance.Structures)):
                if len(NetInstance.Dropout) > i:
                    Dropout = NetInstance.Dropout[i]
                else:
                    Dropout = NetInstance.Dropout[-1]

                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):

                    # Make hidden layers
                    HiddenLayers = list()
                    Structure = NetInstance.Structures[i]

                    RawWeights = NetInstance._WeightData[i]
                    RawBias = NetInstance._BiasData[i]

                    for j in range(1, len(Structure)):
                        NrIn = Structure[j - 1]
                        NrHidden = Structure[j]
                        # fill old weights in new structure
                        ThisWeightData = RawWeights[j - 1]
                        ThisBiasData = RawBias[j - 1]

                        HiddenLayers.append(
                            _construct_hidden_layer(
                                NrIn,
                                NrHidden,
                                NetInstance.WeightType,
                                ThisWeightData,
                                NetInstance.BiasType,
                                ThisBiasData,
                                NetInstance.MakeAllVariable))

                    # Make input layer
                    if len(NetInstance._WeightData) != 0:
                        NrInputs = NetInstance._WeightData[i][0].shape[0]
                    else:
                        NrInputs = Structure[0]

                    InputLayer = _construct_input_layer(NrInputs)
                    # Connect input to first hidden layer
                    FirstWeights = HiddenLayers[0][0]
                    FirstBiases = HiddenLayers[0][1]
                    Network = _connect_layers(
                        InputLayer,
                        FirstWeights,
                        FirstBiases,
                        NetInstance.ActFun,
                        NetInstance.ActFunParam,
                        Dropout)

                    for l in range(1, len(HiddenLayers)):
                        # Connect ouput of in layer to second hidden layer

                        if l == len(HiddenLayers) - 1:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(
                                Network, Weights, Biases, "none", NetInstance.ActFunParam, Dropout)
                        else:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(
                                Network, Weights, Biases, NetInstance.ActFun, NetInstance.ActFunParam, Dropout)

                    if NetInstance.UseForce:
                        InputForce = _tf.placeholder(
                            _tf.float32, shape=[None, NrInputs, 3 * sum(NetInstance.NumberOfAtomsPerType)])
                        Normalization = _tf.placeholder(
                            _tf.float32, shape=[None, NrInputs])
                        AtomicNNs.append(
                            [i, Network, InputLayer, InputForce, Normalization])
                    else:
                        AtomicNNs.append(
                            [i, Network, InputLayer])
        else:
            print("No network data found!")

        self.AtomicNNs = AtomicNNs

    def make_atomic_networks(self, NetInstance):
        """Creates the specified network."""

        AllHiddenLayers = list()
        AtomicNNs = list()
        # Start Session
        if not NetInstance.Multiple:
            NetInstance._Session = _tf.Session(config=_tf.ConfigProto(
                intra_op_parallelism_threads=_multiprocessing.cpu_count()))

        OldBiasNr = 0
        OldShape = None
        if len(NetInstance.Structures) != len(
                NetInstance.NumberOfAtomsPerType):
            raise ValueError(
                "Length of Structures does not match length of NumberOfAtomsPerType")
        if isinstance(NetInstance.Structures[0], _PartitionedStructure):
            raise ValueError("Please set IsPartitioned = True !")
        else:
            # create layers for the different atom types
            for i in range(0, len(NetInstance.Structures)):
                if len(NetInstance.Dropout) > i:
                    Dropout = NetInstance.Dropout[i]
                else:
                    Dropout = NetInstance.Dropout[-1]

                # Make hidden layers
                HiddenLayers = list()
                Structure = NetInstance.Structures[i]
                if len(NetInstance._WeightData) != 0:

                    RawBias = NetInstance._BiasData[i]

                    for j in range(1, len(Structure)):
                        NrIn = Structure[j - 1]
                        NrHidden = Structure[j]

                        if j == len(Structure) - \
                                1 and NetInstance.MakeLastLayerConstant:
                            HiddenLayers.append(_construct_not_trainable_layer(
                                NrIn, NrHidden, NetInstance._MinOfOut))
                        else:
                            if j >= len(
                                    NetInstance._WeightData[i]) and NetInstance.MakeLastLayerConstant:
                                tempWeights, tempBias = _construct_hidden_layer(NrIn, NrHidden, NetInstance.WeightType, [], NetInstance.BiasType, [],
                                                                                True, NetInstance.InitMean, NetInstance.InitStddev)

                                indices = []
                                values = []
                                thisShape = tempWeights.get_shape().as_list()
                                if thisShape[0] == thisShape[1]:
                                    for q in range(0, OldBiasNr):
                                        indices.append([q, q])
                                        values += [1.0]

                                    delta = _tf.SparseTensor(
                                        indices, values, thisShape)
                                    tempWeights = tempWeights + \
                                        _tf.sparse_tensor_to_dense(delta)

                                HiddenLayers.append([tempWeights, tempBias])
                            else:
                                if len(RawBias) >= j:
                                    OldBiasNr = len(
                                        NetInstance._BiasData[i][j - 1])
                                    OldShape = NetInstance._WeightData[i][j - 1].shape
                                    # fill old weights in new structure
                                    if OldBiasNr < NrHidden:
                                        ThisWeightData = _np.random.normal(
                                            loc=0.0, scale=0.01, size=(NrIn, NrHidden))
                                        ThisWeightData[0:OldShape[0], 0:OldShape[1]
                                                       ] = NetInstance._WeightData[i][j - 1]
                                        ThisBiasData = _np.zeros([NrHidden])
                                        ThisBiasData[0:OldBiasNr] = NetInstance._BiasData[i][j - 1]
                                    elif OldBiasNr > NrHidden:
                                        ThisWeightData = _np.zeros(
                                            (NrIn, NrHidden))
                                        ThisWeightData[0:,
                                                       0:] = NetInstance._WeightData[i][j - 1][0:NrIn,
                                                                                               0:NrHidden]
                                        ThisBiasData = _np.zeros([NrHidden])
                                        ThisBiasData[0:OldBiasNr] = NetInstance._BiasData[i][j -
                                                                                             1][0:NrIn, 0:NrHidden]
                                    else:
                                        ThisWeightData = NetInstance._WeightData[i][j - 1]
                                        ThisBiasData = NetInstance._BiasData[i][j - 1]

                                    HiddenLayers.append(
                                        _construct_hidden_layer(
                                            NrIn,
                                            NrHidden,
                                            NetInstance.WeightType,
                                            ThisWeightData,
                                            NetInstance.BiasType,
                                            ThisBiasData,
                                            NetInstance.MakeAllVariable))
                                else:
                                    raise ValueError("Number of layers doesn't match[" + str(len(RawBias)) + str(
                                        len(Structure)) + "], MakeLastLayerConstant has to be set to True!")

                else:
                    for j in range(1, len(Structure)):
                        NrIn = Structure[j - 1]
                        NrHidden = Structure[j]
                        if j == len(Structure) - \
                                1 and NetInstance.MakeLastLayerConstant:
                            HiddenLayers.append(_construct_not_trainable_layer(
                                NrIn, NrHidden, NetInstance._MinOfOut))
                        else:
                            HiddenLayers.append(
                                _construct_hidden_layer(
                                    NrIn,
                                    NrHidden,
                                    NetInstance.WeightType,
                                    [],
                                    NetInstance.BiasType))

                AllHiddenLayers.append(HiddenLayers)
                # create network for each atom
                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):
                    # Make input layer
                    if len(NetInstance._WeightData) != 0:
                        NrInputs = NetInstance._WeightData[i][0].shape[0]
                    else:
                        NrInputs = Structure[0]

                    InputLayer = _construct_input_layer(NrInputs)
                    # Connect input to first hidden layer
                    FirstWeights = HiddenLayers[0][0]
                    NetInstance._FirstWeights.append(FirstWeights)
                    FirstBiases = HiddenLayers[0][1]
                    Network = _connect_layers(
                        InputLayer,
                        FirstWeights,
                        FirstBiases,
                        NetInstance.ActFun,
                        NetInstance.ActFunParam,
                        Dropout)

                    for l in range(1, len(HiddenLayers)):
                        # Connect layers

                        if l == len(HiddenLayers) - 1:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(
                                Network, Weights, Biases, "none", NetInstance.ActFunParam, Dropout)
                        else:
                            Weights = HiddenLayers[l][0]
                            Biases = HiddenLayers[l][1]
                            Network = _connect_layers(
                                Network, Weights, Biases, NetInstance.ActFun, NetInstance.ActFunParam, Dropout)

                    if NetInstance.UseForce:
                        InputForce = _tf.placeholder(
                            _tf.float32, shape=[None, NrInputs, 3 * sum(NetInstance.NumberOfAtomsPerType)])
                        Normalization = _tf.placeholder(
                            _tf.float32, shape=[None, NrInputs])
                        AtomicNNs.append(
                            [i, Network, InputLayer, InputForce, Normalization])
                    else:
                        AtomicNNs.append([i, Network, InputLayer])

            self.AtomicNNs = AtomicNNs
            self.VariablesDictionary = AllHiddenLayers


class _PartitionedAtomicNetwork(object):

    def __init__(self):
        self.AtomicNNs = []
        self.VariablesDictionary = {}

    def get_structure_from_data(self, TrainedData):
        raise(NotImplemented)

    def force_of_all_atomic_networks(self, NetInstance):
        raise(NotImplemented)

    def energy_of_all_atomic_networks(self):
        """This function constructs the energy expression for
        the partitioned atomic networks.

        Returns:
            Prediction: A tensor which represents the energy output of
                        the partitioned network.
            AllEnergies: A list of tensors which represent the single Network
                        energy contributions."""

        Prediction = 0
        AllEnergies = list()
        for i in range(0, len(self.AtomicNNs)):
            # Get network data
            AtomicNetwork = self.AtomicNNs[i]
            Networks = AtomicNetwork[1]
            for j in range(0, 2):
                SubNet = Networks[j]
                if SubNet != j:
                    # Get input data for network
                    AllEnergies.append(SubNet)

        Prediction = _tf.add_n(AllEnergies)

        return Prediction, AllEnergies

    def get_trained_variables(self, Session):
        """Prepares the data for saving.
        It gets the weights and biases from the session.

        Returns:
            NetworkData (list): All the network parameters as a list"""

        NetworkData = list()

        for Network in self.VariablesDictionary:
            NetLayers = []
            for i in range(0, len(Network)):
                SubNetLayers = []
                if Network[i] != i:  # if SNetwork[i]==i means no net data
                    SubNet = Network[i]
                    for j in range(0, len(SubNet)):
                        Weights = Session.run(SubNet[j][0])
                        Biases = Session.run(SubNet[j][1])
                        SubNetLayers.append([Weights, Biases])

                NetLayers.append(SubNetLayers)

            NetworkData.append(NetLayers)

        return NetworkData

    def make_layers_for_atomicNNs(
            self, OutputLayer=[], OutputLayerForce=None, AppendForce=True):
        """Sorts the input placeholders in the correct order for feeding.
        Each atom has a seperate placeholder which must be feed at each step.
        The placeholders have to match the symmetry function input.
        For training the output placeholder also has to be feed.

        Returns:
            Layers (list):All placeholders as a list."""

        Layers = []
        for i in range(0, len(self.AtomicNNs)):
            Layer_parts = self.AtomicNNs[i][2]
            # Append layers for each part
            for j in range(0, 2):
                if Layer_parts[j] != j:  # Layer_parts is range(2) if empty
                    Layers.append(Layer_parts[j])
        if not(isinstance(OutputLayer, (list, tuple))):
            Layers.append(OutputLayer)

        return Layers

    def make_data_for_atomicNNs(self, GData, OutData=[], GDerivatives=[
    ], ForceOutput=[], Normalization=[], AppendForce=True):
        """Sorts the symmetry function data for feeding.
            For training the output data also has to be added.
        Returns:
            CombinedData(list): Sorted data for the batch as a list."""
        CombinedData = []
        for i in range(0, len(self.AtomicNNs)):
            Data = GData[i]
            # Append layers for each part
            for j in range(0, 2):
                if j == 0:  # force field data
                    CombinedData.append(Data)
                else:
                    CombinedData.append(Data)

        if len(OutData) != 0:
            CombinedData.append(OutData)

        return CombinedData

    def prepare_data_environment(
            self,
            GData,
            OutputLayer=None,
            OutData=[],
            OutputLayerForce=None,
            GDerivatives=[],
            ForceOutput=[],
            Normalization=[],
            AppendForce=True):
        """Prepares the data and the input placeholders for the training in a
        partitioned NN.
        Returns:
            Layers (list):Sorted placeholders
            CombinedData (list):Sorted data as lists"""
        Layers = self.make_layers_for_atomicNNs(self, OutputLayer)
        CombinedData = self.make_data_for_atomicNNs(self, GData, OutData)

        return Layers, CombinedData

    def evaluate_all_atomicnns(self, GData):
        """Evaluates the partitioned networks and calculates the energy as a
        sum of all network outputs.
        
        Returns:
            Energy (float):The predicted energy as a float."""
        Energy = 0
        Layers, Data = self.prepare_data_environment(GData, list(), list())
        ct = 0
        for i in range(0, len(self.AtomicNNs)):
            AllAtomicNetworks = self.AtomicNNs[i][1]
            for j in range(0, 2):
                SubNet = AllAtomicNetworks[j]
                if SubNet != j:
                    Energy += self.evaluate(SubNet, [Layers[ct]], Data[ct])
                    ct = ct + 1

        return Energy

    def get_weights_biases_from_data(self, TrainedVariables, Multi=False):
        """Reads out the saved network data for partitioned nets and sorts them
        into weights and biases

        Returns:
            Weights (list) List of numpy arrays
            Biases (list) List of numpy arrays"""

        Weights = list()
        Biases = list()
        for i in range(0, len(TrainedVariables)):
            NetworkData = TrainedVariables[i]
            ThisWeights = _PartitionedNetworkData()
            ThisBiases = _PartitionedNetworkData()
            for j in range(0, len(NetworkData)):
                SubNetData = NetworkData[j]
                for k in range(0, len(SubNetData)):
                    if j == 0:
                        ThisWeights.ForceFieldNetworkData.append(
                            SubNetData[k][0])
                        if not Multi:
                            ThisWeights.ForceFieldVariable = False
                        else:
                            ThisWeights.ForceFieldVariable = True

                        ThisBiases.ForceFieldNetworkData.append(
                            SubNetData[k][1])
                    elif j == 1:
                        ThisWeights.CorrectionNetworkData.append(
                            SubNetData[k][0])
                        if not Multi:
                            ThisWeights.CorretionVariable = False
                        else:
                            ThisWeights.CorretionVariable = True

                        ThisBiases.CorrectionNetworkData.append(
                            SubNetData[k][1])
            Weights.append(ThisWeights)
            Biases.append(ThisBiases)

        return Weights, Biases

    def make_parallel_atomic_networks(self, NetInstance):
        """Creates the specified partitioned network with separate varibale
        tensors for each atoms.(Only for evaluation)"""

        AtomicNNs = list()
        # Start Session
        NetInstance._Session = _tf.Session()
        if len(NetInstance.Structures) != len(
                NetInstance.NumberOfAtomsPerType):
            raise ValueError(
                "Length of Structures does not match length of\
                NumberOfAtomsPerType")
        if len(NetInstance._WeightData) != 0:
            # make all the networks for the different atom types
            for i in range(0, len(NetInstance.Structures)):
                if len(NetInstance.Dropout) > i:
                    Dropout = NetInstance.Dropout[i]
                else:
                    Dropout = NetInstance.Dropout[-1]

                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):

                    ForceFieldHiddenLayers = list()
                    CorrectionHiddenLayers = list()

                    ForceFieldWeights = list()
                    CorrectionWeights = list()

                    ForceFieldBias = list()
                    CorrectionBias = list()

                    ForceFieldNetwork = None
                    CorrectionNetwork = None

                    ForceFieldInputLayer = None
                    CorrectionInputLayer = None

                    # Read structures for specific network parts
                    StructureForAtom = NetInstance.Structures[i]
                    ForceFieldStructure = StructureForAtom.ForceFieldNetworkStructure
                    CorrectionStructure = StructureForAtom.CorrectionNetworkStructure
                    # Construct networks out of loaded data

                    # Load data for atom
                    WeightData = NetInstance._WeightData[i]
                    BiasData = NetInstance._BiasData[i]
                    if len(WeightData.ForceFieldNetworkData) > 0:

                        # Recreate force field network
                        ForceFieldWeights = WeightData.ForceFieldNetworkData
                        ForceFieldBias = BiasData.ForceFieldNetworkData

                        for j in range(1, len(ForceFieldStructure)):

                            ThisWeightData = ForceFieldWeights[j - 1]
                            ThisBiasData = ForceFieldBias[j - 1]
                            ForceFieldNrIn = ThisWeightData.shape[0]
                            ForceFieldNrHidden = ThisWeightData.shape[1]
                            ForceFieldHiddenLayers.append(
                                _construct_hidden_layer(
                                    ForceFieldNrIn,
                                    ForceFieldNrHidden,
                                    NetInstance.WeightType,
                                    ThisWeightData,
                                    NetInstance.BiasType,
                                    ThisBiasData,
                                    WeightData.ForceFieldVariable,
                                    NetInstance.InitMean,
                                    NetInstance.InitStddev))

                    if len(WeightData.CorrectionNetworkData) > 0:
                        # Recreate correction network
                        CorrectionWeights = WeightData.CorrectionNetworkData
                        CorrectionBias = BiasData.CorrectionNetworkData

                        for j in range(1, len(CorrectionStructure)):

                            ThisWeightData = CorrectionWeights[j - 1]
                            ThisBiasData = CorrectionBias[j - 1]
                            CorrectionNrIn = ThisWeightData.shape[0]
                            CorrectionNrHidden = ThisWeightData.shape[1]
                            CorrectionHiddenLayers.append(
                                _construct_hidden_layer(
                                    CorrectionNrIn,
                                    CorrectionNrHidden,
                                    NetInstance.WeightType,
                                    ThisWeightData,
                                    NetInstance.BiasType,
                                    ThisBiasData,
                                    WeightData.CorrectionVariable,
                                    NetInstance.InitMean,
                                    NetInstance.InitStddev))

                    if len(ForceFieldHiddenLayers) > 0:
                        # Make force field input layer
                        ForceFieldWeightData = NetInstance._WeightData[i].ForceFieldNetworkData
                        ForceFieldNrInputs = ForceFieldWeightData[0].shape[0]

                        ForceFieldInputLayer = _construct_input_layer(
                            ForceFieldNrInputs)
                        # Connect force field input to first hidden layer
                        ForceFieldFirstWeights = ForceFieldHiddenLayers[0][0]
                        ForceFieldFirstBiases = ForceFieldHiddenLayers[0][1]
                        ForceFieldNetwork = _connect_layers(
                            ForceFieldInputLayer,
                            ForceFieldFirstWeights,
                            ForceFieldFirstBiases,
                            NetInstance.ActFun,
                            NetInstance.ActFunParam,
                            Dropout)
                        # Connect force field hidden layers
                        for l in range(1, len(ForceFieldHiddenLayers)):
                            ForceFieldTempWeights = ForceFieldHiddenLayers[l][0]
                            ForceFieldTempBiases = ForceFieldHiddenLayers[l][1]
                            if l == len(ForceFieldHiddenLayers) - 1:
                                ForceFieldNetwork = _connect_layers(
                                    ForceFieldNetwork,
                                    ForceFieldTempWeights,
                                    ForceFieldTempBiases,
                                    "none",
                                    NetInstance.ActFunParam,
                                    Dropout)
                            else:
                                ForceFieldNetwork = _connect_layers(
                                    ForceFieldNetwork,
                                    ForceFieldTempWeights,
                                    ForceFieldTempBiases,
                                    NetInstance.ActFun,
                                    NetInstance.ActFunParam,
                                    Dropout)

                    if len(CorrectionHiddenLayers) > 0:
                        # Make correction input layer
                        CorrectionWeightData = NetInstance._WeightData[i].CorrectionNetworkData
                        CorrectionNrInputs = CorrectionWeightData[0].shape[0]

                        CorrectionInputLayer = _construct_input_layer(
                            CorrectionNrInputs)
                        # Connect Correction input to first hidden layer
                        CorrectionFirstWeights = CorrectionHiddenLayers[0][0]
                        CorrectionFirstBiases = CorrectionHiddenLayers[0][1]
                        CorrectionNetwork = _connect_layers(
                            CorrectionInputLayer,
                            CorrectionFirstWeights,
                            CorrectionFirstBiases,
                            NetInstance.ActFun,
                            NetInstance.ActFunParam,
                            Dropout)
                        # Connect Correction hidden layers
                        for l in range(1, len(CorrectionHiddenLayers)):
                            CorrectionTempWeights = CorrectionHiddenLayers[l][0]
                            CorrectionTempBiases = CorrectionHiddenLayers[l][1]
                            if l == len(CorrectionHiddenLayers) - 1:
                                CorrectionNetwork = _connect_layers(
                                    CorrectionNetwork,
                                    CorrectionTempWeights,
                                    CorrectionTempBiases,
                                    "none",
                                    NetInstance.ActFunParam,
                                    Dropout)
                            else:
                                CorrectionNetwork = _connect_layers(
                                    CorrectionNetwork,
                                    CorrectionTempWeights,
                                    CorrectionTempBiases,
                                    NetInstance.ActFun,
                                    NetInstance.ActFunParam,
                                    Dropout)

                    # Store all networks
                    Network = range(2)
                    if ForceFieldNetwork is not None:
                        Network[0] = ForceFieldNetwork
                    if CorrectionNetwork is not None:
                        Network[1] = CorrectionNetwork

                    # Store all input layers
                    InputLayer = range(2)
                    if ForceFieldInputLayer is not None:
                        InputLayer[0] = ForceFieldInputLayer
                    if CorrectionInputLayer is not None:
                        InputLayer[1] = CorrectionInputLayer

                    AtomicNNs.append([i, Network, InputLayer])
        else:
            print("No network data found!")

        self.AtomicNNs = AtomicNNs

    def make_atomic_networks(self, NetInstance):
        """Creates the specified partitioned network."""

        AtomicNNs = list()
        AllHiddenLayers = list()
        # Start Session
        if not NetInstance.Multiple:
            NetInstance._Session = _tf.Session(config=_tf.ConfigProto(
                intra_op_parallelism_threads=_multiprocessing.cpu_count()))
        if len(NetInstance.Structures) != len(
                NetInstance.NumberOfAtomsPerType):
            raise ValueError(
                "Length of Structures does not match length of NumberOfAtomsPerType")
        if not(isinstance(NetInstance.Structures[0], _PartitionedStructure)):
            raise ValueError("Please set IsPartitioned = False !")

        else:
            # make all the networks for the different atom types
            for i in range(0, len(NetInstance.Structures)):

                if len(NetInstance.Dropout) > i:
                    Dropout = NetInstance.Dropout[i]
                else:
                    Dropout = NetInstance.Dropout[-1]

                NetworkHiddenLayers = range(2)

                ForceFieldHiddenLayers = list()
                CorrectionHiddenLayers = list()

                ForceFieldWeights = list()
                CorrectionWeights = list()

                ForceFieldBias = list()
                CorrectionBias = list()

                ForceFieldNetwork = None
                CorrectionNetwork = None

                ForceFieldInputLayer = None
                CorrectionInputLayer = None

                CreateNewForceField = True
                CreateNewCorrection = True

                # Read structures for specific network parts
                StructureForAtom = NetInstance.Structures[i]
                ForceFieldStructure = StructureForAtom.ForceFieldNetworkStructure
                CorrectionStructure = StructureForAtom.CorrectionNetworkStructure
                # Construct networks out of loaded data
                if len(NetInstance._WeightData) != 0:
                    # Load data for atom
                    WeightData = NetInstance._WeightData[i]
                    BiasData = NetInstance._BiasData[i]
                    if len(WeightData.ForceFieldNetworkData) > 0:

                        CreateNewForceField = False
                        # Recreate force field network
                        ForceFieldWeights = WeightData.ForceFieldNetworkData
                        ForceFieldBias = BiasData.ForceFieldNetworkData

                        for j in range(1, len(ForceFieldStructure)):

                            ThisWeightData = ForceFieldWeights[j - 1]
                            ThisBiasData = ForceFieldBias[j - 1]
                            ForceFieldNrIn = ThisWeightData.shape[0]
                            ForceFieldNrHidden = ThisWeightData.shape[1]
                            ForceFieldHiddenLayers.append(
                                _construct_hidden_layer(
                                    ForceFieldNrIn,
                                    ForceFieldNrHidden,
                                    NetInstance.WeightType,
                                    ThisWeightData,
                                    NetInstance.BiasType,
                                    ThisBiasData,
                                    WeightData.ForceFieldVariable,
                                    NetInstance.InitMean,
                                    NetInstance.InitStddev))

                        NetworkHiddenLayers[0] = ForceFieldHiddenLayers

                    if len(WeightData.CorrectionNetworkData) > 0:
                        CreateNewCorrection = False
                        # Recreate correction network
                        CorrectionWeights = WeightData.CorrectionNetworkData
                        CorrectionBias = BiasData.CorrectionNetworkData

                        for j in range(1, len(CorrectionStructure)):

                            ThisWeightData = CorrectionWeights[j - 1]
                            ThisBiasData = CorrectionBias[j - 1]
                            CorrectionNrIn = ThisWeightData.shape[0]
                            CorrectionNrHidden = ThisWeightData.shape[1]
                            CorrectionHiddenLayers.append(
                                _construct_hidden_layer(
                                    CorrectionNrIn,
                                    CorrectionNrHidden,
                                    NetInstance.WeightType,
                                    ThisWeightData,
                                    NetInstance.BiasType,
                                    ThisBiasData,
                                    WeightData.CorrectionVariable,
                                    NetInstance.InitMean,
                                    NetInstance.InitStddev))

                        NetworkHiddenLayers[1] = CorrectionHiddenLayers

                if CreateNewForceField:
                    # Create force field network
                    for j in range(1, len(ForceFieldStructure)):
                        ForceFieldNrIn = ForceFieldStructure[j - 1]
                        ForceFieldNrHidden = ForceFieldStructure[j]
                        ForceFieldHiddenLayers.append(
                            _construct_hidden_layer(
                                ForceFieldNrIn,
                                ForceFieldNrHidden,
                                NetInstance.WeightType,
                                [],
                                NetInstance.BiasType,
                                [],
                                True,
                                NetInstance.InitMean,
                                NetInstance.InitStddev))

                    NetworkHiddenLayers[0] = ForceFieldHiddenLayers

                if CreateNewCorrection:
                    # Create correction network
                    for j in range(1, len(CorrectionStructure)):
                        CorrectionNrIn = CorrectionStructure[j - 1]
                        CorrectionNrHidden = CorrectionStructure[j]
                        CorrectionHiddenLayers.append(
                            _construct_hidden_layer(
                                CorrectionNrIn,
                                CorrectionNrHidden,
                                NetInstance.WeightType,
                                [],
                                NetInstance.BiasType,
                                [],
                                True,
                                NetInstance.InitMean,
                                NetInstance.InitStddev))

                    NetworkHiddenLayers[1] = CorrectionHiddenLayers

                AllHiddenLayers.append(NetworkHiddenLayers)

                for k in range(0, NetInstance.NumberOfAtomsPerType[i]):

                    if len(ForceFieldHiddenLayers) > 0:
                        # Make force field input layer
                        if not CreateNewForceField:
                            ForceFieldWeightData = NetInstance._WeightData[i].ForceFieldNetworkData
                            ForceFieldNrInputs = ForceFieldWeightData[0].shape[0]
                        else:
                            ForceFieldNrInputs = ForceFieldStructure[0]

                        ForceFieldInputLayer = _construct_input_layer(
                            ForceFieldNrInputs)
                        # Connect force field input to first hidden layer
                        ForceFieldFirstWeights = ForceFieldHiddenLayers[0][0]
                        NetInstance._FirstWeights.append(
                            ForceFieldFirstWeights)
                        ForceFieldFirstBiases = ForceFieldHiddenLayers[0][1]
                        ForceFieldNetwork = _connect_layers(
                            ForceFieldInputLayer,
                            ForceFieldFirstWeights,
                            ForceFieldFirstBiases,
                            NetInstance.ActFun,
                            NetInstance.ActFunParam,
                            Dropout)
                        # Connect force field hidden layers
                        for l in range(1, len(ForceFieldHiddenLayers)):
                            ForceFieldTempWeights = ForceFieldHiddenLayers[l][0]
                            ForceFieldTempBiases = ForceFieldHiddenLayers[l][1]
                            if l == len(ForceFieldHiddenLayers) - 1:
                                ForceFieldNetwork = _connect_layers(
                                    ForceFieldNetwork,
                                    ForceFieldTempWeights,
                                    ForceFieldTempBiases,
                                    "none",
                                    NetInstance.ActFunParam,
                                    Dropout)
                            else:
                                ForceFieldNetwork = _connect_layers(
                                    ForceFieldNetwork,
                                    ForceFieldTempWeights,
                                    ForceFieldTempBiases,
                                    NetInstance.ActFun,
                                    NetInstance.ActFunParam,
                                    Dropout)

                    if len(CorrectionHiddenLayers) > 0:
                        # Make correction input layer
                        if not CreateNewCorrection:
                            CorrectionWeightData = NetInstance._WeightData[i].CorrectionNetworkData
                            CorrectionNrInputs = CorrectionWeightData[0].shape[0]
                        else:
                            CorrectionNrInputs = CorrectionStructure[0]

                        CorrectionInputLayer = _construct_input_layer(
                            CorrectionNrInputs)
                        # Connect Correction input to first hidden layer
                        CorrectionFirstWeights = CorrectionHiddenLayers[0][0]
                        NetInstance._FirstWeights.append(
                            CorrectionFirstWeights)
                        CorrectionFirstBiases = CorrectionHiddenLayers[0][1]
                        CorrectionNetwork = _connect_layers(
                            CorrectionInputLayer,
                            CorrectionFirstWeights,
                            CorrectionFirstBiases,
                            NetInstance.ActFun,
                            NetInstance.ActFunParam,
                            Dropout)
                        # Connect Correction hidden layers
                        for l in range(1, len(CorrectionHiddenLayers)):
                            CorrectionTempWeights = CorrectionHiddenLayers[l][0]
                            CorrectionTempBiases = CorrectionHiddenLayers[l][1]
                            if l == len(CorrectionHiddenLayers) - 1:
                                CorrectionNetwork = _connect_layers(
                                    CorrectionNetwork,
                                    CorrectionTempWeights,
                                    CorrectionTempBiases,
                                    "none",
                                    NetInstance.ActFunParam,
                                    Dropout)
                            else:
                                CorrectionNetwork = _connect_layers(
                                    CorrectionNetwork,
                                    CorrectionTempWeights,
                                    CorrectionTempBiases,
                                    NetInstance.ActFun,
                                    NetInstance.ActFunParam,
                                    Dropout)

                    # Store all networks
                    Network = range(2)
                    if ForceFieldNetwork is not None:
                        Network[0] = ForceFieldNetwork
                    if CorrectionNetwork is not None:
                        Network[1] = CorrectionNetwork

                    # Store all input layers
                    InputLayer = range(2)
                    if ForceFieldInputLayer is not None:
                        InputLayer[0] = ForceFieldInputLayer
                    if CorrectionInputLayer is not None:
                        InputLayer[1] = CorrectionInputLayer

                    AtomicNNs.append([i, Network, InputLayer])

            self.AtomicNNs = AtomicNNs
            self.VariablesDictionary = AllHiddenLayers


class MultipleInstanceTraining(object):
    """This class implements the possibillities to train multiple training
    instances at once. This is neccessary if the datasets have a different
    number of atoms per species. """

    def __init__(self):
        # Training variables
        self.TrainingInstances = list()
        self.EpochsPerCycle = 1
        self.GlobalEpochs = 100
        self.GlobalStructures = list()
        self.GlobalLearningRate = 0.001
        self.GlobalCostCriterium = 0
        self.Global_dE_Criterium = 0
        self.GlobalRegularization = "L2"
        self.GlobalRegularizationParam = 0.0001
        self.GlobalOptimizer = "Adam"
        self.GlobalTrainingCosts = list()
        self.GlobalValidationCosts = list()
        self.GlobalMinOfOut = 0
        self.MakePlots = False
        self.IsPartitioned = False
        self.GlobalSession = _tf.Session()

    def initialize_multiple_instances(self):
        """Initializes all instances with the same parameters."""

        Execute = True
        if len(self.TrainingInstances) == 0:
            Execute = False
            print("No training instances available!")

        if Execute:
            # Initialize all instances with same settings
            for Instance in self.TrainingInstances:
                Instance.Multiple = True
                Instance.Epochs = self.EpochsPerCycle
                Instance.MakeAllVariable = True
                Instance.Structures = self.GlobalStructures
                Instance.Session = self.GlobalSession
                Instance.MakePlots = False
                Instance.ActFun = "relu"
                Instance.CostCriterium = 0
                Instance.dE_Criterium = 0
                Instance.IsPartitioned = self.IsPartitioned
                Instance.WeightType = "truncated_normal"
                Instance.LearningRate = self.GlobalLearningRate
                Instance.OptimizerType = self.GlobalOptimizer
                Instance.Regularization = self.GlobalRegularization
                Instance.RegularizationParam = self.GlobalRegularizationParam
                if Instance._MinOfOut < self.GlobalMinOfOut:
                    self.GlobalMinOfOut = Instance.MinOfOut

                # Clear unnecessary data
                Instance.Ds.geometries = list()
                Instance.Ds.Energies = list()
                Instance.Batches = list()
                Instance.AllGeometries = list()
            # Write global minimum to all instances
            for Instance in self.TrainingInstances:
                Instance.MinOfOut = self.GlobalMinOfOut

    def set_session(self):
        """Sets the session of the currently trained instance to the
        global session"""

        for Instance in self.TrainingInstances:
            Instance.Session = self.GlobalSession

    def train_multiple_instances(self, StartModelName=None):
        """Trains each instance for EpochsPerCylce epochs then uses the resulting network
        as a basis for the next training instance.
        Args:
            StartModelName (str): Path to a saved model."""

        print("Startet multiple instance training!")
        ct = 0
        LastStepsModelData = list()
        for i in range(0, self.GlobalEpochs):
            for Instance in self.TrainingInstances:
                if ct == 0:
                    if StartModelName is not None:
                        Instance.expand_existing_net(ModelName=StartModelName)
                    else:
                        Instance.make_and_initialize_network()
                else:
                    Instance.expand_existing_net(ModelData=LastStepsModelData)

                LastStepsModelData = Instance.start_batch_training()
                _tf.reset_default_graph()
                self.GlobalSession = _tf.Session()
                self.set_session()
                self.GlobalTrainingCosts += Instance.OverallTrainingCosts
                self.GlobalValidationCosts += Instance.OverallValidationCosts
                if ct % max(int((self.GlobalEpochs * len(self.TrainingInstances)) / 50),
                            1) == 0 or i == (self.GlobalEpochs - 1):
                    if self.MakePlots:
                        if ct == 0:
                            fig, ax, TrainingCostPlot, ValidationCostPlot, RunningMeanPlot = _initialize_cost_plot(
                                self.GlobalTrainingCosts, self.GlobalValidationCosts)
                        else:
                            _update_cost_plot(
                                fig,
                                ax,
                                TrainingCostPlot,
                                self.GlobalTrainingCosts,
                                ValidationCostPlot,
                                self.GlobalValidationCosts,
                                RunningMeanPlot)

                    # Finished percentage output
                    print(str(100 * ct / (self.GlobalEpochs *
                                          len(self.TrainingInstances))) + " %")
                    _np.save("trained_variables", LastStepsModelData)
                ct = ct + 1
                # Abort criteria
                if self.GlobalTrainingCosts <= self.GlobalCostCriterium and \
                self.GlobalValidationCosts <= self.GloablCostCriterium or \
                Instance.DeltaE < self.Global_dE_Criterium:

                    if self.GlobalValidationCosts != 0:
                        print("Reached criterium!")
                        print(
                            "Cost= " + str((self.GlobalTrainingCosts + \
                                            self.GlobalValidationCosts) / 2))
                        print("delta E = " + str(Instance.DeltaE) + " ev")
                        print("Epoch = " + str(i))
                        print("")

                    else:
                        print("Reached criterium!")
                        print("Cost= " + str(self.GlobalTrainingCosts))
                        print("delta E = " + str(Instance.DeltaE) + " ev")
                        print("Epoch = " + str(i))
                        print("")

                    print("Training finished")
                    break

                if i == (self.GlobalEpochs - 1):
                    print("Training finished")
                    print("delta E = " + str(Instance.DeltaE) + " ev")
                    print("Epoch = " + str(i))
                    print("")


class _PartitionedStructure(object):
    """This class is a container for the partitioned network structures."""

    def __init__(self):

        self.ForceFieldNetworkStructure = list()
        self.CorrectionNetworkStructure = list()


class _PartitionedNetworkData(object):
    """This class is a container for the partitioned network data."""

    def __init__(self):

        self.ForceFieldNetworkData = list()
        self.CorrectionNetworkData = list()
        self.ForceFieldVariable = False
        self.CorrectionVariable = False
