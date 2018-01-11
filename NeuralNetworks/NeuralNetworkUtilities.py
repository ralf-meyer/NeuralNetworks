#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:30:10 2017

@author: Fuchs Alexander
"""
import os as _os
import random as _rand
import time as _time
import warnings

import matplotlib.pyplot as _plt
import numpy as _np
import tensorflow as _tf
from psutil import virtual_memory

import NeuralNetworks.descriptors.SymmetryFunctionSet as _SymmetryFunctionSet
from NeuralNetworks import DataSet as _DataSet
from NeuralNetworks.types.PartitionedAtomicNetwork import _PartitionedAtomicNetwork
from NeuralNetworks.types.StandardAtomicNetwork import _StandardAtomicNetwork
from data_generation import data_readers as _readers

_plt.ion()
_tf.reset_default_graph()




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

def _initialize_delta_e_plot(delta_e):
    fig = _plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscaley_on(True)
    de_plot, = ax.semilogy(
        _np.arange(0, len(delta_e)), delta_e)
    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel("batches")
    ax.set_ylabel("\delta E")
    ax.set_title("Averaged energy difference /eV")
    fig.legend(handles=[de_plot],labels=["$\Delta$ E"], loc=1)
    # We need to draw *and* flush
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig,ax,de_plot

def _update_delta_e_plot(delta_e,delta_e_plot,figure,ax):
    delta_e_plot.set_data(_np.arange(0, len(delta_e)), delta_e)
    # Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    # We need to draw *and* flush
    figure.canvas.draw()
    figure.canvas.flush_events()

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
    ax.set_xlabel("dataset")
    ax.set_ylabel("log(cost)")
    ax.set_title("Normalized cost per batch")
    if len(ValidationData) != 0:
        fig.legend(handles=[TrainingCostPlot,ValidationCostPlot,RunningMeanPlot],labels=["Training cost","Validation cost","Running avg"],loc=1)
    else:
        fig.legend(handles=[TrainingCostPlot,RunningMeanPlot],labels=["Training cost","Running avg"],loc=1)
        
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
        self.ClippingValue = 1e10
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
        self.LearningDecayEpochs = 1000
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
        self.CostCriterion = 0
        self.dE_Criterion = 0
        self.OptimizerType = None
        self.OptimizerProp = None
        self.DeltaE = []
        self.TrainingCosts = []
        self.ValidationCosts = []
        self.OverallTrainingCosts = []
        self.OverallValidationCosts = []
        self.TrainedVariables = []
        self.CostFunType = "squared-difference"
        self.SavingDirectory = "save"
        self.CalcDatasetStatistics=True
        self._IsFromCheck=False
        # Symmetry function set settings
        self.NumberOfRadialFunctions = 20
        self.Rs = []
        self.R_Etas = []
        self.Etas = []
        self.Zetas = []
        self.Lambs = []
        self.Cutoff=7
        self.CutoffType="cos"
        # Private variables

        # Class instances
        self._Session = None
        self._Net = None
        self._SymmFunSet = None
        self._RadialFunSet=None
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
        self._ForceFieldTrainingInputs = []
        self._ForceFieldValidationInputs = []
        self._ForceFieldTrainingDerivatives = []
        self._ForceFieldValidationDerivatives = []
        # Net parameters
        self._MeansOfDs = []
        self._MinOfOut = None
        self._VarianceOfDs = []
        self._BiasData = []
        self._WeightData = None
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
        self._AllGVectors = []
        self._AllGDerivatives = []
        self._AllForceFieldInputs=[]
        self._AllForceFieldDerivatives=[]
        self.PESCheck=None
        self.TextOutput=True

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
            self._OutputLayer = _tf.placeholder(_tf.float64, shape=[None, 1])
            if self.UseForce:
                self._OutputLayerForce = _tf.placeholder(_tf.float64, shape=[None,sum(self.NumberOfAtomsPerType) * 3])

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
           if self.TextOutput:
               print("Evaluation only no training\
                 supported if all networks are constant!")
            # Initialize session
        self._Session.run(_tf.global_variables_initializer())

    def load_model(self, ModelName="save/trained_variables",load_statistics=True):
        """Loads the model in the specified folder.
        Args:
            ModelName(str):Path to the model.
        Returns:
            1
        """
        if ".npy" not in ModelName:
            ModelName = ModelName + ".npy"
            # try:

        rare_model = _np.load(ModelName)
        self.TrainedVariables = rare_model[0]

        if load_statistics:
            self._MeansOfDs = rare_model[1]
            self._VarianceOfDs = rare_model[2]
        self._MinOfOut = rare_model[3]
        self.Rs=rare_model[4]
        self.R_Etas=rare_model[5]
        self.Etas=rare_model[6]
        self.Lambs=rare_model[7]
        self.Zetas=rare_model[8]
        self.NumberOfRadialFunctions=rare_model[9]
        try:
            self.Cutoff=rare_model[10]
        except:
            self.Cutoff=7
        try:
            self.IsPartitioned=rare_model[11]
        except:
            self.IsPartitioned=False
        try:
            self.Dropout=rare_model[12]
        except:
            self.Dropout=[0]


        if self.IsPartitioned:
            if len(self.Atomtypes) == 0:
                ANN = self.TrainedVariables[0]
                for i, NetForType in enumerate(ANN):
                    if isinstance(NetForType[0][-1], basestring):
                        self.Atomtypes.append(NetForType[0][-1])
                        if self.TextOutput:
                            print("Atomtype " + str(i) + " is " + str(self.Atomtypes[-1]))
        else:
            if len(self.Atomtypes) == 0:
                for i, NetForType in enumerate(self.TrainedVariables):
                    if isinstance(NetForType[0][-1], basestring):
                        self.Atomtypes.append(NetForType[0][-1])
                        if self.TextOutput:
                            print("Atomtype " + str(i) + " is " + str(self.Atomtypes[-1]))


        return 1

    def expand_existing_net(
            self,
            ModelName="save/trained_variables",
            MakeAllVariable=True,
            ModelData=None,
            ConvertToPartitioned=False,
            load_statistics=True):
        """Creates a new network out of stored data.
        Args:
            MakeAllVariables(bool): Specifies if all layers can be trained
            ModelData (list): Passes the model directly from a training before.
            ConvertToPartitioned(bool):Converts a StandardAtomicNetwork to a
            PartitionedAtomicNetwork network with the StandardAtomicNetwork
            beeing the force network part."""
        if ModelData is None:
            Success = self.load_model(ModelName,load_statistics=load_statistics)
        else:
            self.TrainedVariables = ModelData[0]
            Success = 1
        if Success == 1:
            if self.TextOutput:
                print("Model successfully loaded!")
            self._Net=None
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
                self._WeightData, self._BiasData,struct= \
                self._Net.get_weights_biases_from_data(
                    self.TrainedVariables)
            if len(self.Structures)==0:
                self.Structures=struct
                if self.TextOutput:
                    print("Loaded structure: " + str(self.Structures))
            else:
                if self.Structures[0][0]!=struct[0][0]:
                    print("Specified: "+str(self.Structures))
                    print("Loaded: "+str(struct))
                    raise ValueError("Specified and loaded structure do not match!")

            self.MakeAllVariable = MakeAllVariable
            # try:
            self.make_and_initialize_network()
        else:
            raise ValueError("No model : "+str(ModelName))
            # except:
            #    print("Partitioned network loaded, please set IsPartitioned=True")


    def evaluate(self, Tensor, Layers, Data):
        """Evaluate model for given input data
        Returns:
            The output of the given tensor"""
        if len(Layers) == 1:
            return self._Session.run(Tensor, feed_dict={Layers[0]: Data})
        else:
            return self._Session.run(
                Tensor, feed_dict={
                    i: d for i, d in zip(
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
                feed_dict={i: d for i, d in zip(Layers, Data)})

        return Cost

    def _validate_step(self, Layers, Data):
        """Calculates the validation cost for this training step,
        without optimizing the net.

        Returns:
            Cost (float): The cost for the data"""
        # Evaluate cost function without changing the network
        Cost = self._Session.run(
            self.CostFun, feed_dict={
                i: d for i, d in zip(
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

            if TrainCost[-1] < self.CostCriterion:
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
            if self.IsPartitioned:
                FFTrainingInputs=self.TrainingBatches[i][5]
            else:
                FFTrainingInputs = []

            TrainingData = self._Net.make_data_for_atomicNNs(
                TrainingInputs,
                TrainingOutputs,
                [],
                [],
                [],
                False,
                FFTrainingInputs,
                [])

            if i == 0:
                train_dE = self.evaluate(self._dE_Fun, Layers, TrainingData)
            else:
                temp = self.evaluate(self._dE_Fun, Layers, TrainingData)
                train_dE = _np.concatenate((train_dE, temp))

        for i in range(0, len(self.ValidationBatches)):
            ValidationInputs = self.ValidationBatches[i][0]
            ValidationOutputs = self.ValidationBatches[i][1]

            if self.IsPartitioned:
                FFValidationInputs=self.TrainingBatches[i][5]
            else:
                FFValidationInputs=[]

            ValidationData = self._Net.make_data_for_atomicNNs(
                ValidationInputs,
                ValidationOutputs,
                [],
                [],
                [],
                False,
                FFValidationInputs,
                [])

            if i == 0:
                val_dE = self.evaluate(self._dE_Fun, Layers, ValidationData)
            else:
                temp = self.evaluate(self._dE_Fun, Layers, ValidationData)
                val_dE = _np.concatenate((val_dE, temp))

        #with self._Session.as_default():
        #    train_dE = train_dE.eval().tolist()
        #    val_dE = val_dE.eval().tolist()

        train_mean = _np.mean(train_dE)
        train_var = _np.var(train_dE)
        val_mean = _np.mean(val_dE)
        val_var = _np.var(val_dE)

        train_stat = [train_mean, train_var]
        val_stat = [val_mean, val_var]

        return train_stat, val_stat

    def energy_for_geometry(self, geometry):
        "Evaluates the energy for a given geometry"
        return self.eval_dataset_energy(
                self._convert_single_geometry(geometry),0)

    def force_for_geometry(self, geometry):
        "Evaluates the force for a given geometry"
        return self.eval_dataset_force(
                self._convert_single_geometry(geometry),0)


    def energy_and_force_for_geometry(self, geometry):
        "Evaluates the force for a given geometry"
        input=self._convert_single_geometry(geometry)
        energy=self.eval_dataset_energy(input,0)
        force= self.eval_dataset_force(input,0)

        return energy,force

    def eval_dataset_energy(self, Batches, BatchNr=0,FFInputs=[]):
        """Prepares and evaluates the dataset for the loaded network.

        Args:
            Batches (list):List of a list of numpy arrays.
            BatchNr (int): Index of the batch to evaluate.

        Returns:
            List of the predicted energies for the dataset
        """
        AllData = Batches[BatchNr]
        GData = AllData[0]
        if self.IsPartitioned and len(FFInputs)==0:
            if self.UseForce:
                FFInputs=AllData[5]
            else:
                FFInputs=AllData[2]
        Layers, Data = self._Net.prepare_data_environment(
                                                    GData,
                                                    None,
                                                    [],
                                                    None,
                                                    [],
                                                    [],
                                                    [],
                                                    False,
                                                    FFInputs,
                                                    [])

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
        if self.IsPartitioned:
            FFInputs = AllData[5]
            FFDerivatives = AllData[6]
        else:
            FFInputs = []
            FFDerivatives = []


        Layers, Data = self._Net.prepare_data_environment(
                                                        GData,
                                                        None,
                                                        [],
                                                        None,
                                                        DerGData,
                                                        [],
                                                        norm,
                                                        True,
                                                        FFInputs,
                                                        FFDerivatives)


        return self.evaluate(self._OutputForce, Layers, Data)


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

        if self._DataSet is None:
            raise Exception(
                    "No dataset initialized please call\
                    init_dataset first!")
        if self._SymmFunSet is None:
            raise Exception(
                    "No symmetry function set initialized please call\
                    create_symmetry_functions or init_dataset first!")

        if self._Net is None:
            raise Exception(
                    "No atomic network specified please call\
                    make_and_initialize_network or expand_existing_net first!")

        if self.TrainingBatches ==[]:
            raise Exception(
                    "No training batches specified please call\
                    make_training_and_validation_data first!")


        # Clear cost array for multi instance training
        self.OverallTrainingCosts = []
        self.OverallValidationCosts = []
        NormalizationTraining = []
        NormalizationValidation = []
        self.DeltaE=[]

        start = _time.time()
        Execute = True
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
                        if self.IsPartitioned:
                            self._ForceFieldTrainingInputs = self.TrainingBatches[rnd][5]
                            self._ForceFieldTrainingDerivatives= self.TrainingBatches[rnd][6]
                        if self.ValidationBatches:
                            self._ForceValidationInput = self.ValidationBatches[rnd][2]
                            NormalizationValidation = self.ValidationBatches[rnd][3]
                            self._ForceValidationOutput = self.ValidationBatches[rnd][4]
                            if self.IsPartitioned:
                                self._ForceFieldValidationInputs = self.ValidationBatches[rnd][5]
                                self._ForceFieldValidationDerivatives=self.ValidationBatches[rnd][6]
                    else:
                        if self.IsPartitioned:
                            self._ForceFieldTrainingInputs = self.TrainingBatches[rnd][5]
                            self._ForceFieldValidationInputs = self.ValidationBatches[rnd][5]

                    # Prepare data and layers for feeding
                    if i == 0:
                        EnergyLayers = self._Net.make_layers_for_atomicNNs(
                            self._OutputLayer, [], False)
                        Layers, TrainingData = self._Net.prepare_data_environment(
                            self._TrainingInputs,
                            self._OutputLayer,
                            self._TrainingOutputs,
                            self._OutputLayerForce,
                            self._ForceTrainingInput,
                            self._ForceTrainingOutput,
                            NormalizationTraining,
                            self.UseForce,
                            self._ForceFieldTrainingInputs,
                            self._ForceFieldTrainingDerivatives)
                    else:
                        TrainingData = self._Net.make_data_for_atomicNNs(
                            self._TrainingInputs,
                            self._TrainingOutputs,
                            self._ForceTrainingInput,
                            self._ForceTrainingOutput,
                            NormalizationTraining,
                            self.UseForce,
                            self._ForceFieldTrainingInputs,
                            self._ForceFieldTrainingDerivatives)
                    # Make validation input vector
                    if len(self._ValidationInputs) > 0:
                        ValidationData = self._Net.make_data_for_atomicNNs(
                            self._ValidationInputs,
                            self._ValidationOutputs,
                            self._ForceValidationInput,
                            self._ForceValidationOutput,
                            NormalizationValidation,
                            self.UseForce,
                            self._ForceFieldValidationInputs,
                            self._ForceFieldValidationDerivatives)
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
                    self.DeltaE.append((self.calc_dE(Layers, TrainingData) +
                                   self.calc_dE(Layers, TrainingData)) / 2)
                else:
                    self.DeltaE.append(self.calc_dE(Layers, TrainingData))

                if not self.Multiple:
                    if i % max(int(self.Epochs / 100),
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
                                    de_fig, de_ax, de_plot = _initialize_delta_e_plot(self.DeltaE)
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
                                    if self.PESCheck != None:
                                        self.PESCheck.pes_check()
                                    _update_cost_plot(
                                        fig,
                                        ax,
                                        TrainingCostPlot,
                                        self.OverallTrainingCosts,
                                        ValidationCostPlot,
                                        self.OverallValidationCosts,
                                        RunningMeanPlot)
                                    _update_delta_e_plot(self.DeltaE, de_plot, de_fig, de_ax)
                        # Finished percentage output
                        print([str(100 * i / self.Epochs) + " %",
                               "deltaE = " + str(self.DeltaE[-1]) + " eV",
                               "Cost = " + str(self.TrainingCosts),
                               "t = " + str(_time.time() - start) + " s",
                               "global step: " + str(self._Session.run(self.GlobalStep))])
                        Prediction = self.eval_dataset_energy(
                            [[self._TrainingInputs]],FFInputs=self._ForceFieldTrainingInputs)
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
                            self._Session,self.Atomtypes)

                        if not _os.path.exists(self.SavingDirectory):
                            _os.makedirs(self.SavingDirectory)
                        _np.save(self.SavingDirectory + "/trained_variables",
                                 [self.TrainedVariables,
                                  self._MeansOfDs,
                                  self._VarianceOfDs,
                                  self._MinOfOut,
                                  self.Rs,
                                  self.R_Etas,
                                  self.Etas,
                                  self.Lambs,
                                  self.Zetas,
                                  self.NumberOfRadialFunctions,
                                  self.Cutoff,
                                  self.IsPartitioned,
                                  self.Dropout])

                    # Abort criteria
                    if self.TrainingCosts != 0 and self.TrainingCosts <= self.CostCriterion and self.ValidationCosts <= self.CostCriterion or self.DeltaE[-1] < self.dE_Criterion:

                        if self.ValidationCosts != 0:
                            print("Reached Criterion!")
                            print(
                                "Cost= " + str((self.TrainingCosts + self.ValidationCosts) / 2))
                            print("delta E = " + str(self.DeltaE[-1]) + " eV")
                            print("t = " + str(_time.time() - start) + " s")
                            print("Epoch = " + str(i))
                            print("")

                        else:
                            print("Reached Criterion!")
                            print("Cost= " + str(self.TrainingCosts))
                            print("delta E = " + str(self.DeltaE[-1]) + " eV")
                            print("t = " + str(_time.time() - start) + " s")
                            print("Epoch = " + str(i))
                            print("")

                        if self.Multiple==False:
                            print("Calculation of whole dataset energy difference ...")
                            train_stat, val_stat = self._dE_stat(EnergyLayers)
                            print("Training dataset error= " +
                                  str(train_stat[0]) +
                                  "+-" +
                                  str(_np.sqrt(train_stat[1])) +
                                  " eV")
                            print("Validation dataset error= " +
                                  str(val_stat[0]) +
                                  "+-" +
                                  str(_np.sqrt(val_stat[1])) +
                                  " eV")
                            print("Training finished")
                            break

                    if i == (self.Epochs - 1) and self.Multiple==False:

                        print("Training finished")
                        print("delta E = " + str(self.DeltaE[-1]) + " eV")
                        print("t = " + str(_time.time() - start) + " s")
                        print("")

                        train_stat, val_stat = self._dE_stat(EnergyLayers)
                        print("Training dataset error= " +
                              str(train_stat[0]) +
                              "+-" +
                              str(_np.sqrt(train_stat[1])) +
                              " eV")
                        print("Validation dataset error= " +
                              str(val_stat[0]) +
                              "+-" +
                              str(_np.sqrt(val_stat[1])) +
                              " eV")

            if self.Multiple:
                self.TrainedVariables = self._Net.get_trained_variables(
                    self._Session, self.Atomtypes)
                return [self.TrainedVariables, self._MinOfOut]

    def create_symmetry_functions(self):

        if self.Atomtypes==[]:
            raise Exception("No Atomtypes specified!")

        if self.Etas==[]and self.Lambs==[] and self.Zetas==[]:
            print("No angular symmetry functions specified!")

        self.SizeOfInputsPerType = []
        #Add symmetry functions for the atomic neural net
        if self.IsPartitioned:
            self.CutoffType="shortRange"
        self._SymmFunSet = _SymmetryFunctionSet.SymmetryFunctionSet(
            self.Atomtypes, self.Cutoff)
        if len(self.Rs) == 0:
            self._SymmFunSet.add_radial_functions_evenly(
                self.NumberOfRadialFunctions)
        else:
            self._SymmFunSet.add_radial_functions(self.Rs, self.R_Etas,self.CutoffType,self.Cutoff)

        self._SymmFunSet.add_angular_functions(self.Etas, self.Zetas, self.Lambs)
        if self.IsPartitioned:
            #Add symmetry function set for force field
            self._RadialFunSet = _SymmetryFunctionSet.SymmetryFunctionSet(
                self.Atomtypes, 100)#get symmetry function set with a large global cutoff
            for type1 in self.Atomtypes:
                for type2 in self.Atomtypes:
                    self._RadialFunSet.add_TwoBodySymmetryFunction(type1,
                                                                   type2,
                                                                   "OneOverR6",
                                                                   [],
                                                                   "longRange",
                                                                   self.Cutoff)
                    self._RadialFunSet.add_TwoBodySymmetryFunction(type1,
                                                                   type2,
                                                                   "OneOverR8",
                                                                   [],
                                                                   "longRange",
                                                                   self.Cutoff)
                    self._RadialFunSet.add_TwoBodySymmetryFunction(type1,
                                                                   type2,
                                                                   "OneOverR10",
                                                                   [],
                                                                   "longRange",
                                                                   self.Cutoff)





        self.SizeOfInputsPerType = self._SymmFunSet.num_Gs
        for i, a_type in enumerate(self.NumberOfAtomsPerType):
            for j in range(0, a_type):
                self.SizeOfInputsPerAtom.append(self.SizeOfInputsPerType[i])

    def _convert_single_geometry(self,geometry):
        """Converts a single geometry to symmetry functions"""
        dGs=[]
        FFIn=[]
        dFFIn=[]
        Gs=[_np.asarray(self._SymmFunSet.eval_geometry(geometry))]
        if self.IsPartitioned:
            FFIn=[_np.asarray(self._RadialFunSet.eval_geometry(geometry))]

        if self.UseForce:
            dGs=[_np.asarray(self._SymmFunSet.eval_geometry_derivatives(geometry))]
            if self.IsPartitioned:
                dFFIn=[_np.asarray(self._RadialFunSet.eval_geometry_derivatives(geometry))]
        if self.CalcDatasetStatistics:
            AllTemp=[G for G in Gs[0]]
            self._calculate_statistics_for_dataset(AllTemp)
        GInputs, GDerivativesInput,Normalization,FFInputs,FFDerivatives=self._sort_and_normalize_data(1,Gs,dGs,FFIn,dFFIn)

        if self.UseForce:
            if self.IsPartitioned:
                return [[GInputs,[], GDerivativesInput,Normalization,[],FFInputs,FFDerivatives]]
            else:
                return [[GInputs, [], GDerivativesInput, Normalization, []]]
        else:
            if self.IsPartitioned:
                return [[GInputs, [],[],[],FFInputs]]
            else:
                return [[GInputs, []]]

    def _convert_dataset(self, TakeAsReference,DataPointsPercentage):
        """Converts the cartesian coordinates to a symmetry function vector and
        calculates the mean value and the variance of the symmetry function
        vector.

        Args:
            TakeAsReference(bool): Specifies if the variance and mean values should be
                                set according to this dataset.
        """
        if self.TextOutput:
            print("Converting data to neural net input format...")
        NrGeom = int(len(self._DataSet.geometries)*DataPointsPercentage/100)
        AllTemp = []
        #Guess size in memory for all geometries
        test_size=_np.asarray(self._SymmFunSet.eval_geometry(
                self._DataSet.geometries[0])).nbytes
        if self.UseForce:
            test_size+=_np.asarray(
                        self._SymmFunSet.eval_geometry_derivatives(
                            self._DataSet.geometries[0])).nbytes
        if self.TextOutput:
            print("Needed memory:" + str(2 * NrGeom * test_size / 1e9) + " GB") # Factor of two because of batch generation
        if 2*test_size*NrGeom>virtual_memory().total:
            warnings.warn("Not enough memory for all geometries!")

        # Get G vectors
        for i in range(0, NrGeom):

            temp = self._SymmFunSet.eval_geometry(
                        self._DataSet.geometries[i])

            self._AllGVectors.append(_np.asarray(temp))
            if self.UseForce:
                self._AllGDerivatives.append(
                    _np.asarray(
                        self._SymmFunSet.eval_geometry_derivatives(
                            self._DataSet.geometries[i])))

            if self._RadialFunSet != None:
                self._AllForceFieldInputs.append(
                    _np.asarray(
                        self._RadialFunSet.eval_geometry(self._DataSet.geometries[i])))
                if self.UseForce!= None:
                    self._AllForceFieldDerivatives.append(_np.asarray(
                        self._RadialFunSet.eval_geometry_derivatives(self._DataSet.geometries[i])))

            if i % max(int(NrGeom / 25), 1) == 0 and self.TextOutput:
                print(str(100 * i / NrGeom) + " %")

            AllTemp+=[Gs for Gs in temp]

        if TakeAsReference:
            self._calculate_statistics_for_dataset(AllTemp)

    def _calculate_statistics_for_dataset(self, AllTemp):
        """To be documented..."""

        NrAtoms = sum(self.NumberOfAtomsPerType)
        # calculate mean and sigmas for all Gs
        if self.TextOutput:
            print("Calculating mean values and variances...")
        # Input statistics
        #try:
        if self.CalcDatasetStatistics:
            self._MeansOfDs = _np.mean(AllTemp, axis=0)
            raw_var =_np.var(AllTemp, axis=0)
            L_finite =_np.isfinite(raw_var)
            L_non_zero = raw_var > 0
            L=_np.logical_and(L_non_zero,L_finite)
            var=_np.ones_like(raw_var)
            var[L]=raw_var[L]
            self._VarianceOfDs =_np.maximum(1e-2,var)
        #else:
        #    self._MeansOfDs=_np.multiply(_np.ones((self.SizeOfInputsPerType[0])),6)
        #    self._VarianceOfDs=_np.multiply(_np.ones((self.SizeOfInputsPerType[0])),25)

        self._VarianceOfDs=_np.nan_to_num(self._VarianceOfDs)

        #except:
        #    raise ValueError(
        #            "Number of atoms per type does not match input!")
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
            LoadGeometries=True,
            DataPointsPercentage=100,
            calibrate=True,
            Calibration=[]):
        """Reads lammps files,adds symmetry functions to the symmetry function
        basis and converts the cartesian corrdinates to symmetry function vectors.

        Args:
            TakeAsReference(bool): Specifies if the variance and mean values should be
                                set according to this dataset.
            LoadGeometries(bool): Specifies if the conversion of the geometry
                                coordinates should be performed."""

        my_calibration = []
        self._DataSet = _DataSet.DataSet()
        self._Reader = _readers.QE_MD_Reader()
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

        self._Reader.get_files(path)
        self._Reader.read_all_files()
        if calibrate:
            if len(Calibration) > 0:
                my_calibration=[]
                for i in range(len(Calibration)):
                    my_calibration.append((Calibration[i],self._Reader.nr_atoms_per_type[i]))
            self._Reader.Calibration=my_calibration
            self._Reader.calibrate_energy()
        else:
            self._Reader.energies=self._Reader.e_pot

        self.Atomtypes = self._Reader.atom_types
        self.NumberOfAtomsPerType = self._Reader.nr_atoms_per_type
        self.init_dataset(self._Reader.geometries,self._Reader.energies,
                     self._Reader.forces, TakeAsReference,DataPointsPercentage)




    def read_lammps_files(
            self,
            path,
            energy_unit="eV",
            dist_unit="A",
            TakeAsReference=True,
            LoadGeometries=True,
            DataPointsPercentage=100,
            calibrate=True):
        """Reads lammps files,adds symmetry functions to the symmetry function
        basis and converts the cartesian corrdinates to symmetry function vectors.

        Args:
            TakeAsReference(bool): Specifies if the MinOfOut Parameter should be
                                set according to this dataset.
            LoadGeometries(bool): Specifies if the conversion of the geometry
                                coordinates should be performed."""


        self._DataSet = _DataSet.DataSet()
        self._Reader = _readers.LammpsReader()
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

        self._Reader.read_folder(path)
        if calibrate:
            self._Reader.calibrate_energy()
        else:
            self._Reader.energies=self._Reader.e_pot

        self.Atomtypes = self._Reader.atom_types
        self.NumberOfAtomsPerType = self._Reader.nr_atoms_per_type
        self.init_dataset(self._Reader.geometries,self._Reader.energies,
                     self._Reader.forces, TakeAsReference,DataPointsPercentage)

        print("Added dataset!")

    def init_dataset(self, geometries, energies,
                     forces=[], TakeAsReference=True,DataPointsPercentage=100):
        """Initializes a loaded dataset.

        Args:
            geometries (list): List of geomtries
            energies (list) : List of energies
            forces (list): List of G-vector derivatives
            TakeAsReference (bool): Specifies if the variance and mean values should be
                                set according to this dataset."""

        if len(geometries) == len(energies):
            self._DataSet = _DataSet.DataSet()
            self._DataSet.energies = energies
            self._DataSet.geometries = geometries
            self._DataSet.forces = forces
            if TakeAsReference:
                self._VarianceOfDs = []
                self._MeansOfDs = []
            if self._SymmFunSet==None:
                self.create_symmetry_functions()

            self._convert_dataset(TakeAsReference,DataPointsPercentage)
        else:
            print("Number of energies: " +
                  str(len(energies)) +
                  " does not match number of geometries: " +
                  str(len(geometries)))

    def prepare_evaluation(self,model_name,nr_atoms_per_type,structure=[],use_force=True,atom_types=[]):

        #Default symmetry function set
        self.Structures=[]
        self.Atomtypes=[]
        self.NumberOfAtomsPerType=[]
        self._DataSet = _DataSet.DataSet()
        self.UseForce=use_force
        if not("trained_variables" in model_name):
            model_name=model_name+"/trained_variables"
        if not(self._IsFromCheck):
            self.load_model(model_name,load_statistics=True)
        if len(atom_types)>0:
            self.Atomtypes = atom_types
        self.NumberOfAtomsPerType = nr_atoms_per_type
        self.create_symmetry_functions()

        # if len(structure)==0:
        #     MyStructure=[80,60,40,20,1]
        # else:
        #     MyStructure=structure
        #
        # for i in range(len(self.Atomtypes)):
        #     ThisStruture=[self.SizeOfInputsPerType[i]]+MyStructure
        #     self.Structures.append(ThisStruture)

        self.ActFun="elu"
        self.MakeLastLayerConstant=False
        self.expand_existing_net(ModelName=model_name,MakeAllVariable=self.MakeAllVariable)

    def create_eval_data(self, geometries, NoBatches=True):
        """Converts the geometries in compatible format and prepares the data
        for evaluation.

        Args:
            geometries (list): List of geometries
            NoBatches (bool): Specifies if the data is split into differnt
                batches or only consits of a single not randomized batch.
        """
        dummy_energies = [0] * len(geometries)
        self.init_dataset(geometries, dummy_energies,
                          TakeAsReference=self.CalcDatasetStatistics)

        self.EvalData = self.get_data(NoBatches=True)

        return self.EvalData

    def _get_data_batch(self, BatchSize=100, NoBatches=False):
        """Creates a data batch by drawing a random sample out of the dataset
        The symmetry function vector is then normalized.

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
        ForceFieldInput=[]
        ForceFieldDerivatives=[]
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
                BatchSize = len(self._AllGVectors)

            EnergyData = _np.empty((BatchSize, 1))
            ForceData = _np.empty(
                (BatchSize, sum(self.NumberOfAtomsPerType) * 3))

            # Create a list with all possible random values
            ValuesForDrawingSamples = list(
                range(0, len(self._AllGVectors)))

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

                GeomData.append(self._AllGVectors[MyNr])
                EnergyData[i] = self._DataSet.energies[MyNr]
                if len(self._DataSet.forces) > 0:
                    ForceData[i] = [f for atom in self._DataSet.forces[MyNr]
                                    for f in atom]
                if self.UseForce:
                    GDerivativesInput.append(self._AllGDerivatives[MyNr])

                if self.IsPartitioned:
                    ForceFieldInput.append(self._AllForceFieldInputs[MyNr])
                    if self.UseForce:
                        ForceFieldDerivatives.append(self._AllForceFieldDerivatives[MyNr])

            #Sort data into batches
            Inputs, GDerivativesInput, Normalization,FFInputs,FFDerivatives = self._sort_and_normalize_data(
                BatchSize, GeomData, GDerivativesInput,ForceFieldInput,ForceFieldDerivatives)

            if self.UseForce:
                if self.IsPartitioned:
                    return Inputs, EnergyData, GDerivativesInput, Normalization, ForceData, FFInputs,FFDerivatives
                else:
                    return Inputs, EnergyData, GDerivativesInput, Normalization, ForceData
            else:
                if self.IsPartitioned:
                    return Inputs,EnergyData,[],[],[],FFInputs
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

            EnergyDataSetLength = len(self._AllGVectors)
            SetLength = int(EnergyDataSetLength * CoverageOfSetInPercent / 100)

            if not NoBatches:
                if BatchSize > len(self._AllGVectors) / 2:
                    BatchSize = int(len(self._AllGVectors) / 2)
                    print("Shrunk batches to size:" + str(BatchSize))
                NrOfBatches = max(1, int(round(SetLength / BatchSize, 0)))
            else:
                NrOfBatches = 1
            if self.TextOutput:
                print("Creating and normalizing " +
                      str(NrOfBatches) + " batches...")
            for i in range(0, NrOfBatches):
                Batches.append(self._get_data_batch(BatchSize, NoBatches))
                if not NoBatches:
                    if i % max(int(NrOfBatches / 10), 1) == 0 and self.TextOutput:
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

        if self._DataSet is None:
            raise Exception(
                    "No dataset initialized please call\
                    init_dataset first!")
        if self._SymmFunSet is None:
            raise Exception(
                    "No symmetry function set initialized please call\
                    create_symmetry_functions or init_dataset first!")


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

    def cost_for_network(self,Prediction, ReferenceValue, Type):
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
            Cost = _tf.losses.mean_squared_error(ReferenceValue,
                                                 Prediction)  # 0.5 * _tf.reduce_sum((Prediction - ReferenceValue)**2)
        elif Type == "Adaptive_1":
            epsilon = 10e-9
            Cost = 0.5 * _tf.reduce_sum((Prediction - ReferenceValue) ** 2
                                        * (_tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)) - 0.5)
                                        + (0.5 + _tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)))
                                        * _tf.pow(_tf.abs(Prediction - ReferenceValue + epsilon), 1.25))
        elif Type == "Adaptive_2":
            epsilon = 10e-9
            Cost = 0.5 * _tf.reduce_sum((Prediction - ReferenceValue) ** 2
                                        * (_tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)) - 0.5)
                                        + (0.5 + _tf.sigmoid(_tf.abs(Prediction - ReferenceValue + epsilon)))
                                        * _tf.abs(Prediction - ReferenceValue + epsilon))

        return Cost

    def _atomic_cost_function(self):
        """The atomic cost function consists of multiple parts which are each
        represented by a tensor.
        The main part is the energy cost.
        The reqularization and the force cost is optional.

        Returns:
            A tensor which is the sum of all costs"""

        self._TotalEnergy, AllEnergies = self._Net.energy_of_all_atomic_networks()

        self._EnergyCost = self.cost_for_network(
            self._TotalEnergy, self._OutputLayer, self.CostFunType)
        Cost = self._EnergyCost

        # add force cost
        if self.UseForce:
            self._OutputForce, AllForces = self._Net.force_of_all_atomic_networks(
                self)
            self._ForceCost = self.ForceCostParam* _tf.divide(
                self.cost_for_network(
                    self._OutputForce, self._OutputLayerForce, self.CostFunType), sum(
                    self.NumberOfAtomsPerType))
            Cost += self._ForceCost

        trainableVars = _tf.trainable_variables()
        regVars=[]
        for var in trainableVars:
            shape=var.get_shape()
            if shape[-1]!=1:
                regVars.append(var)
        if self.Regularization == "L1":

            l1_regularizer = _tf.contrib.layers.l1_regularizer(
                scale=self.RegularizationParam, scope=None)
            self._RegLoss = _tf.contrib.layers.apply_regularization(
                l1_regularizer, regVars)
            Cost += self._RegLoss
        elif self.Regularization == "L2":
            l2_regularizer=_tf.contrib.layers.l2_regularizer(
                    scale=self.RegularizationParam, scope=None)
            self._RegLoss = _tf.contrib.layers.apply_regularization(
                l2_regularizer, regVars)
            Cost += self._RegLoss

        # Create tensor for energy difference calculation
        self._dE_Fun = _tf.abs(self._TotalEnergy - self._OutputLayer)

        return Cost

    def _sort_and_normalize_data(self, BatchSize, GData, GDerivativesData=[],FFData=[],FFDerivativesData=[]):
        """Normalizes the input data.

        Args:
            BatchSize (int): Specifies the number of data points per batch.
            GeomData (list): Raw G vector data
            ForceData (list): (Optional) Raw derivatives of input vector

        Returns:
            Inputs: Normalized inputs
            DerInputs: If GDerivativesData is available a list of numpy array is returned
                    ,else an empty list is returned."""

        Inputs = []
        DerInputs = []
        Norm = []
        FFInputs = []
        FFDerivatives = []
        ct = 0
        TotalNrAtoms=sum(self.NumberOfAtomsPerType)
        #try:
        for NrAtoms in self.NumberOfAtomsPerType:

            for i in range(NrAtoms):
                Inputs.append(
                    _np.zeros((BatchSize, self.SizeOfInputsPerAtom[ct])))
                #If force data is available
                if len(GDerivativesData) > 0:
                    DerInputs.append(_np.zeros(
                        (BatchSize, self.SizeOfInputsPerAtom[ct], 3 * TotalNrAtoms)))
                    Norm.append(
                        _np.zeros((BatchSize, self.SizeOfInputsPerAtom[ct])))
                #If is partitioned network
                if len(FFData)>0:
                    FFInputs.append(
                        _np.zeros((BatchSize, 3*len(self.Atomtypes))))
                if len(FFDerivativesData)>0:
                    FFDerivatives.append(
                            _np.zeros((BatchSize, 3*len(self.Atomtypes),3*TotalNrAtoms)))
                # exclude nan values
                for j in range(0, len(GData)): #j = geometry number, ct = atom number
                    temp = _np.divide(
                        _np.subtract(
                            GData[j][ct], self._MeansOfDs), _np.sqrt(
                        self._VarianceOfDs))

                    Inputs[ct][j] = _np.nan_to_num(temp)
                    if len(GDerivativesData) > 0:
                        try:
                            DerInputs[ct][j] = GDerivativesData[j][ct]
                            Norm[ct] = _np.tile(_np.divide(
                                1, _np.sqrt(self._VarianceOfDs)), (BatchSize, 1))
                        except:
                            raise ValueError("Wrong number of atoms per type")
                    if len(FFData)>0:
                        FFInputs[ct][j]=FFData[j][ct]

                    if len(FFDerivativesData)>0:
                        FFDerivatives[ct][j]=FFDerivativesData[j][ct]

                ct += 1
        #except:
        #    raise ValueError("Atomtypes or NumberOfAtomsPerType do not match the loaded model!")

        return Inputs, DerInputs, Norm,FFInputs,FFDerivatives

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
        self.GlobalCostCriterion = 0
        self.Global_dE_Criterion = 0
        self.GlobalRegularization = "L2"
        self.GlobalRegularizationParam = 0.0001
        self.GlobalOptimizer = "Adam"
        self.GlobalTrainingCosts = []
        self.GlobalValidationCosts = []
        self.GlobalDE=[]
        self.GlobalMinOfOut = 0
        self.MakePlots = False
        self.IsPartitioned = False
        self.GlobalSession = _tf.InteractiveSession()
        self.SavingDirectory=""
        self.PESCheck=None

    def initialize_multiple_instances(self,MakeAllVariable=True):
        """Initializes all instances with the same parameters."""

        Execute = True
        if len(self.TrainingInstances) == 0:
            Execute = False
            print("No training instances available!")

        if Execute:
            # Initialize all instances with same settings
            for i in range(len(self.TrainingInstances)):
                self.TrainingInstances[i].Multiple = True
                self.TrainingInstances[i].Epochs = self.EpochsPerCycle
                self.TrainingInstances[i].MakeAllVariable = MakeAllVariable
                self.TrainingInstances[i].Structures = self.GlobalStructures
                self.TrainingInstances[i]._Session = self.GlobalSession
                self.TrainingInstances[i].MakePlots = False
                self.TrainingInstances[i].ActFun = "elu"
                self.TrainingInstances[i].CostCriterion = 0
                self.TrainingInstances[i].dE_Criterion = 0
                self.TrainingInstances[i].IsPartitioned = self.IsPartitioned
                self.TrainingInstances[i].WeightType = "truncated_normal"
                self.TrainingInstances[i].LearningRate = self.GlobalLearningRate
                self.TrainingInstances[i].OptimizerType = self.GlobalOptimizer
                self.TrainingInstances[i].Regularization = self.GlobalRegularization
                self.TrainingInstances[i].RegularizationParam = self.GlobalRegularizationParam
                self.TrainingInstances[i].TextOutput=False
                # Clear unnecessary data
                self.TrainingInstances[i]._DataSet.geometries = []
                self.TrainingInstances[i]._DataSet.Energies = []
                self.TrainingInstances[i]._DataSet.Forces = []
                self.TrainingInstances[i].Batches = []
                self.TrainingInstances[i].AllGeometries = []


    def set_session(self):
        """Sets the session of the currently trained instance to the
        global session"""

        for i in range(len(self.TrainingInstances)):
            self.TrainingInstances[i]._Session = self.GlobalSession

    def train_multiple_instances(self, StartModelName=None):
        """Trains each instance for EpochsPerCylce epochs then uses the resulting network
        as a basis for the next training instance.
        Args:
            StartModelName (str): Path to a saved model."""

        print("Startet multiple instance training!")
        ct = 0
        LastStepsModelData = list()
        for i in range(0, self.GlobalEpochs):
            for i in range(len(self.TrainingInstances)):
                if ct == 0:
                    if StartModelName is not None:
                        self.TrainingInstances[i].expand_existing_net(ModelName=StartModelName,load_statistics=False)
                    else:
                        self.TrainingInstances[i].make_and_initialize_network()
                else:
                    self.TrainingInstances[i].expand_existing_net(ModelData=LastStepsModelData)

                LastStepsModelData = self.TrainingInstances[i].start_batch_training()
                _tf.reset_default_graph()
                self.TrainingInstances[i]._Session.close()
                self.GlobalSession = _tf.InteractiveSession()
                self.set_session()
                self.GlobalTrainingCosts += self.TrainingInstances[i].OverallTrainingCosts
                self.GlobalValidationCosts += self.TrainingInstances[i].OverallValidationCosts
                self.GlobalDE += self.TrainingInstances[i].DeltaE
                if ct % max(int((self.GlobalEpochs * len(self.TrainingInstances)) / 100),
                            1) == 0 or i == (self.GlobalEpochs - 1):
                    if self.MakePlots:

                        if ct == 0:
                            fig, ax, TrainingCostPlot, ValidationCostPlot, RunningMeanPlot = _initialize_cost_plot(
                                self.GlobalTrainingCosts, self.GlobalValidationCosts)
                            de_fig, de_ax, de_plot = _initialize_delta_e_plot(self.GlobalDE)
                        else:
                            if self.PESCheck != None:
                                self.PESCheck.pes_check()
                            _update_cost_plot(
                                fig,
                                ax,
                                TrainingCostPlot,
                                self.GlobalTrainingCosts,
                                ValidationCostPlot,
                                self.GlobalValidationCosts,
                                RunningMeanPlot)
                            _update_delta_e_plot(self.GlobalDE, de_plot, de_fig, de_ax)

                    # Finished percentage output
                    print(str(100 * ct / (self.GlobalEpochs *
                                          len(self.TrainingInstances))) + " %")
                    if not _os.path.exists(self.SavingDirectory):
                        _os.makedirs(self.SavingDirectory)
                    _np.save(self.SavingDirectory + "/trained_variables",
                             [LastStepsModelData[0],
                              self.TrainingInstances[i]._MeansOfDs,
                              self.TrainingInstances[i]._VarianceOfDs,
                              self.TrainingInstances[i]._MinOfOut,
                              self.TrainingInstances[i].Rs,
                              self.TrainingInstances[i].R_Etas,
                              self.TrainingInstances[i].Etas,
                              self.TrainingInstances[i].Lambs,
                              self.TrainingInstances[i].Zetas,
                              self.TrainingInstances[i].NumberOfRadialFunctions,
                              self.TrainingInstances[i].Cutoff])
                ct = ct + 1
                # Abort criteria
                if self.GlobalTrainingCosts <= self.GlobalCostCriterion and \
                self.GlobalValidationCosts <= self.GloablCostCriterion or \
                self.TrainingInstances[i].DeltaE < self.Global_dE_Criterion:

                    if self.GlobalValidationCosts != 0:
                        print("Reached Criterion!")
                        print(
                            "Cost= " + str((self.GlobalTrainingCosts + \
                                            self.GlobalValidationCosts) / 2))
                        print("delta E = " + str(self.TrainingInstances[i].DeltaE[-1]) + " eV")
                        print("Epoch = " + str(i))
                        print("")

                    else:
                        print("Reached Criterion!")
                        print("Cost= " + str(self.GlobalTrainingCosts))
                        print("delta E = " + str(self.TrainingInstances[i].DeltaE[-1]) + " eV")
                        print("Epoch = " + str(i))
                        print("")

                    print("Training finished")
                    break

                if i == (self.GlobalEpochs - 1):
                    print("Training finished")
                    print("delta E = " + str(self.TrainingInstances[i].DeltaE[-1]) + " eV")
                    print("Epoch = " + str(i))
                    print("")


class Deprecated(object):

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


    def _convert_standard_to_partitioned_net(self):
        """Converts a standard (Behler) network to the force-field part of  a
        partitioned network.

        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""

        WeightData, BiasData,_ = self._Net.get_weights_biases_from_data(
            self.TrainedVariables)
        OutWeights = []
        OutBiases = []
        for i in range(0, len(self.TrainedVariables)):
            Network = self.TrainedVariables[i]
            DataStructWeights = _PartitionedWeights()
            DataStructBiases = _PartitionedBiases()
            for j in range(0, len(Network)):
                Weights = WeightData[i][j]
                Biases = BiasData[i][j]
                DataStructWeights.ANNWeights.append(Weights)
                DataStructBiases.ANNBiases.append(Biases)

            OutWeights.append(DataStructWeights)
            OutBiases.append(DataStructBiases)

        return OutWeights, OutBiases
