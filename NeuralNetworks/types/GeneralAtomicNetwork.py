from __future__ import absolute_import, division, print_function
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import tensorflow as _tf
from tensorflow.python.framework import ops
import numpy as _np



class _AtomicNetwork(object):

    def dropout_selu(self,x, rate, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                     noise_shape=None, seed=None, name=None, training=False):
        """Dropout to a value with rescaling."""

        def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
            keep_prob = 1.0 - rate
            x = ops.convert_to_tensor(x, name="x")
            if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
                raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                 "range (0, 1], got %g" % keep_prob)
            keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
            keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

            alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
            alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

            if tensor_util.constant_value(keep_prob) == 1:
                return x

            noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
            random_tensor = keep_prob
            random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
            binary_tensor = math_ops.floor(random_tensor)
            ret = x * binary_tensor + alpha * (1 - binary_tensor)

            a = math_ops.sqrt(fixedPointVar / (
            keep_prob * ((1 - keep_prob) * math_ops.pow(alpha - fixedPointMean, 2) + fixedPointVar)))

            b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
            ret = a * ret + b
            ret.set_shape(x.get_shape())
            return ret

        with ops.name_scope(name, "dropout", [x]) as name:
            return utils.smart_cond(training,
                                    lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                    lambda: array_ops.identity(x))

    def _construct_input_layer(self,InputUnits):
        """Construct input for the NN.

        Args:
            InputUnits (int):Number of input units

        Returns:
            Inputs (tensor):The input placeholder
        """

        Inputs = _tf.cast(_tf.placeholder(_tf.float32, shape=[None, InputUnits]), dtype=_tf.float32)

        return Inputs

    def _construct_hidden_layer(self,
            PreviousLayerUnits,
            ThisHiddenUnits,
            WeightType=None,
            WeightData=[],
            BiasType=None,
            BiasData=[],
            MakeAllVariable=False,
            Mean=0.0,
            Stddev=1.0,
            i=0,
            j=0,
            include_histograms=False):
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
            i (int): Network number
            j (int): layer number
            include_histograms(bool):Flag signaling whether histograms
             should be included in the event file

        Returns:
            Weights (tensor): Weights tensor
            Biases (tensor):Biases tensor
        """
        if Stddev==0:
            Stddev=1/_np.sqrt(PreviousLayerUnits)
        with _tf.name_scope("t_"+str(i+1)+"_layer_"+str(j)):
            if len(WeightData) == 0:
                if WeightType is not None:
                    if WeightType == "zeros":
                        Weights = _tf.Variable(_tf.zeros(
                            [PreviousLayerUnits, ThisHiddenUnits],
                            dtype=_tf.float32), dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "ones":
                        Weights = _tf.Variable(_tf.ones(
                            [PreviousLayerUnits, ThisHiddenUnits],
                            dtype=_tf.float32), dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "fill":
                        Weights = _tf.Variable(_tf.fill(
                            [PreviousLayerUnits, ThisHiddenUnits],
                            dtype=_tf.float32),
                            dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "random_normal":
                        Weights = _tf.Variable(
                            _tf.random_normal(
                                [
                                    PreviousLayerUnits,
                                    ThisHiddenUnits],
                                mean=Mean,
                                stddev=Stddev,
                                dtype=_tf.float32),
                            dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "truncated_normal":
                        Weights = _tf.Variable(
                            _tf.truncated_normal(
                                [
                                    PreviousLayerUnits,
                                    ThisHiddenUnits],
                                mean=Mean,
                                stddev=Stddev,
                                dtype=_tf.float32),
                            dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "random_uniform":
                        Weights = _tf.Variable(_tf.random_uniform(
                            [PreviousLayerUnits, ThisHiddenUnits]), dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "random_shuffle":
                        Weights = _tf.Variable(_tf.random_shuffle(
                            [PreviousLayerUnits, ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "random_crop":
                        Weights = _tf.Variable(_tf.random_crop(
                            [PreviousLayerUnits, ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32,
                            name="variable")
                    elif WeightType == "random_gamma":
                        Weights = _tf.Variable(_tf.random_gamma(
                            [PreviousLayerUnits, ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32,
                            name="variable")
                    else:
                        # Assume random weights if no WeightType is given
                        Weights = _tf.Variable(_tf.random_uniform(
                            [PreviousLayerUnits, ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32,
                            name="variable")
                else:
                    # Assume random weights if no WeightType is given
                    Weights = _tf.Variable(_tf.random_uniform(
                        [PreviousLayerUnits, ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32,
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

                # if not MakeAllVariable:
                #     Biases = _tf.constant(BiasData, dtype=_tf.float32, name="bias")
                # else:
                Biases = _tf.Variable(BiasData, dtype=_tf.float32, name="bias")

            else:
                if BiasType == "zeros":
                    Biases = _tf.Variable(
                        _tf.zeros([ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32, name="bias")
                elif BiasType == "ones":
                    Biases = _tf.Variable(
                        _tf.ones([ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32, name="bias")
                elif BiasType == "fill":
                    Biases = _tf.Variable(
                        _tf.fill(
                            [ThisHiddenUnits],
                            BiasData,
                            dtype=_tf.float32),
                        dtype=_tf.float32,
                        name="bias")
                elif BiasType == "random_normal":
                    Biases = _tf.Variable(
                        _tf.random_normal(
                            [ThisHiddenUnits],
                            mean=Mean,
                            stddev=Stddev,
                            dtype=_tf.float32),
                        dtype=_tf.float32,
                        name="bias")
                elif BiasType == "truncated_normal":
                    Biases = _tf.Variable(
                        _tf.truncated_normal(
                            [ThisHiddenUnits],
                            mean=Mean,
                            stddev=Stddev,
                            dtype=_tf.float32),
                        dtype=_tf.float32,
                        name="bias")
                elif BiasType == "random_uniform":
                    Biases = _tf.Variable(_tf.random_uniform(
                        [ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32, name="bias")
                elif BiasType == "random_shuffle":
                    Biases = _tf.Variable(_tf.random_shuffle(
                        [ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32, name="bias")
                elif BiasType == "random_crop":
                    Biases = _tf.Variable(_tf.random_crop(
                        [ThisHiddenUnits], BiasData, dtype=_tf.float32), dtype=_tf.float32, name="bias")
                elif BiasType == "random_gamma":
                    Biases = _tf.Variable(_tf.random_gamma(
                        [ThisHiddenUnits], BiasData, dtype=_tf.float32), dtype=_tf.float32, name="bias")
                else:
                    Biases = _tf.Variable(_tf.random_uniform(
                        [ThisHiddenUnits], dtype=_tf.float32), dtype=_tf.float32, name="bias")

            if include_histograms:
                _tf.summary.histogram("t_"+str(i+1)+"_layer_"+str(j)+"_weights",Weights)
                _tf.summary.histogram("t_" + str(i + 1) + "_layer_" + str(j) + "_biases", Biases)
            return Weights, Biases

    def _construct_output_layer(self,OutputUnits):
        """#Constructs the output layer for the NN.

        Args:
            OutputUnits(int): Number of output units

        Returns:
            Outputs (tensor): Output placeholder
        """
        Outputs = _tf.placeholder(_tf.float32, shape=[None, OutputUnits])

        return Outputs

    def _construct_not_trainable_layer(self,NrInputs, NrOutputs, Min):
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

    def _construct_pass_through_weights(self,weights, size):
        """Constructs weights being all 1 in the main diagonal
        Args:
            weights(tensor):Weight tensor which is modified
            size(int): Size of weights as (size,size)
        Returns:
            weights(tensor):Modified weights"""
        indices = []
        values = []
        thisShape = weights.get_shape().as_list()
        if thisShape[0] == thisShape[1]:
            for q in range(0, size):
                indices.append([q, q])
                values += [1.0]

            delta = _tf.SparseTensor(
                indices, values, thisShape)
            weights = weights + \
                      _tf.sparse_tensor_to_dense(delta)

        return weights


    def _connect_layers(self,InputsForLayer, ThisLayerWeights,
                        ThisLayerBias, ActFun=None, FunParam=None,
                        Dropout=0,i=0,j=0,include_histograms=False):

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
            i (int): Network number
            j (int): layer number
            include_histograms (bool):Flag signaling whether histograms
             should be included in the event file

        Returns:
            Out (tensor): The connected tensor."""


        def morse(x):
            return 0.28707*(_tf.exp(-2* 0.601533*(x-0.25))-2*_tf.exp(- 0.601533*(x-0.25)))

        def morse_all_1(x):
            return 1*(_tf.exp(-2* 1*(x-1))-2*_tf.exp(- 1*(x-1)))

        def shift_morse(x):
            return  0.793876+2*(_tf.exp(-2* 1 *(x))-2*_tf.exp(- 1*(x)))

        if ActFun is not None:
            if ActFun == "sigmoid":
                with _tf.name_scope("net_"+str(i+1)+"_act_fun_"+str(j+1)):
                    Out = _tf.nn.sigmoid(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "tanh":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.tanh(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "relu":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.relu(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "relu6":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.relu6(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "crelu":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.crelu(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "elu":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.elu(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "selu":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.selu(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "softplus":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.softplus(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "dropout":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.dropout(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias, FunParam)
            elif ActFun == "bias_add":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                    Out = _tf.nn.bias_add(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias, FunParam)
            elif ActFun == "morse":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j + 1)):
                    Out=morse(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "shift_morse":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j + 1)):
                    Out=shift_morse(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "morse_all_1":
                with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j + 1)):
                    Out=morse_all_1(_tf.matmul(
                        InputsForLayer, ThisLayerWeights) + ThisLayerBias)
            elif ActFun == "none":
                with _tf.name_scope("E_"+str(i)):
                    Out = _tf.matmul(InputsForLayer, ThisLayerWeights) + ThisLayerBias
        else:
            with _tf.name_scope("E_" + str(i)):
                Out = _tf.matmul(InputsForLayer, ThisLayerWeights) + ThisLayerBias
        if Dropout != 0:
            # Apply dropout between layers
            with _tf.name_scope("net_" + str(i + 1) + "_act_fun_" + str(j+1)):
                if ActFun=="selu":
                    Out=self.dropout_selu(Out,1-Dropout)
                else:
                    Out = _tf.nn.dropout(Out, Dropout)
        if include_histograms:
            _tf.summary.histogram("net_" + str(i + 1) + "_act_fun_" + str(j+1),Out)
        return Out

    def make_feed_forward_neuralnetwork(self,
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
        InputLayer = self._construct_input_layer(NrInputs)
        # Make hidden layers
        HiddenLayers = list()
        for i in range(1, len(Structure)):
            NrIn = Structure[i - 1]
            NrHidden = Structure[i]
            HiddenLayers.append(self._construct_hidden_layer(
                NrIn, NrHidden, WeightType, WeightData[i - 1], BiasType,
                BiasData[i - 1]))

        # Make output layer
        OutputLayer = self._construct_output_layer(Structure[-1])

        # Connect input to first hidden layer
        FirstWeights = HiddenLayers[0][0]
        FirstBiases = HiddenLayers[0][1]
        InConnection = self._connect_layers(
            InputLayer, FirstWeights, FirstBiases, ActFun, ActFunParam, Dropout)
        Network = InConnection

        for j in range(1, len(HiddenLayers)):
            # Connect ouput of in layer to second hidden layer
            if j == 1:
                SecondWeights = HiddenLayers[j][0]
                SecondBiases = HiddenLayers[j][1]
                Network = self._connect_layers(
                    Network,
                    SecondWeights,
                    SecondBiases,
                    ActFun,
                    ActFunParam,
                    Dropout)
            else:
                Weights = HiddenLayers[j][0]
                Biases = HiddenLayers[j][1]
                Network = self._connect_layers(
                    Network, Weights, Biases, ActFun, ActFunParam, Dropout)

        return Network, InputLayer, OutputLayer



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


    def force_of_all_atomic_networks(self, NetInstance):
        """This function constructs the force expression for the atomic networks.

        Returns:
            A tensor which represents the force output of the network"""

    def energy_of_all_atomic_networks(self):
        """This function constructs the energy expression for
        the atomic networks.

        Returns:
            Prediction: A tensor which represents the energy output of
                        the partitioned network.
            AllEnergies: A list of tensors which represent the single Network
                        energy contributions."""


    def get_trained_variables(self, Session,atom_types=[]):
        """Prepares the data for saving.
        It gets the weights and biases from the session.

        Returns:
            NetworkData (list): All the network parameters as a list"""


    def make_layers_for_atomicNNs(
            self, OutputLayer=None, OutputLayerForce=None, AppendForce=True):
        """Sorts the input placeholders in the correct order for feeding.
        Each atom has a seperate placeholder which must be feed at each step.
        The placeholders have to match the symmetry function input.
        For training the output placeholder also has to be feed.

        Returns:
            Layers (list):All placeholders as a list."""

    def make_data_for_atomicNNs(self, GData, OutData=[], GDerivatives=[
    ], ForceOutput=[], Normalization=[], AppendForce=True):
        """Sorts the symmetry function data for feeding.
            For training the output data also has to be added.
        Returns:
            CombinedData(list): Sorted data for the batch as a list."""

    def prepare_data_environment(
            self,
            GData,
            OutputLayer=None,
            OutData=[],
            OutputLayerForce=None,
            GDerivatives=[],
            ForceOutput=[],
            Normalization=[],
            AppendForce=True,
            Placeholder1=[],
            Placeholder2=[]
            ):
        """Prepares the data and the input placeholders for the training in a NN.
        Returns:
            Layers (list):List of sorted placeholders
            Data (list): List of sorted data"""


    def evaluate_all_atomicnns(self, GData, AppendForce=False):
        """Evaluates the networks and calculates the energy as a sum of all network
        outputs.

        Returns:
            Energy(float): Predicted energy as a float."""

    def get_weights_biases_from_data(self, TrainedVariables):
        """Reads out the saved network data and sorts them into
        weights and biases.

        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""


    def make_parallel_atomic_networks(self, NetInstance):
        """Creates the specified network with separate varibale tensors
            for each atoms.(Only for evaluation)"""


    def make_atomic_networks(self, NetInstance):
        """Creates the specified network."""
