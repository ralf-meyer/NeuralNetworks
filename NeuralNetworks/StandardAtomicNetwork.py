import tensorflow as _tf
from NeuralNetworkUtilities import _connect_layers,\
    _construct_hidden_layer,\
    _construct_input_layer,\
    _construct_not_trainable_layer,\
    _construct_pass_through_weights
import numpy as _np
import multiprocessing as _multiprocessing

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

        Structures = []
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

        F = None
        Fi = []
        for i in range(0, len(self.AtomicNNs)):
            AtomicNet = self.AtomicNNs[i]
            Type = AtomicNet[0]
            G_Input = AtomicNet[2]
            dGij_dxk = AtomicNet[3]
            norm = AtomicNet[4]
            dGij_dxk_t = _tf.transpose(dGij_dxk, perm=[0, 2, 1])
            Gradient = _tf.gradients(NetInstance._TotalEnergy, G_Input)
            dEi_dGij_n = _tf.multiply(Gradient, norm)
            dEi_dGij = _tf.reshape(
                dEi_dGij_n, [-1, NetInstance.SizeOfInputsPerType[Type], 1])
            mul = _tf.matmul(dGij_dxk_t, dEi_dGij)
            dim_red = _tf.reshape(mul,
                                  [-1,sum(NetInstance.NumberOfAtomsPerType) * 3])

            if i == 0:
                F = dim_red
            else:
                F += dim_red
            Fi.append(dim_red)

        F=_tf.scalar_mul(-1,F)
        F=_tf.where(_tf.is_finite(F),F,_tf.zeros_like(F))
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
        AllEnergies = []

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

    def get_trained_variables(self, Session,atom_types=[]):
        """Prepares the data for saving.
        It gets the weights and biases from the session.

        Returns:
            NetworkData (list): All the network parameters as a list"""
        NetworkData = []

        for i,HiddenLayers in enumerate(self.VariablesDictionary):
            #print("Saving species "+str(atom_types[i])+"...")
            Layers = []
            for j in range(0, len(HiddenLayers)):
                Weights = Session.run(HiddenLayers[j][0])
                Biases = Session.run(HiddenLayers[j][1])
                Layers.append([Weights, Biases,atom_types[i]])
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
        Layers, Data = self.prepare_data_environment(GData, OutputLayer=None,
                                                     OutData=[],
                                                     OutputLayerForce=None,
                                                     GDerivativesInput=[],
                                                     ForceOutput=[],
                                                     AppendForce=False)
        for i in range(0, len(self.AtomicNNs)):
            AtomicNetwork = self.AtomicNNs[i]
            Energy += self.evaluate(AtomicNetwork[1], [Layers[i]], Data[i])

        return Energy

    def get_weights_biases_from_data(self, TrainedVariables):
        """Reads out the saved network data and sorts them into
        weights and biases.

        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""

        loaded_structure=[]
        Weights = []
        Biases = []
        for i in range(0, len(TrainedVariables)):
            NetworkData = TrainedVariables[i]
            ThisWeights = []
            ThisBiases = []
            sub_struct=[]
            for j in range(0, len(NetworkData)):
                ThisWeights.append(NetworkData[j][0])
                ThisBiases.append(NetworkData[j][1])
                sub_struct.append(NetworkData[j][0].shape[0])
            sub_struct.append(NetworkData[j][0].shape[1])
            Weights.append(ThisWeights)
            Biases.append(ThisBiases)
            loaded_structure.append(sub_struct)

        return Weights, Biases,loaded_structure

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
                            _tf.float64, shape=[None, NrInputs, 3 * sum(NetInstance.NumberOfAtomsPerType)])
                        Normalization = _tf.placeholder(
                            _tf.float64, shape=[None, NrInputs])
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
        AllHiddenLayers = []
        AtomicNNs = []
        self.AtomicNNs=[]
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

        # create layers for the different atom types
        for i in range(0, len(NetInstance.Structures)):
            if len(NetInstance.Dropout) > i:
                Dropout = NetInstance.Dropout[i]
            else:
                Dropout = NetInstance.Dropout[-1]
            # Make hidden layers
            HiddenLayers = []
            Structure = NetInstance.Structures[i]
            if len(NetInstance._WeightData) != 0: #if a net was loaded
                RawBias = NetInstance._BiasData[i]
                for j in range(1, len(Structure)):
                    NrIn = Structure[j - 1]
                    NrHidden = Structure[j]

                    if j == len(Structure) - \
                            1 and NetInstance.MakeLastLayerConstant: #if last layer has to be set constant
                                                                     #(pretraining)
                        HiddenLayers.append(_construct_not_trainable_layer(
                            NrIn, NrHidden, NetInstance._MinOfOut))
                    else:
                        if j >= len(NetInstance._WeightData[i])\
                                and NetInstance.MakeLastLayerConstant:#if new layers which are not part of the
                                                                      # loaded model, pass through values
                            tempWeights, tempBias = _construct_hidden_layer(NrIn, NrHidden, NetInstance.WeightType,
                                                                            [], NetInstance.BiasType, [],
                                                                            True, Mean=NetInstance.InitMean,
                                                                            Stddev=NetInstance.InitStddev)

                            tempWeights=_construct_pass_through_weights(tempWeights,OldBiasNr)

                            HiddenLayers.append([tempWeights, tempBias])
                        else:
                            if j <= len(RawBias): #if there is old net data available fill in the weight data
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
                                        NetInstance.MakeAllVariable,
                                        Stddev=NetInstance.InitStddev,
                                        Mean=NetInstance.InitMean))
                            else:#if the new net is deeper then the loaded one add a trainable layer
                                HiddenLayers.append(
                                    _construct_hidden_layer(
                                        NrIn,
                                        NrHidden,
                                        NetInstance.WeightType,
                                        [],
                                        NetInstance.BiasType,
                                        MakeAllVariable=True,
                                        Stddev=NetInstance.InitStddev,
                                        Mean=NetInstance.InitMean))

            else:#if no net was loaded
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
                                NetInstance.BiasType,
                                MakeAllVariable=NetInstance.MakeAllVariable,
                                Stddev=NetInstance.InitStddev,
                                Mean=NetInstance.InitMean
                            ))

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
                    Weights = HiddenLayers[l][0]
                    Biases = HiddenLayers[l][1]
                    if l == len(HiddenLayers) - 1:
                        Network = _connect_layers(
                            Network, Weights, Biases, "none", NetInstance.ActFunParam, Dropout)
                    else:
                        Network = _connect_layers(
                            Network, Weights, Biases, NetInstance.ActFun, NetInstance.ActFunParam, Dropout)

                if NetInstance.UseForce:
                    InputForce = _tf.placeholder(
                        _tf.float64, shape=[None, NrInputs, 3 * sum(NetInstance.NumberOfAtomsPerType)])
                    Normalization = _tf.placeholder(
                        _tf.float64, shape=[None, NrInputs])
                    AtomicNNs.append(
                        [i, Network, InputLayer, InputForce, Normalization])
                else:
                    AtomicNNs.append([i, Network, InputLayer])

        self.AtomicNNs = AtomicNNs
        self.VariablesDictionary = AllHiddenLayers
