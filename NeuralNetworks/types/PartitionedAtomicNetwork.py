import tensorflow as _tf
from NeuralNetworks.types.StandardAtomicNetwork import _StandardAtomicNetwork
from NeuralNetworks.types.GeneralAtomicNetwork import _AtomicNetwork


class _PartitionedAtomicNetwork(_AtomicNetwork):

    def __init__(self):
        self.AtomicNNs = _PartitionedNetwork()
        self.VariablesDictionary = _PartitionedNetworkDictionary()

    def get_structure_from_data(self, TrainedData):
        Structures = []
        for i in range(0, len(TrainedData[0])):
            for j in range(0,len(TrainedData[i])):
                ThisNetwork = TrainedData[i][j]
                Structure = []
                for k in range(0, len(ThisNetwork)):
                    Weights = ThisNetwork[k][0]
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
        TotalNrOfAtoms = sum(NetInstance.NumberOfAtomsPerType)
        #Force of ANNs
        for i in range(0, len(self.AtomicNNs.ANNets)):
            AtomicNet = self.AtomicNNs.ANNets[i]
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
                                  [-1,TotalNrOfAtoms * 3])

            if i == 0:
                F = dim_red
            else:
                F += dim_red
            Fi.append(dim_red)
        #Force of force fields
        for j in range(0, len(self.AtomicNNs.ForceFieldNets)):
            FFNet = self.AtomicNNs.ForceFieldNets[j]
            Type = FFNet[0]
            FF_Input = FFNet[2]
            d_FF = FFNet[3]
            d_FF_t = _tf.transpose(d_FF, perm=[0, 2, 1])
            Gradient = _tf.gradients(NetInstance._TotalEnergy,FF_Input)
            dEi_dFF = _tf.reshape(
                d_FF_t, [-1, 3*len(NetInstance.Atomtypes), 1])
            mul = _tf.matmul(dGij_dxk_t, dEi_dGij)
            dim_red = _tf.reshape(mul,
                                  [-1,TotalNrOfAtoms * 3])

            F += dim_red
            Fi.append(dim_red)

        F=_tf.scalar_mul(-1,F)
        F=_tf.where(_tf.is_finite(F),F,_tf.zeros_like(F))
        return F, Fi

    def energy_of_all_atomic_networks(self):
        """This function constructs the energy expression for
        the partitioned atomic networks.

        Returns:
            Prediction: A tensor which represents the energy output of
                        the partitioned network.
            AllEnergies: A list of tensors which represent the single Network
                        energy contributions."""

        Prediction = 0
        AllEnergies = []

        for Net in self.AtomicNNs.ANNets:
            # Get network data
            Network = Net[1]
            E_no_nan = _tf.where(_tf.is_nan(Network),
                                 _tf.zeros_like(Network), Network)
            # Get input data for network
            AllEnergies.append(E_no_nan)

        for Net in self.AtomicNNs.ForceFieldNets:
            # Get network data
            Network = Net[1]
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
        #get ANN
        ThisNet=[]
        for i,HiddenLayers in enumerate(self.VariablesDictionary.ANNDict):
            #print("Saving species "+str(atom_types[i])+"...")
            Layers = []
            for j in range(0, len(HiddenLayers)):
                Weights = Session.run(HiddenLayers[j][0])
                Biases = Session.run(HiddenLayers[j][1])
                Layers.append([Weights, Biases,atom_types[i]])
            ThisNet.append(Layers)

        NetworkData.append(ThisNet)
        #get FF
        ThisNet=[]

        for i,HiddenLayers in enumerate(self.VariablesDictionary.ForceFieldDict):
            #print("Saving species "+str(atom_types[i])+"...")
            Layers = []
            for j in range(0, len(HiddenLayers)):
                Weights = Session.run(HiddenLayers[j][0])
                Biases = Session.run(HiddenLayers[j][1])
                Layers.append([Weights, Biases,atom_types[i]])
            ThisNet.append(Layers)

        NetworkData.append(ThisNet)

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
        ForceLayer=False
        for i in range(len(self.AtomicNNs.ANNets)):
            #Layers for ANNs
            AtomicNetwork=self.AtomicNNs.ANNets[i]
            Layers.append(AtomicNetwork[2])
            if len(AtomicNetwork) > 3 and AppendForce:
                Layers.append(AtomicNetwork[3])  # G-derivatives
                Layers.append(AtomicNetwork[4])  # Normalization
                ForceLayer = True
        for j in range(len(self.AtomicNNs.ForceFieldNets)):
            #Layers for force field
            FFNetwork=self.AtomicNNs.ForceFieldNets[j]
            Layers.append(FFNetwork[2]) #FF data
            if len(FFNetwork) > 3 and AppendForce:
                Layers.append(FFNetwork[3]) #FF derivatives
                ForceLayer = True


        if OutputLayer is not None:
            Layers.append(OutputLayer)
            if ForceLayer and AppendForce:
                Layers.append(OutputLayerForce)

        return Layers

    def make_data_for_atomicNNs(self, GData, OutData=[], GDerivatives=[
    ], ForceOutput=[], Normalization=[], AppendForce=True,FFData=[],FFDerivatives=[]):
        """Sorts the symmetry function data for feeding.
            For training the output data also has to be added.
        Returns:
            CombinedData(list): Sorted data for the batch as a list."""
        # Sort data matching the placeholders
        CombinedData = []
        if AppendForce:
            for e,f,g in zip(GData,GDerivatives, Normalization):
                CombinedData.append(e)
                CombinedData.append(f)
                CombinedData.append(g)
        else:
            for Data in GData:
                CombinedData.append(Data)
        #force field data
        if AppendForce:
            for h,i in zip(FFData,FFDerivatives):
                CombinedData.append(h)
                CombinedData.append(i)
        else:
            for Data in FFData:
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
            AppendForce=True,
            FFData=[],
            FFDerivatives=[]):
        """Prepares the data and the input placeholders for the training in a
        partitioned NN.
        Returns:
            Layers (list):Sorted placeholders
            CombinedData (list):Sorted data as lists"""

        Layers = self.make_layers_for_atomicNNs(OutputLayer,
                                                    OutputLayerForce,
                                                    AppendForce)

        CombinedData = self.make_data_for_atomicNNs(GData,
                                                    OutData,
                                                    GDerivatives,
                                                    ForceOutput,
                                                    Normalization,
                                                    AppendForce,
                                                    FFData,
                                                    FFDerivatives)

        return Layers, CombinedData

    def evaluate_all_atomicnns(self, GData):
        """Evaluates the partitioned networks and calculates the energy as a
        sum of all network outputs.

        Returns:
            Energy (float):The predicted energy as a float."""
        Energy = 0
        Layers, Data = self.prepare_data_environment(GData,
                                                     OutputLayer=None,
                                                     OutData=[],
                                                     OutputLayerForce=None,
                                                     GDerivativesInput=[],
                                                     ForceOutput=[],
                                                     AppendForce=False,)
        for i in range(0, len(self.AtomicNNs.ANNets)):
            AtomicNetwork = self.AtomicNNs.ANNets[i]
            Energy += self.evaluate(AtomicNetwork[1], [Layers[i]], Data[i])

        for j in range(0, len(self.AtomicNNs.ForceFieldNets)):
            AtomicNetwork = self.AtomicNNs.ForceFieldNets[j]
            Energy += self.evaluate(AtomicNetwork[1],
                                    [Layers[len(self.AtomicNNs.ANNets)+j-1]],
                                    Data[len(self.AtomicNNs.ANNets)+j-1])

        return Energy

    def get_weights_biases_from_data(self, TrainedVariables):
        """Reads out the saved network data and sorts them into
        weights and biases.

        Returns:
            Weights (list):List of numpy arrays
            Biases (list):List of numpy arrays"""
        loaded_structure=[]
        Weights =_PartitionedWeights()
        Biases = _PartitionedBiases()

        #Get ANN data
        for ANN in TrainedVariables[0]:
            ThisWeights = []
            ThisBiases = []
            sub_struct = []
            for j in range(0, len(ANN)):
                ThisWeights.append(ANN[j][0])
                ThisBiases.append(ANN[j][1])
                sub_struct.append(ANN[j][0].shape[0])
            sub_struct.append(ANN[j][0].shape[1])
            Weights.ANNWeights.append(ThisWeights)
            Biases.ANNBiases.append(ThisBiases)
            loaded_structure.append(sub_struct)
        #Get FF data
        for FF in TrainedVariables[1]:
            ThisWeights = []
            ThisBiases = []
            for j in range(0, len(FF)):
                ThisWeights.append(FF[j][0])
                ThisBiases.append(FF[j][1])
            Weights.ForceFieldWeights.append(ThisWeights)
            Biases.ForceFieldBiases.append(ThisBiases)


        return Weights, Biases,loaded_structure


    def make_parallel_atomic_networks(self, NetInstance):
        """Creates the specified partitioned network with separate variable
        tensors for each atoms.(Only for evaluation)"""
        NInputs=len(NetInstance.Atomtypes)*3
        ANN = _StandardAtomicNetwork()
        # copy data for force field
        FFWeights = NetInstance._WeightData.ForceFieldWeights
        FFBiases = NetInstance._BiasData.ForceFieldBiases
        # Set atomic neural network weights and biases for creation
        NetInstance._WeightData = NetInstance._WeightData.ANNWeights
        NetInstance._BiasData = NetInstance._BiasData.ANNBiases
        # Create atomic neural network
        ANN.make_parallel_atomic_networks(NetInstance)
        self.AtomicNNs.ANNets = ANN.AtomicNNs
        self.VariablesDictionary.ANNData = ANN.VariablesDictionary
        # Create the force field nets
        FFNets = []
        VariablesFF = []
        for i,Type in enumerate(NetInstance.Atomtypes):
            WeightsForType=FFWeights[i]
            BiasesForType=FFBiases[i]
            for j in range(NetInstance.NumberOfAtomsPerType[i]):

                Weights, Biases = self._construct_hidden_layer(NInputs,
                                                          1,
                                                          WeightType="truncated_normal",
                                                          WeightData=WeightsForType[0],
                                                          BiasType="truncated_normal",
                                                          BiasData=BiasesForType[0],
                                                          MakeAllVariable=True)
                VariablesFF.append([Weights,Biases])
                self.VariablesDictionary.ForceFieldData.append(VariablesFF)
                Input=self._construct_input_layer(NInputs)
                Network=self._connect_layers(Input,Weights,Biases,ActFun="none")

                if NetInstance.UseForce:
                    InputForce = _tf.placeholder(
                        _tf.float64, shape=[None, NInputs, 3* sum(NetInstance.NumberOfAtomsPerType)])
                    Normalization = _tf.placeholder(
                        _tf.float64, shape=[None, NInputs])
                    FFNets.append(
                        [i, Network, Input, InputForce, Normalization])
                else:
                    FFNets.append([i, Network, Input])

        self.AtomicNNs.ForceFieldNets=FFNets

    def make_atomic_networks(self, NetInstance):
        """Creates the specified partitioned network."""
        ANN=_StandardAtomicNetwork()
        NInputs = len(NetInstance.Atomtypes) * 3
        #copy data for force field
        FFWeights = []
        FFBiases=[]

        if NetInstance._WeightData!=None:
            FFWeights=NetInstance._WeightData.ForceFieldWeights
            FFBiases=NetInstance._BiasData.ForceFieldBiases
            #Set atomic neural network weights and biases for creation
            NetInstance._WeightData=NetInstance._WeightData.ANNWeights
            NetInstance._BiasData=NetInstance._BiasData.ANNBiases
        #Create atomic neural network
        ANN.make_atomic_networks(NetInstance)
        self.AtomicNNs.ANNets=ANN.AtomicNNs
        self.VariablesDictionary.ANNDict=ANN.VariablesDictionary
        #Create the force field nets
        FFNets=[]
        VariablesFF=[]
        for i,Type in enumerate(NetInstance.Atomtypes):
            if len(FFWeights)>0:
                WeightsForType=FFWeights[i]
                BiasesForType=FFBiases[i]
                Weights,Biases = self._construct_hidden_layer(NInputs,
                                                        1,
                                                        WeightType="truncated_normal",
                                                        WeightData=WeightsForType[0],
                                                        BiasType="truncated_normal",
                                                        BiasData=BiasesForType[0],
                                                        MakeAllVariable=True)

                VariablesFF.append([Weights,Biases])
            else:
                Weights, Biases = self._construct_hidden_layer(NInputs,
                                                        1,
                                                        WeightType="truncated_normal",
                                                        WeightData=[],
                                                        BiasType="truncated_normal",
                                                        BiasData=[],
                                                        MakeAllVariable=True)

                VariablesFF.append([Weights, Biases])

            self.VariablesDictionary.ForceFieldDict.append(VariablesFF)

            for j in range(NetInstance.NumberOfAtomsPerType[i]):

                Input=self._construct_input_layer(NInputs)
                Network=self._connect_layers(Input,Weights,Biases,ActFun=None)

                if NetInstance.UseForce:
                    InputForce = _tf.placeholder(
                        _tf.float64, shape=[None, NInputs, 3* sum(NetInstance.NumberOfAtomsPerType)])
                    Normalization = _tf.placeholder(
                        _tf.float64, shape=[None, NInputs])
                    FFNets.append(
                        [i, Network, Input, InputForce])
                else:
                    FFNets.append([i, Network, Input])

        self.AtomicNNs.ForceFieldNets=FFNets

class _PartitionedNetwork(object):
    """This class is a container for the partitioned network data."""

    def __init__(self):

        self.ForceFieldNets = []
        self.ANNets = []

class _PartitionedWeights(object):
    """This class is a container for the partitioned weights data."""

    def __init__(self):

        self.ForceFieldWeights = []
        self.ANNWeights = []

class _PartitionedBiases(object):
    """This class is a container for the partitioned bias data."""

    def __init__(self):

        self.ForceFieldBiases=[]
        self.ANNBiases = []

class _PartitionedNetworkDictionary(object):
    """This class is a container for the partitioned network data."""

    def __init__(self):

        self.ForceFieldDict = []
        self.ANNDict = []