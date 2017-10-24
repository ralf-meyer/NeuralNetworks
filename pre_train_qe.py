#model pretraining script
import sys 
from NeuralNetworks import NeuralNetworkUtilities as _NN

#Get input
plots=False
learning_rate=0.005
epochs=20000
data_file=""
nr_species=0
for i,arg in enumerate(sys.argv):
    if "-input" in arg:
        data_file=sys.argv[i+1]
    if "-nr_species" in arg:
        nr_species=int(sys.argv[i+1])
    if "-epochs" in arg:
        epochs=sys.argv[i+1]
    if "-v" in arg:
        plots=True
    if "-lr" in arg:
        learning_rate = float(sys.argv[i+1])

if data_file=="":
    print("Please specify a MD file")
    print("Option: -input x")
    quit()
    

if nr_species==0:
    print("Please specify for how many species you pre-train the network!")
    print("Option: -nr_atoms x")
    quit()
else:
    print("Pre-training for "+str(nr_species)+" atom species...")


#Get instance
Training=_NN.AtomicNeuralNetInstance()
#Read files
for i in range(nr_species):
    Training.Atomtypes.append("H"+str(i+1))
Training.read_qe_md_files(data_file,"Ry")

#Default symmetry function set
Training.NumberOfRadialFunctions=25
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]   

#Create batches
batch_size=len(Training.Ds.energies)/50 
Training.make_training_and_validation_data(batch_size,70,30)

#Default trainings settings
for i in range(nr_species):
    Training.Structures.append([Training.SizeOfInputs[0],100,100,1])

#Dropout and regularization for generalizing the net
Training.Dropout=[0,0.5,0]
Training.RegularizationParam=0.1
Training.HiddenType="truncated_normal"
Training.ActFun="elu"
Training.LearningRate=learning_rate
Training.Epochs=epochs
Training.MakePlots=plots
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.CostFunType="Adaptive_2"
Training.LearningRateType="exponential_decay"
Training.SavingDirectory="pretraining_"+str(nr_species)+"_species"
Training.LearningDecayEpochs=100
Training.MakeLastLayerConstant=True
Training.make_and_initialize_network()

#Start pre-training
Training.start_batch_training()