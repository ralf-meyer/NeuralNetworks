#model pretraining script
import sys 
from NeuralNetworks import NeuralNetworkUtilities as _NN
from NeuralNetworks import check_pes

#Get input
plots=False
learning_rate=0.01
epochs=20000
data_file=""
nr_species=0
e_unit="Ry"
dist_unit="A"
force=False
load_model=False
model=""


def str2bool(v):

  return v.lower() in ("yes", "true", "t", "1")

for i,arg in enumerate(sys.argv):
    if "-input" in arg:
        data_file=sys.argv[i+1]
    if "-output" in arg:
        model_dir=sys.argv[i+1]
    if "-epochs" in arg:
        epochs=int(sys.argv[i+1])
    if "-force" in arg:
        force=str2bool(sys.argv[i+1])
    if "-load_model" in arg:
        load_model=str2bool(sys.argv[i+1])
    if "-v" in arg:
        plots=True
    if "-lr" in arg:
        learning_rate = float(sys.argv[i+1])
    if "-e_unit" in arg:
        e_unit=sys.argv[i+1]
    if "-dist_unit" in arg:
        dist_unit=sys.argv[i+1]
    if "-model" in arg:
        model=sys.argv[i+1]
    if "-source" in arg:
        source = sys.argv[i + 1]
    if "-data_percentage" in arg:
        percentage_of_data = float(sys.argv[i + 1])
    if "-nr_species" in arg:
        nr_species = int(sys.argv[i + 1])

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
Training.UseForce=force
#Default symmetry function set
#Training.NumberOfRadialFunctions=25
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.2,1,10]#[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.01]
Training.Rs=[1,1.8,2,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4,4.2,4.4,5,7]
Training.R_Etas=[0.1,0.3,0.8,0.8,0.8,3,3,3,3,3,2,0.8,0.8,0.8,0.8,0.3,0.1]
#Read files
for i in range(nr_species):
    Training.Atomtypes.append("X"+str(i+1))
Training.read_qe_md_files(data_file,e_unit,dist_unit)
#Create batches
batch_size=len(Training._DataSet.energies)/50 
Training.make_training_and_validation_data(batch_size,90,10)

#Default trainings settings
for i in range(nr_species):
    Training.Structures.append([Training.SizeOfInputsPerType[i],80,60,1])

#Dropout and regularization for generalizing the net
Training.Dropout=[0,0.5,0]
Training.RegularizationParam=0.1
Training.InitStddev=0.1
Training.HiddenType="truncated_normal"
Training.ActFun="elu"
Training.LearningRate=learning_rate
Training.LearningDecayEpochs=10000
Training.Epochs=epochs
Training.ForceCostParam=0.0001
Training.MakePlots=plots
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.CostFunType="Adaptive_2"
Training.LearningRateType="exponential_decay"
Training.SavingDirectory="../data/pretraining_"+str(nr_species)+"_species"
Training.LearningDecayEpochs=100
Training.MakeLastLayerConstant=True
#Training.PESCheck=check_pes.PES(Training.SavingDirectory)
if load_model:
    #Load pretrained net
    try:
        if model=="":
            Training.expand_existing_net(ModelName="../data/pretraining_"+str(nr_species)+"_species/trained_variables")
        else:
            Training.expand_existing_net(ModelName=model+"/trained_variables")
    except:
        raise IOError("Model not found, please specify model directory via -model x")
else:
    Training.make_and_initialize_network()


#Start pre-training
Training.start_batch_training()