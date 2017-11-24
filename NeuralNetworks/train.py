#model training script
import sys 
from NeuralNetworks import NeuralNetworkUtilities as _NN
import os
from NeuralNetworks import check_pes


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
#Get input
plots=False
learning_rate=0.0001
epochs=5000
data_file=""
force=False
e_unit="Ry"
dist_unit="A"
load_model=True
model=""
model_dir="save_no_name"
source=""
percentage_of_data=100

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

if data_file=="":
    print("Please specify a MD file")
    print("Option: -input x")
    exit

if source == "":
    print("Please specify your source (QE = Quantum Espresso, LAMMPS=Lammps")
    print("Option: -source x")
    exit

print("Learning rate = "+str(learning_rate))
print("Epochs = "+str(epochs))
print("Energy unit = "+e_unit)
print("Distance unit = "+ dist_unit)
print("Training file : "+data_file)
print("Used percentage of data : "+str(percentage_of_data)+" %")
print("Force training = "+str(force))
print("Use model = "+str(load_model))
if load_model:
    print("Loaded model = "+model)
print("Save path : "+os.path.join(os.getcwd(),model_dir))


#Load trainings instance
Training=_NN.AtomicNeuralNetInstance()
Training.UseForce=force
#Default symmetry function set
Training.NumberOfRadialFunctions=25
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]   
#Read file
if source == "QE":
    Training.read_qe_md_files(data_file,e_unit,dist_unit,DataPointsPercentage=percentage_of_data)
else:
    Training.read_lammps_files(data_file,energy_unit=e_unit,dist_unit=dist_unit,DataPointsPercentage=percentage_of_data)

print("Starting training for:")
for i in range(0,len(Training.Atomtypes)):
    print(Training.Atomtypes[i]+" "+str(Training.NumberOfAtomsPerType[i]))

#Default trainings settings
for i in range(len(Training.Atomtypes)):
    Training.Structures.append([Training.SizeOfInputsPerType[i],80,60,40,20,1])


Training.Dropout=[0,0,0,0]
Training.RegularizationParam=0.1
Training.InitStddev=0.1
Training.LearningRate=learning_rate
Training.LearningDecayEpochs=100
Training.CostCriterium=0
Training.dE_Criterium=0.02
Training.WeightType="truncated_normal"
Training.BiasType="truncated_normal"
Training.Epochs=epochs
Training.ForceCostParam=0.001
Training.MakePlots=plots
Training.ActFun="elu"
Training.CostFunType="Adaptive_2"
Training.OptimizerType="Adam"
Training.SavingDirectory=model_dir
Training.MakeLastLayerConstant=False
Training.PESCheck=check_pes.PES(model_dir)

if load_model:
    Training.MakeAllVariable = True
    #Load pretrained net
    try:
        if model=="":
            Training.expand_existing_net(ModelName="../data/pretraining_"+str(len(Training.Atomtypes))+"_species",MakeAllVariable=Training.MakeAllVariable)
        else:
            Training.expand_existing_net(ModelName=model,MakeAllVariable=Training.MakeAllVariable)
    except:
        raise IOError("Model not found, please specify model directory via -model x")
else:
    Training.MakeAllVariable = True
    Training.make_and_initialize_network()

#Create batches
batch_size=len(Training._DataSet.energies)*(percentage_of_data/100)/50

Training.make_training_and_validation_data(batch_size,90,10)


#Start training
Training.start_batch_training()