#model training script
import sys 
from NeuralNetworks import NeuralNetworkUtilities as _NN

#Get input
plots=False
learning_rate=0.001
epochs=5000
data_file=""
force=False
e_unit="Ry"
dist_unit="A"
load_model=True
model=""
model_dir="save_no_name"

for i,arg in enumerate(sys.argv):
    if "-input" in arg:
        data_file=sys.argv[i+1]
    if "-output" in arg:
        model_dir=sys.argv[i+1]
    if "-epochs" in arg:
        epochs=int(sys.argv[i+1])
    if "-force" in arg:
        force=bool(sys.argv[i+1])
    if "-load_model" in arg:
        load_model=bool(sys.argv[i+1])
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
  
if data_file=="":
    print("Please specify a MD file")
    print("Option: -input x")
    exit


#Load trainings instance
Training=_NN.AtomicNeuralNetInstance()
Training.UseForce=force
#Default symmetry function set
Training.NumberOfRadialFunctions=25
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]   
#Read file
Training.read_qe_md_files(data_file,e_unit,dist_unit)
#Default trainings settings
for i in range(len(Training.Atomtypes)):
    Training.Structures.append([Training.SizeOfInputsPerType[i],100,100,40,20,1])


Training.Dropout=[0,0,0,0,0]
Training.RegularizationParam=0.001

Training.LearningRate=learning_rate
Training.CostCriterium=0
Training.dE_Criterium=0.02

Training.Epochs=epochs
Training.ForceCostParam=0.001
Training.MakePlots=plots
Training.ActFun="elu"
Training.CostFunType="Adaptive_2"
Training.OptimizerType="Adam"
Training.SavingDirectory=model_dir
Training.MakeLastLayerConstant=False
Training.MakeAllVariable=True

if load_model:
    #Load pretrained net
    if model=="":
        Training.expand_existing_net(ModelName="pretraining_"+str(len(Training.Atomtypes))+"_species/trained_variables")
    else:
        Training.expand_existing_net(ModelName=model+"/trained_variables")
else:
    Training.make_and_initialize_network()

#Create batches
batch_size=len(Training._DataSet.energies)/50 
Training.make_training_and_validation_data(batch_size,90,10)


#Start training
Training.start_batch_training()