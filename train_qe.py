#model training script
import sys 
from NeuralNetworks import NeuralNetworkUtilities as _NN

#Get input
plots=False
learning_rate=0.0001
epochs=1500
data_file=""
force=False
for i,arg in enumerate(sys.argv):
    if "-input" in arg:
        data_file=sys.argv[i+1]
    if "-output" in arg:
        model_dir=sys.argv[i+1]
    if "-epochs" in arg:
        epochs=sys.argv[i+1]
    if "-force" in arg:
        force=bool(sys.argv[i+1])
    if "-v" in arg:
        plots=True
    if "-lr" in arg:
        learning_rate = sys.argv[i+1]
  
if data_file=="":
    print("Please specify a MD file")
    print("Option: -input x")
    exit



#Load trainings instance
Training=_NN.AtomicNeuralNetInstance()
Training.UseForce=force
#Default symmetry function set
Training.NumberOfRadialFunctions=20
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]   
#Read file
Training.read_qe_md_files(data_file,"Ry",TakeAsReference=False)
#Default trainings settings
for i in range(len(Training.Atomtypes)):
    Training.Structures.append([Training.SizeOfInputsPerType[0],100,100,40,20,1])


Training.Dropout=[0,0,0,0,0]
Training.RegularizationParam=0.1

Training.LearningRate=learning_rate
Training.CostCriterium=0
Training.dE_Criterium=0

Training.Epochs=epochs
Training.MakePlots=plots
Training.ActFun="elu"
Training.CostFunType="Adaptive_2"
Training.OptimizerType="Adam"
Training.SavingDirectory=model_dir
Training.MakeLastLayerConstant=True
Training.MakeAllVariable=False
#Load pretrained net
Training.expand_existing_net(ModelName="pretrained_"+str(len(Training.Atomtypes))+"_species/trained_variables")

#Create batches
batch_size=len(Training._DataSet.energies)/50 
Training.make_training_and_validation_data(batch_size,70,30)


#Start training
Training.start_batch_training()