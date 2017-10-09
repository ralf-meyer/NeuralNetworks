#model training script
import sys 
from NeuralNetworks import NeuralNetworkUtilities as _NN
from NeuralNetworks import ReadQEData as _reader

#Get input
plots=False
learning_rate=0.0001
epochs=1500
data_file=""

for i,arg in enumerate(sys.argv):
    if "-input" in arg:
        data_file=sys.argv[i+1]
    if "-output" in arg:
        model_dir=sys.argv[i+1]
    if "-epochs" in arg:
        epochs=sys.argv[i+1]
    if "-v" in arg:
        plots=True
    if "-lr" in arg:
        learning_rate = sys.argv[i+1]
  
if data_file=="":
    print("Please specify a MD file")
    print("Option: -input x")
    exit

#Get quantum espresso md run parser
md_reader=_reader.QE_MD_Reader()
md_reader.E_conv_factor=13.605698066 #From Ry to ev
md_reader.Geom_conv_factor=1 #To Angstroem
md_reader.get_files(data_file)
#Read data
md_reader.read_all_files()
#Take minimum as zero
md_reader.calibrate_energy()


#Load trainings instance
Training=_NN.AtomicNeuralNetInstance()
#Default symmetry function set
Training.NumberOfRadialFunctions=25
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]   

#Load trainings data
Training.atomtypes=md_reader.atom_types
Training.init_dataset(md_reader.geometries,md_reader.e_pot_rel)

#Create batches
batch_size=len(md_reader.e_pot_rel)/50 
Training.make_training_and_validation_data(batch_size,70,30)

#Default trainings settings
for i in range(len(md_reader.atom_types)):
    Training.Structures.append([Training.SizeOfInputs[0],100,100,40,20,1])


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
Training.NumberOfAtomsPerType=md_reader.nr_atoms_per_type
#Load pretrained net
Training.expand_existing_net(ModelName="pretrained_"+md_reader.nr_atoms_per_type+"_species/trained_variables")
#Start training
Training.start_batch_training()