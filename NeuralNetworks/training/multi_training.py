#model training script
import sys
from NeuralNetworks import NeuralNetworkUtilities as _NN
import os
from NeuralNetworks import check_pes
import numpy as _np
import matplotlib

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
#Get input
plots=False
learning_rate=0.001
epochs=200
data_file=""
force=True
e_unit="Ry"
dist_unit="A"
load_model=False
model="/home/afuchs/Git/NeuralNetworks/NeuralNetworks/training/save_no_name/trained_variables.npy"
model_dir="multi"
source="QE"
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

# if data_file=="":
#     print("Please specify a MD file")
#     print("Option: -input x")
#     exit
#
# if source == "":
#     print("Please specify your source (QE = Quantum Espresso, LAMMPS=Lammps")
#     print("Option: -source x")
#     exit

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

#"/home/afuchs/Documents/Ni15Au15/",
data_files=["/home/afuchs/Documents/Ni1Au2/",
            "/home/afuchs/Documents/Ni2Au1",
            "/home/afuchs/Documents/Ni2Au2",
            "/home/afuchs/Documents/NiAu20"]
# data_files=["/home/afuchs/Documents/Ni1Au2/",
#             "/home/afuchs/Documents/Ni2Au1",
#             "/home/afuchs/Documents/Ni2Au2",
#             "/home/afuchs/Documents/Ni5Au5_6000/"]
Multi=_NN.MultipleInstanceTraining()

for i in range(len(data_files)):
    data_file=data_files[i]
    #Load trainings instance
    Training=_NN.AtomicNeuralNetInstance()
    Training.IsPartitioned=False
    Training.CalcDatasetStatistics=True
    Training.UseForce=force
    #Default symmetry function set
    #Training.NumberOfRadialFunctions=15
    Training.Lambs=[1.0,-1.0]
    Training.Zetas=[0.2,0.5,1,3,10]#[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
    Training.Etas=[0.01]
    Training.Rs = [1, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4]
    Training.R_Etas =[0.2,0.3,0.8,0.8,0.8,1,1,1,1,1,1,0.8,0.3,0.3,0.2]
    #Read file
    if source == "QE":
        Training.read_qe_md_files(data_file,e_unit,dist_unit,DataPointsPercentage=percentage_of_data,Calibration=["/home/afuchs/Documents/Calibration/Ni","/home/afuchs/Documents/Calibration/Au"])
    else:
        Training.read_lammps_files(data_file,energy_unit=e_unit,dist_unit=dist_unit,DataPointsPercentage=percentage_of_data,calibrate=False)

    # Default trainings settings
    for i in range(len(Training.Atomtypes)):
        Training.Structures.append([Training.SizeOfInputsPerType[i],100,80,60,40,1])
    if not("trained_variables" in model):
        model=os.path.join(model,"trained_variables.npy")
    Training.Dropout=[0,0,0]
    Training.RegularizationParam=0.01
    Training.InitStddev=0.1
    Training.LearningDecayEpochs=1000
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
    Multi.TrainingInstances.append(Training)

#Create a normalization for the data
for i,Training in enumerate(Multi.TrainingInstances):
    vars=[]
    means=[]
    min_energy=1e10
    vars.append(Training._VarianceOfDs)
    means.append(Training._MeansOfDs)
    temp=min(Training._DataSet.energies)
    if temp<min_energy:
        min_energy=temp

max_means=means[0]
max_vars=vars[0]
for j in range(1,len(means)):
    max_means=_np.maximum(max_means,means[j])
    max_vars=_np.maximum(max_means,vars[j])



for i,Training in enumerate(Multi.TrainingInstances):
    Multi.TrainingInstances[i]._MeansOfDs = max_means
    Multi.TrainingInstances[i]._VarianceOfDs = max_vars
    #Create batches
    batch_size=len(Training._DataSet.energies)*(percentage_of_data/100)/50
    Multi.TrainingInstances[i].make_training_and_validation_data(batch_size,90,10)



Multi.MakePlots=True
Multi.EpochsPerCycle=2
Multi.GlobalEpochs=epochs
Multi.GlobalLearningRate=learning_rate
Multi.GlobalRegularization="L2"
Multi.GlobalRegularizationParam=0.01
Multi.GlobalStructures=Training.Structures
Multi.MakePlots=True
Multi.SavingDirectory=model_dir
#Multi.PESCheck=check_pes.PES(model_dir,Training)
Multi.initialize_multiple_instances(MakeAllVariable=True)
if load_model:
    Multi.train_multiple_instances(StartModelName=model)
else:
    Multi.train_multiple_instances()
#Start training
#Training.start_batch_training()
#Save figures
figures=[manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

for i, figure in enumerate(figures):
    figure.savefig(os.path.join(model_dir,'figure%d.png' % i))