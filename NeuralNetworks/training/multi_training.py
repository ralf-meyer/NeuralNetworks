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
epochs=50000
data_file=""
force=False
e_unit="Ry"
dist_unit="A"
load_model=False
model="/home/afuchs/Documents/SELU/selu_morse_1"
model_dir="morse_selu_morse_test1"
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
data_files=["/home/afuchs/Documents/13atomic/Ni1Au12",
            "/home/afuchs/Documents/13atomic/Ni2Au11",
            "/home/afuchs/Documents/13atomic/Ni3Au10",
            "/home/afuchs/Documents/13atomic/Ni4Au9",
            "/home/afuchs/Documents/13atomic/Ni5Au8",
            "/home/afuchs/Documents/13atomic/Ni6Au7",
            "/home/afuchs/Documents/13atomic/Ni7Au6",
            "/home/afuchs/Documents/13atomic/Ni8Au5",
            "/home/afuchs/Documents/13atomic/Ni9Au4",
            "/home/afuchs/Documents/13atomic/Ni10Au3",
            "/home/afuchs/Documents/13atomic/Ni11Au2",
            "/home/afuchs/Documents/13atomic/Ni12Au1",
            "/home/afuchs/Documents/home/Ni1Au54",
            "/home/afuchs/Documents/home/Ni2Au53",
            "/home/afuchs/Documents/home/Ni3Au52",
            "/home/afuchs/Documents/home/Ni6Au49",
            "/home/afuchs/Documents/home/Ni8Au47",
            "/home/afuchs/Documents/home/Ni10Au45",
            "/home/afuchs/Documents/home/Ni11Au44",
            "/home/afuchs/Documents/home/Ni12Au43",
            "/home/afuchs/Documents/home/Ni13Au42",
            "/home/afuchs/Documents/home/Ni14Au41",
            "/home/afuchs/Documents/home/Ni15Au40",
            "/home/afuchs/Documents/home/Ni17Au38",
            "/home/afuchs/Documents/home/Ni18Au37",
            "/home/afuchs/Documents/home/Ni19Au36",
            "/home/afuchs/Documents/home/Ni20Au35",
            "/home/afuchs/Documents/home/Ni21Au34",
            "/home/afuchs/Documents/home/Ni22Au33",
            "/home/afuchs/Documents/home/Ni23Au32",
            "/home/afuchs/Documents/home/Ni24Au31",
            "/home/afuchs/Documents/home/Ni25Au30",
            "/home/afuchs/Documents/home/Ni26Au29",
            "/home/afuchs/Documents/home/Ni27Au28",
            "/home/afuchs/Documents/home/Ni28Au27",
            #"/home/afuchs/Documents/home/Ni29Au26",
            #"/home/afuchs/Documents/home/Ni36Au19",
            #"/home/afuchs/Documents/home/Ni37Au18"
            ]
# data_files=["/home/afuchs/Documents/Ni1Au2/",
#             "/home/afuchs/Documents/Ni2Au1",
#             "/home/afuchs/Documents/Ni2Au2",
#             "/home/afuchs/Documents/Ni5Au5_6000/"]
Multi=_NN.MultipleInstanceTraining()
Multi.GlobalLearningRate=learning_rate
for i in range(len(data_files)):
    data_file=data_files[i]
    print(data_file)
    #Load trainings instance
    Training=_NN.AtomicNeuralNetInstance()
    Training.IsPartitioned=False
    Training.IncludeMorse=True
    Training.CalcDatasetStatistics=True
    Training.UseForce=force
    #Default symmetry function set
    #Training.NumberOfRadialFunctions=15
    bohr2ang = 0.529177249
    #Training.Lambs=[1.0,-1.0]
    #Training.Zetas=[0.2,0.5,1,3,10]#[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
    #Training.Etas=[0.01]
    Training.Rs = [0,0,0,0,0,0,0,0]#,1.16674542, 1.81456625,2.3, 2.89256287, 4.53134823, 6.56226301, 6.92845869]
    Training.R_Etas = [0.4/bohr2ang**2, 0.2/bohr2ang**2, 0.1/bohr2ang**2, 0.06/bohr2ang**2, 0.035/bohr2ang**2, 0.02/bohr2ang**2, 0.01/bohr2ang**2, 0.0009/bohr2ang**2]#, 2.13448882, 1.97223806,1.2, 0.81916839, 0.47314626, 0.95010978, 7.37062645]
    #Training.Rs = [0, 0, 0, 0]  # ,1.16674542, 1.81456625,2.3, 2.89256287, 4.53134823, 6.56226301, 6.92845869]
    #Training.R_Etas = [0.4 / bohr2ang ** 2,  0.1 / bohr2ang ** 2,0.035 / bohr2ang ** 2, 0.01 / bohr2ang ** 2]
    Training.Cutoff=7
    #Read file
    if source == "QE":
        Training.read_qe_md_files(data_file,e_unit,dist_unit,DataPointsPercentage=percentage_of_data,Calibration=["/home/afuchs/Documents/Calibration/Ni","/home/afuchs/Documents/Calibration/Au"])
    else:
        Training.read_lammps_files(data_file,energy_unit=e_unit,dist_unit=dist_unit,DataPointsPercentage=percentage_of_data,calibrate=False)

    # Default trainings settings
    for i in range(len(Training.Atomtypes)):
        Training.Structures.append([Training.SizeOfInputsPerType[i],80,60,40,1])
    if not("trained_variables" in model):
        model=os.path.join(model,"trained_variables.npy")
    Training.Dropout=[0,0,0]
    Training.Regularization = "L2"
    Training.MakeLastLayerConstant=False
    Training.InitStddev=0
    Training.LearningDecayEpochs=500
    Training.Epochs=epochs
    Training.ForceCostParam=0.001
    Training.MakePlots=plots
    Training.ActFun="selu"
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
    batch_size=10#len(Training._DataSet.energies)*(percentage_of_data/100)/50
    Multi.TrainingInstances[i].make_training_and_validation_data(batch_size,90,10)



Multi.MakePlots=True
Multi.EpochsPerCycle=1
Multi.GlobalEpochs=epochs
Multi.GlobalLearningRate=learning_rate
Multi.GlobalRegularization="L2"
Multi.GlobalRegularizationParam=0.001
Multi.GlobalStructures=Training.Structures
Multi.MakePlots=True
Multi.SavingDirectory=model_dir
#Multi.PESCheck=check_pes.PES(model_dir,Training)
Multi.initialize_multiple_instances(MakeAllVariable=True)
if load_model:
    Multi.train_multiple_instances(ModelDirectory=model)
else:
    Multi.train_multiple_instances()
#Start training
#Training.start_batch_training()
#Save figures
figures=[manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

for i, figure in enumerate(figures):
    figure.savefig(os.path.join(model_dir,'figure%d.png' % i))

print(model_dir)