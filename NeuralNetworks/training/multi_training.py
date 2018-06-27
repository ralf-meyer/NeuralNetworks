#model training script
import sys
from NeuralNetworks import NeuralNetworkUtilities as _NN
import os
from NeuralNetworks import check_pes
import numpy as _np
import matplotlib


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
model_dir="morse_ab_50"
source="QE"
percentage_of_data=100


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


data_path="/home/afuchs/Documents/Au_Dataset2-80/"
data_files=[os.path.join(data_path,f) for f in os.listdir(data_path)]
for this in data_files:
    folder_files = [os.path.join(this, f) for f in os.listdir(this)]
    for myfile in folder_files:
        temp=open(myfile,"r").read()
        if "error" in temp:
            print("Error in :"+str(myfile))
print(data_files)

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
    Training.Lambs=[1.0,-1.0]
    Training.Zetas=[0.2,0.5,1,3,10]#[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
    Training.Etas=[0.01]
    Training.Rs = [0,0,0,0,0,0,0,0]#,1.16674542, 1.81456625,2.3, 2.89256287, 4.53134823, 6.56226301, 6.92845869]
    Training.R_Etas = [0.4/bohr2ang**2, 0.2/bohr2ang**2, 0.1/bohr2ang**2, 0.06/bohr2ang**2, 0.035/bohr2ang**2, 0.02/bohr2ang**2, 0.01/bohr2ang**2, 0.0009/bohr2ang**2]#, 2.13448882, 1.97223806,1.2, 0.81916839, 0.47314626, 0.95010978, 7.37062645]
    #Training.Rs = [0, 0, 0, 0]  # ,1.16674542, 1.81456625,2.3, 2.89256287, 4.53134823, 6.56226301, 6.92845869]
    #Training.R_Etas = [0.4 / bohr2ang ** 2,  0.1 / bohr2ang ** 2,0.035 / bohr2ang ** 2, 0.01 / bohr2ang ** 2]
    Training.Cutoff=7
    #Read file
    if source == "QE":
        Training.read_qe_md_files(data_file,e_unit,dist_unit,DataPointsPercentage=percentage_of_data,Calibration=["/home/afuchs/Documents/Conference_Lausanne/Au_Calibration/Au1_sp.out"])
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
    Training.LearningDecayEpochs=1000
    Training.Epochs=epochs
    Training.ForceCostParam=0.001#might be too high if training yields bad results reduce to 10⁻4
    Training.MakePlots=plots
    Training.ActFun=["selu","morse","selu","none"]
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


ct=0
for i,Training in enumerate(Multi.TrainingInstances):
    Multi.TrainingInstances[i]._MeansOfDs = max_means
    Multi.TrainingInstances[i]._VarianceOfDs = max_vars
    #Create batches
    ct+=len(Multi.TrainingInstances[i]._DataSet.energies)
    batch_size=10#len(Training._DataSet.energies)*(percentage_of_data/100)/50
    print(data_files[i])
    Multi.TrainingInstances[i].make_training_and_validation_data(batch_size,90,10)

print(ct)


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