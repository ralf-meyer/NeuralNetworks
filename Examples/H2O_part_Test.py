from NeuralNetworks import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworks import ReadQEData as reader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NeuralNetworks import ClusterGeometries
import scipy
import pyQChem as qc

plt.close("all")




aimd_out = qc.read("Q-Chem/md.out",silent=True)
aimd_job1 = aimd_out.list_of_jobs[0] # Frequency calculation, can be ignored
aimd_job2 = aimd_out.list_of_jobs[1] # Aimd calculation

trajectory_geometries =  aimd_job2.aimd.geometries
np_energies=np.asarray(aimd_job2.aimd.energies)
trajectory_energies =  list((np_energies-np.min(np_energies))*27.211427)

#Create surface data

theta=np.arange(60*np.pi/180,180*np.pi/180,0.005)
rhs=np.arange(0.1,5,0.01)

ct=0
geoms=[]
geom=[]
xs=[]
ys=[]

for w in theta:
    for r in rhs:
        for i in range(2):
            if ct==0:
                x=r
                y=0
                z=0
                ct=ct+1
                geom.append(("O",np.zeros(3)))
                h1=[x,y,z]
                geom.append(("H",np.asarray(h1)))
            else:
                x=r*np.cos(w)
                y=r*np.sin(w)
                z=0
                h2=[x,y,z]
                geom.append(("H",np.asarray(h2)))
                geoms.append(geom)
                ct=0
                geom=[]


#Load first trainings data 
Training=NN.AtomicNeuralNetInstance()
Training2=NN.AtomicNeuralNetInstance()
Evaluation=NN.AtomicNeuralNetInstance()

Training.XYZfile="/home/afuchs/Lammps-Rechnungen/H2O/water.xyz"
Training.Logfile="/home/afuchs/Lammps-Rechnungen/H2O/log.water"
Training.atomtypes=["O","H"]
Training.NumberOfRadialFunctions=7
#angular symmetry function settings
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]

Training.read_files(True)
Training.Ds.energies=list(np.asarray(Training.Ds.energies)*0.043)
Training.make_training_and_validation_data(100,70,30)


#Train with first data 
NrO=1
NrH=2
MyStructure=NN.PartitionedStructure()
MyStructure.ForceFieldNetworkStructure=[Training.SizeOfInputs[0],10,10,1]
Training.Structures.append(MyStructure)
Training.NumberOfAtomsPerType.append(NrO)
Training.Structures.append(MyStructure)
Training.NumberOfAtomsPerType.append(NrH)
Training.LearningRate=0.01
Training.CostFunType="squared-difference"
Training.dE_Criterium=0.03
Training.Epochs=1500
Training.MakePlots=True
Training.ActFun="elu"
Training.IsPartitioned=True
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.RegularizationParam=0.0001
Training.LearningRateType="exponential_decay"
Training.LearningRateDecaySteps=100
Training.SavingDirectory="save_stage_1_h2o"
Training.make_and_initialize_network()

#Start first training

Training.start_batch_training()



#Load second trainings data
Training2.atomtypes=["O","H"]
Training2.NumberOfRadialFunctions=7
#angular symmetry function settings
Training2.Lambs=[1.0,-1.0]
Training2.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training2.Etas=[0.1]
#get zero data geometries to match nets
parsed_geoms=NN.parse_qchem_geometries(trajectory_geometries)
#prepare the correction network dataset
Training2.init_correction_network_data(Training,parsed_geoms,trajectory_energies,["O","H"],N_zero_geoms=10000)
Training2.make_training_and_validation_data(100,70,30)


#Train with second data
MyStructure2=NN.PartitionedStructure()
MyStructure2.ForceFieldNetworkStructure=[Training.SizeOfInputs[0],10,10,1]
MyStructure2.CorrectionNetworkStructure=[Training.SizeOfInputs[0],80,80,20,1]
#MyStructure2.CorrectionNetworkStructure=[Training2.SizeOfInputs[0],15,15,1]
Training2.Structures.append(MyStructure2)
Training2.Structures.append(MyStructure2)
Training2.IsPartitioned=True
Training2.LearningRate=0.001
Training2.CostFunType="squared-difference"
Training2.dE_Criterium=0.01
Training2.Epochs=2500
Training2.MakePlots=True
Training2.OptimizerType="Adam"
Training2.ActFun="elu"
Training2.Regularization="L2"
Training2.RegularizationParam=0.001
Training2.LearningRateType="exponential_decay"
Training2.LearningRateDecaySteps=100
Training2.SavingDirectory="save_stage_2_h2o"


Training2.NumberOfAtomsPerType.append(NrO)
Training2.NumberOfAtomsPerType.append(NrH)
Training2.expand_existing_net(ModelName="save_stage_1_h2o/trained_variables")

Training2.start_batch_training()


#start evaluation
Evaluation.atomtypes=["O","H"]
Evaluation.NumberOfRadialFunctions=7
#angular symmetry function settings
Evaluation.Lambs=[1.0,-1.0]
Evaluation.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Evaluation.Etas=[0.1]   
Evaluation.Structures.append(MyStructure2)
Evaluation.Structures.append(MyStructure2)
Evaluation.IsPartitioned=True
Evaluation.create_eval_data(geoms)
Evaluation.NumberOfAtomsPerType.append(NrO)
Evaluation.NumberOfAtomsPerType.append(NrH)
out=Evaluation.start_evaluation(Evaluation.NumberOfAtomsPerType,ModelName="save_stage_2_h2o/trained_variables")
#plot 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = []
t = []
e = []

with open("Q-Chem/scan_files/H20surface.txt", "r") as fin:
    fin.next() # Skip header line
    for line in fin:
        sp = line.split()
        r.append(float(sp[0]))
        t.append(float(sp[1]))
        e.append(float(sp[2]))

Nr = len(np.unique(r))
Nt = len(np.unique(t))
r = np.array(r).reshape((Nr, Nt))
t = np.array(t).reshape((Nr, Nt))
e = np.array(e).reshape((Nr, Nt))

L=e>-60
e[L]=np.min(e)
e=(e-np.min(e))*27.211427
ax.set_zlim(-30,30)
meshX,meshY=np.meshgrid(rhs,theta*180/np.pi)
meshZ=out.reshape(len(theta),len(rhs))
#L=np.abs(meshZ)>45
#meshZ[L]=-45
ax.plot_surface(meshX,meshY,meshZ)
ax.plot_surface(r,t,e)
