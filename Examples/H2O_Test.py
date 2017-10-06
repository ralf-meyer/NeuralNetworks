#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:16:01 2017

@author: alexf1991
"""

from NeuralNetworks import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworks import ReadQEData as reader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from NeuralNetworks import ClusterGeometries
import scipy
import pyQChem as qc

plt.close("all")
#parsing q-chem format to NN compatible format
def parse_geometries(in_geoms):
    out_geoms=[]
    for in_geom in in_geoms:
        atoms_list=in_geom.list_of_atoms
        out_geom=[]
        for atom in atoms_list:
            xyz=[float(atom[1]),float(atom[2]),float(atom[3])]
            my_tuple=(atom[0],np.asarray(xyz))
            out_geom.append(my_tuple)
        out_geoms.append(out_geom)
            
                
        
    return out_geoms

#aimd_out = qc.read("Q-Chem/md.out",silent=True)
#aimd_job1 = aimd_out.list_of_jobs[0] # Frequency calculation, can be ignored
#aimd_job2 = aimd_out.list_of_jobs[1] # Aimd calculation
#
#trajectory_geometries =  aimd_job2.aimd.geometries
##Add zero data geometries
#N_zero=100
#parsed_geoms=parse_geometries(trajectory_geometries)
#ds_r_min,ds_r_max=NN.get_ds_r_min_r_max(parsed_geoms)
#
#zero_ds_geoms=NN.create_zero_diff_geometries(ds_r_min,ds_r_max,["O","H"],[1,2],N_zero)
#all_geoms=parsed_geoms+zero_ds_geoms
##Add zero data energies
#zero_ds_energies=[max(aimd_job2.aimd.energies)]*N_zero
#np_energies=np.asarray(aimd_job2.aimd.energies+zero_ds_energies)
#all_energies =  list((np_energies-np.min(np_energies))*27.211427)

#Create surface data

theta=np.linspace(90*np.pi/180,130*np.pi/180,61)
rhs=np.linspace(2,4,49)

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
                geom.append(("Ni",np.zeros(3)))
                h1=[x,y,z]
                geom.append(("Au",np.asarray(h1)))
            else:
                x=r*np.cos(w)
                y=r*np.sin(w)
                z=0
                h2=[x,y,z]
                geom.append(("Au",np.asarray(h2)))
                geoms.append(geom)
                ct=0
                geom=[]


#ClusterGeometries.make_files(geoms,randomization_factor=0,path='/home/afuchs/QE-Rechnungen/pw.x/H2O/Input/',Name='H2O')

#Load first trainings data 
Training=NN.AtomicNeuralNetInstance()
Training2=NN.AtomicNeuralNetInstance()
Evaluation=NN.AtomicNeuralNetInstance()
#
#scf_reader=reader.QE_SCF_Reader()
#scf_reader.E_conv_factor=13.605698066
#scf_reader.Geom_conv_factor=0.529177249
#scf_reader.get_files("/home/afuchs/QE-Rechnungen/pw.x/H2O/Results")
#scf_reader.read_all_files()
#
#scf_reader.Calibration.append(("/home/afuchs/QE-Rechnungen/pw.x/H2O/single_atoms/h.out",2))
#scf_reader.Calibration.append(("/home/afuchs/QE-Rechnungen/pw.x/H2O/single_atoms/o.out",1))
#scf_reader.calibrate_energy()
#reference_energies=scf_reader.e_tot_rel
##remove outlier
#L=reference_energies!=min(reference_energies)
#pop_idx=np.arange(0,len(scf_reader.geometries))[L==False]
#ref_e=reference_energies[L]
#scf_reader.geometries.pop(pop_idx)
#scf_geoms=scf_reader.geometries
##Rescale
#ref_e=ref_e-min(ref_e)
nr_atoms=2
md_reader_h=reader.QE_MD_Reader()
for i in range(nr_atoms):
    md_reader_h.atom_types.append("H"+str(i+1))
md_reader_h.E_conv_factor=13.605698066
md_reader_h.Geom_conv_factor=0.529177249
md_reader_h.get_files("/home/afuchs/QE-Rechnungen/pw.x/H_MD/Train")
md_reader_h.read_all_files()
#md_reader_h.Calibration.append(("/home/afuchs/QE-Rechnungen/pw.x/H2O/single_atoms/h.out",6))
md_reader_h.calibrate_energy()


md_reader=reader.QE_MD_Reader()
md_reader.E_conv_factor=13.605698066
md_reader.Geom_conv_factor=1
md_reader.get_files("/home/afuchs/QE-Rechnungen/pw.x/MD_pbe")
md_reader.read_all_files()
md_reader.Calibration.append(("/home/afuchs/QE-Rechnungen/pw.x/Single_Au_pbe/Au.out",7))
md_reader.Calibration.append(("/home/afuchs/QE-Rechnungen/pw.x/Single_Ni_pbe/Ni.out",6))
md_reader.calibrate_energy()
niau_energies=md_reader.e_pot_rel-min(md_reader.e_pot_rel)

#ds_r_min,ds_r_max=NN.get_ds_r_min_r_max(md_reader.geometries)
#N_zero=50
#zero_ds_geoms=NN.create_zero_diff_geometries(ds_r_min,ds_r_max,["Ni","Au"],[7,6],N_zero)
#all_geoms=md_reader.geometries+zero_ds_geoms
#zero_ds_energies=[max(niau_energies)]*N_zero
#np_energies=np.asarray(list(niau_energies)+list(zero_ds_energies))
#all_energies =  list((np_energies-np.min(np_energies)))
#train_e=list(ref_e)+list(ref_e2)
#train_geoms=scf_geoms+md_reader.geometries

Training.atomtypes=["H"]
Training.NumberOfRadialFunctions=20

#angular symmetry function settings
Training.Lambs=[1.0,-1.0]
Training.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training.Etas=[0.1]   
#Training.init_dataset(train_geoms,train_e)
h_energies=md_reader_h.e_pot_rel-min(md_reader_h.e_pot_rel)
Training.init_dataset(md_reader_h.geometries,h_energies)
Training.make_training_and_validation_data(10,70,30)



#Train with data 
#NrO=3
NrH=6
Training.Structures=[]
Training.NumberOfAtomsPerType=[]
Training.Structures.append([Training.SizeOfInputs[0],100,100,1])
Training.Dropout=[0,0.5,0]
#Training.NumberOfAtomsPerType.append(NrO)
#Training.Structures.append([Training.SizeOfInputs[1],80,25,1])
Training.NumberOfAtomsPerType.append(NrH)
Training.HiddenType="truncated_normal"
Training.HiddenData=list()
Training.BiasData=list()
Training.ActFun="elu"
Training.LearningRate=0.005
Training.dE_Criterium=0
Training.Epochs=20000
#
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.Regularization="L2"
Training.CostFunType="Adaptive_2"
Training.LearningRateType="exponential_decay"
Training.SavingDirectory="pretraining"
Training.LearningDecayEpochs=100
Training.RegularizationParam=0.1
Training.MakeLastLayerConstant=True
Training.make_and_initialize_network()
#Training.expand_existing_net(ModelName="save_h2o/trained_variables")
#Start first training

Training.start_batch_training()

#
#Load second trainings data

#Training2.TrainingBatches=Training.TrainingBatches
#Training2.ValidationBatches=Training.ValidationBatches
Training2.atomtypes=["Ni","Au"]
Training2.NumberOfRadialFunctions=20
Training2.Lambs=[1.0,-1.0]
Training2.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Training2.Etas=[0.1]   
#Training2.init_dataset(train_geoms,train_e)
Training2.init_dataset(md_reader.geometries,md_reader.e_pot_rel)
Training2.make_training_and_validation_data(2,70,10)
#Train with second data
Training2.Structures=[]
Training2.Structures.append([Training2.SizeOfInputs[0],100,100,40,20,1])
Training2.Structures.append([Training2.SizeOfInputs[1],100,100,40,20,1])
Training2.Dropout=[0,0,0,0,0]
Training2.LearningRate=0.0001
Training2.CostCriterium=0
Training2.dE_Criterium=0
Training2.RegularizationParam=0.1
Training2.Epochs=1500
Training2.MakePlots=True
Training2.ActFun="elu"
Training2.CostFunType="Adaptive_2"
Training2.OptimizerType="Adam"
Training2.SavingDirectory="save_NiAu"
Training2.MakeLastLayerConstant=True
Training2.MakeAllVariable=False

#Evaluate quality of learning transfer
NrNi=6
NrAu=7
Training2.NumberOfAtomsPerType=[]
Training2.NumberOfAtomsPerType.append(NrNi)
Training2.NumberOfAtomsPerType.append(NrAu)
Training2.expand_existing_net(ModelName="pretraining/trained_variables",Pretraining=True)
Training2.start_batch_training()

#start evaluation
Evaluation.atomtypes=["Ni","Au"]
Evaluation.NumberOfRadialFunctions=20
#angular symmetry function settings
Evaluation.Lambs=[1.0,-1.0]
Evaluation.Zetas=[0.025,0.045,0.075,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,18,36,100]
Evaluation.Etas=[0.1]  
Evaluation.Structures=[]
Evaluation.Structures.append([Training2.SizeOfInputs[0],100,100,40,20,1])
Evaluation.Structures.append([Training2.SizeOfInputs[1],100,100,40,20,1])

Evaluation.create_eval_data(geoms)
NrNi=1
NrAu=2
Evaluation.NumberOfAtomsPerType=[]
Evaluation.NumberOfAtomsPerType.append(NrNi)
Evaluation.NumberOfAtomsPerType.append(NrAu)
out=Evaluation.start_evaluation(Evaluation.NumberOfAtomsPerType,ModelName="save_NiAu/trained_variables")
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
#interpolate
for i,line in enumerate(e):
    L=line>-60
    idx=np.where(L)
    if idx[0].size!=0:
        min_idx=idx[0][0]
        max_idx=idx[0][-1]
        y1=line[min_idx-1]
        y2=line[max_idx+1]
        y_int=np.linspace(y1,y2,len(idx))
        line[L]=y_int
        e[i]=line

e=(e-np.min(e))*27.211427
ax.set_zlim(0,0.2)
#out=out-np.min(out)
meshX,meshY=np.meshgrid(rhs,theta*180/np.pi)
meshZ=out.reshape(len(theta),len(rhs))
#L=np.abs(meshZ)>45
#meshZ[L]=-45
ax.plot_surface(meshX,meshY,meshZ)
#Equilibrium position
m=np.min(meshZ)
L=meshZ==m
print(str(meshX[L])+" Angstroem")
print(str(meshY[L])+" grad")
#ax.plot_surface(r,t,e)

#num_e=len(reference_energies)
#xs=[]
#ys=[]
#for w in theta:
#    for r in rhs:
#        if len(xs)<num_e:
#            ys.append(w)
#            xs.append(r)
#temp=np.zeros((len(theta)*len(rhs)))
#temp[0:num_e]=reference_energies
#ref_surf=temp.reshape(len(theta),len(rhs))
#ax.plot_surface(meshX,meshY,ref_surf)
#ax.scatter(xs,ys,reference_energies,c='r',marker='o')