#model training script
import sys
from NeuralNetworks import NeuralNetworkUtilities as _NN
import os
from NeuralNetworks import check_pes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gc
from NeuralNetworks.data_generation import data_readers as _readers


def get_multi_data(model,data_files):

    predictions=[]
    targets=[]
    i=0
    while i < len(data_files):
        try:
            data_file=data_files[i]
            print(data_file)
            #Load trainings instance
            Training=_NN.AtomicNeuralNetInstance()
            Training.TextOutput = False
            Training.CalcDatasetStatistics = False
            Reader=_readers.QE_MD_Reader()
            Reader.E_conv_factor=13.605698066
            Reader.read_folder(data_file)
            offset = (-774.90203736 * Reader.nr_atoms_per_type[1] + Reader.nr_atoms_per_type[0] * -185.29265964) * 13.605698066
            Training.prepare_evaluation(model,Reader.nr_atoms_per_type)
            Training.create_eval_data(Reader.geometries)
            temp=[pred[0]/sum(Reader.nr_atoms_per_type) for pred in Training.eval_dataset_energy(Training.EvalData)]
            predictions+=temp
            targets+=list(np.divide(np.subtract(Reader.e_pot,offset),sum(Reader.nr_atoms_per_type)))
            i+=1
            gc.collect()
        except:
            print("Failed for file "+str(data_file)+" retrying...")

    return predictions,targets


model="/home/afuchs/Documents/NiAu_Training/multi_morse_2nd/trained_variables.npy"
data_files1=["/home/afuchs/Documents/13atomic/Ni1Au12",
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
            "/home/afuchs/Documents/13atomic/Ni12Au1"
            ]

data_files2=["/home/afuchs/Documents/home/Ni1Au54",
            "/home/afuchs/Documents/home/Ni2Au53",
            "/home/afuchs/Documents/home/Ni3Au52",
            "/home/afuchs/Documents/home/Ni6Au49",
            "/home/afuchs/Documents/home/Ni8Au47",
            "/home/afuchs/Documents/home/Ni10Au45",
            "/home/afuchs/Documents/home/Ni11Au44",
            "/home/afuchs/Documents/home/Ni12Au43",
            "/home/afuchs/Documents/home/Ni13Au42",
            "/home/afuchs/Documents/home/Ni14Au41",
            #"/home/afuchs/Documents/home/Ni15Au40",
            #"/home/afuchs/Documents/home/Ni17Au38",
            #"/home/afuchs/Documents/home/Ni18Au37",
            # "/home/afuchs/Documents/home/Ni19Au36",
            # "/home/afuchs/Documents/home/Ni20Au35",
            # "/home/afuchs/Documents/home/Ni21Au34",
            # "/home/afuchs/Documents/home/Ni22Au33",
            # "/home/afuchs/Documents/home/Ni23Au32",
            # "/home/afuchs/Documents/home/Ni24Au31",
            # "/home/afuchs/Documents/home/Ni25Au30",
            # "/home/afuchs/Documents/home/Ni26Au29",
            # "/home/afuchs/Documents/home/Ni27Au28",
            # "/home/afuchs/Documents/home/Ni28Au27",
            # "/home/afuchs/Documents/home/Ni29Au26",
            # "/home/afuchs/Documents/home/Ni36Au19",
            # "/home/afuchs/Documents/home/Ni37Au18"
             ]

data_files3=[#"/home/afuchs/Documents/home/Ni1Au54",
            # "/home/afuchs/Documents/home/Ni2Au53",
            # "/home/afuchs/Documents/home/Ni3Au52",
            # "/home/afuchs/Documents/home/Ni6Au49",
            # "/home/afuchs/Documents/home/Ni8Au47",
            # "/home/afuchs/Documents/home/Ni10Au45",
            # "/home/afuchs/Documents/home/Ni11Au44",
            # "/home/afuchs/Documents/home/Ni12Au43",
            # "/home/afuchs/Documents/home/Ni13Au42",
            # "/home/afuchs/Documents/home/Ni14Au41",
            "/home/afuchs/Documents/home/Ni15Au40",
            "/home/afuchs/Documents/home/Ni17Au38",
            "/home/afuchs/Documents/home/Ni18Au37",
            "/home/afuchs/Documents/home/Ni19Au36",
            "/home/afuchs/Documents/home/Ni20Au35",
            "/home/afuchs/Documents/home/Ni21Au34",
            "/home/afuchs/Documents/home/Ni22Au33",
            "/home/afuchs/Documents/home/Ni23Au32",
            # "/home/afuchs/Documents/home/Ni24Au31",
            # "/home/afuchs/Documents/home/Ni25Au30",
            # "/home/afuchs/Documents/home/Ni26Au29",
            # "/home/afuchs/Documents/home/Ni27Au28",
            # "/home/afuchs/Documents/home/Ni28Au27",
            # "/home/afuchs/Documents/home/Ni29Au26",
            # "/home/afuchs/Documents/home/Ni36Au19",
            # "/home/afuchs/Documents/home/Ni37Au18"
             ]

data_files4=[#"/home/afuchs/Documents/home/Ni1Au54",
            # "/home/afuchs/Documents/home/Ni2Au53",
            # "/home/afuchs/Documents/home/Ni3Au52",
            # "/home/afuchs/Documents/home/Ni6Au49",
            # "/home/afuchs/Documents/home/Ni8Au47",
            # "/home/afuchs/Documents/home/Ni10Au45",
            # "/home/afuchs/Documents/home/Ni11Au44",
            # "/home/afuchs/Documents/home/Ni12Au43",
            # "/home/afuchs/Documents/home/Ni13Au42",
            # "/home/afuchs/Documents/home/Ni14Au41",
            # "/home/afuchs/Documents/home/Ni15Au40",
            # "/home/afuchs/Documents/home/Ni17Au38",
            # "/home/afuchs/Documents/home/Ni18Au37",
            # "/home/afuchs/Documents/home/Ni19Au36",
            # "/home/afuchs/Documents/home/Ni20Au35",
            # "/home/afuchs/Documents/home/Ni21Au34",
            # "/home/afuchs/Documents/home/Ni22Au33",
            # "/home/afuchs/Documents/home/Ni23Au32",
            "/home/afuchs/Documents/home/Ni24Au31",
            "/home/afuchs/Documents/home/Ni25Au30",
            "/home/afuchs/Documents/home/Ni26Au29",
            "/home/afuchs/Documents/home/Ni27Au28",
            "/home/afuchs/Documents/home/Ni28Au27",
            "/home/afuchs/Documents/home/Ni29Au26",
            "/home/afuchs/Documents/home/Ni36Au19",
            "/home/afuchs/Documents/home/Ni37Au18"
             ]

#Save figures

#prediction13,reference13=get_multi_data(model,data_files1)
#np.save("/home/afuchs/Documents/13_atom_data.npy",[prediction13,reference13])
temp=np.load("/home/afuchs/Documents/13_atom_data.npy")
prediction13=temp[0]
reference13=temp[1]
#temp55,ref55=get_multi_data(model,data_files2)
#np.save("/home/afuchs/Documents/55_atom_data_1.npy",[temp55,ref55])
temp=np.load("/home/afuchs/Documents/55_atom_data_1.npy")
temp55=temp[0]
ref55=temp[1]
temp55_2,ref55_2=get_multi_data(model,data_files3)
np.save("/home/afuchs/Documents/55_atom_data_2.npy",[temp55_2,ref55_2])
temp55_3,ref55_3=get_multi_data(model,data_files4)
np.save("/home/afuchs/Documents/55_atom_data_3.npy",[temp55_3,ref55_3])
prediction55=temp55+temp55_2+temp55_3
reference55=ref55+ref55_2+ref55_3
np.save("/home/afuchs/Documents/55_atom_data.npy",[prediction55,reference55])
prediction147,reference147=get_multi_data(model,["/home/afuchs/Documents/Validation_big"])
np.save("/home/afuchs/Documents/147_atom_data.npy",[prediction147,reference147])
plt.figure()
plt.scatter(reference13, prediction13, s = 10, c = "C1", marker = "o", label = "13 atoms")#, edgecolors="k")
plt.scatter(reference55, prediction55 - 0.4, s = 10, c = "C0", marker = "s", label = "55 atoms")
plt.scatter(reference147, prediction147 - 0.7, s = 10, c = "C3", marker = "^", label = "147 atoms")
plt.xlabel("DFT / eV per atom")
plt.ylabel("Prediction / eV per atom")
plt.xlim(-3.8,-1.7)
plt.ylim(-3.8,-1.7)
# plt.text(-1.75, -3.6, "RMSD Training = {:5.3f} eV / atom".format(np.sqrt((np.sum((reference13-prediction13)**2) + sum((reference55-prediction55 + 0.4)**2))/(len(reference13) + len(prediction55)))),
#         horizontalalignment = "right")
# plt.text(-1.75, -3.7, "RMSD 147 atoms = {:5.3f} eV / atom".format(np.sqrt(np.mean((reference147 - prediction147 + 0.7)**2))),
#         horizontalalignment = "right")
xy=np.linspace(-3.8,-1.7)
plt.plot(xy,xy,'k')
figures=[manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

for i, figure in enumerate(figures):
    figure.savefig(os.path.join("/home/afuchs/Documents",'figure%d.png' % i))
