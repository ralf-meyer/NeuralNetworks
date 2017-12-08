from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks.data_generation import data_readers
from os import listdir
from os.path import isfile, join
import numpy as np

input_reader=data_readers.SimpleInputReader()
mypath="/home/afuchs/Documents/Validation_geometries/"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in files:
    input_reader.read(join(mypath,file),skip=3)
energies=np.asarray(input_reader.energies)

Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.TextOutput=False
Training.prepare_evaluation("/home/afuchs/Documents/NiAu_Training/NiAu_Test_without_prex4",nr_atoms_per_type=[1,54],atom_types=["Ni","Au"])
#Training.create_eval_data(input_reader.geometries)
#out=Training.eval_dataset_energy(Training.EvalData)
out=[]
for geometry in input_reader.geometries:
    out.append(Training.energy_for_geometry(geometry))
ref=energies-np.min(energies)
res=out-np.min(out)
for i in range(len(res)):
    print([files[i],res[i],ref[i]])
