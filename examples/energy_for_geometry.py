from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks.data_generation import data_readers
from os import listdir
from os.path import isfile, join
import numpy as np

input_reader=data_readers.SimpleInputReader()
mypath="/home/afuchs/Documents/Validation_big/"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in files:
    input_reader.read(join(mypath,file),skip=3)
energies=np.asarray(input_reader.energies)

Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.TextOutput=False
Training.CalcDatasetStatistics=False
Training.prepare_evaluation("/home/afuchs/Documents/NiAu_Training/multi_no_angular_force",nr_atoms_per_type=[1,146])
#Training.create_eval_data(input_reader.geometries)
#out=Training.eval_dataset_energy(Training.EvalData)
out=[]
for geometry in input_reader.geometries:
    out.append(Training.energy_for_geometry(geometry))
offset=(-774.90203736*146+1*-185.29265964)*13.605698066
ref=energies-offset
res=out
min_ref=energies-min(energies)
min_res=res-min(res)
for i in range(len(res)):
    print([files[i],res[i][0][0],min_res[i][0][0],ref[i],min_ref[i],res[i][0][0]-ref[i],np.abs(min_res[i][0][0]-min_ref[i])])
print(np.mean(np.abs(min_res[:][0][0]-min_ref[:])))
print(np.sqrt(np.var(np.abs(min_res[:][0][0]-min_ref[:]))))