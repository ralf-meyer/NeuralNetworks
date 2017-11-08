from NeuralNetworks import NeuralNetworkUtilities
from NeuralNetworks import ReadLammpsData as _ReaderLammps
import numpy as np
import time 

#r=_ReaderLammps.LammpsReader()
#r.read_lammps("/home/afuchs/Lammps-Rechnungen/Au_md/Au_md.dump","/home/afuchs/Lammps-Rechnungen/Au_md/Au_md.xyz","/home/afuchs/Lammps-Rechnungen/Au_md/Au.log")
#start_geom=r.geometries[0]
start_geom=[('Au', np.asarray([ 0.,  0.,  0.])), 
            ('Au', np.asarray([-8.87803, -0.53645,  0.19258])),
            ('Au', np.asarray([-7.06365,  1.76129, -0.29539])), 
            ('Au', np.asarray([-5.40391,  4.08452, -0.96449])), 
            ('Au', np.asarray([-4.63949,  0.4575 ,  0.77061])), 
            ('Au', np.asarray([-6.28023, -1.92323,  0.60467])),
            ('Au', np.asarray([-4.53662,  1.38337, -1.96234])),
            ('Au', np.asarray([-6.86736, -0.56702, -2.01609])),
            ('Au', np.asarray([-5.52414,  3.03586,  1.87345])),
            ('Au', np.asarray([-7.07995,  0.4345 ,  2.36911])), 
            ('Au', np.asarray([-3.2648 , -2.06975,  1.24991])),
            ('Au', np.asarray([-4.18824, -1.56643, -1.50541])), 
            ('Au', np.asarray([-2.0273 ,  0.15485, -0.53999])), 
            ('Au', np.asarray([ 0.,  0.,  6.])),
            ('Au', np.asarray([-8.87803, -0.53645,  6.19258])), 
            ('Au', np.asarray([-7.06365,  1.76129,  5.29539])),
            ('Au', np.asarray([-5.40391,  4.08452,  5.96449])), 
            ('Au', np.asarray([-4.63949,  0.4575 ,  6.77061])),
            ('Au', np.asarray([-6.28023, -1.92323,  6.60467])),
            ('Au', np.asarray([-4.53662,  1.38337,  4.96234])), 
            ('Au', np.asarray([-6.86736, -0.56702,  4.01609])), 
            ('Au', np.asarray([-5.52414,  3.03586,  7.87345])), 
            ('Au', np.asarray([-7.07995,  0.4345 ,  8.36911])), 
            ('Au', np.asarray([-3.2648 , -2.06975,  7.24991])), 
            ('Au', np.asarray([-4.18824, -1.56643,  4.50541])), 
            ('Au', np.asarray([-2.0273 ,  0.15485,  5.53999])),
            ('Au', np.asarray([0., 0., 0.])),
            ('Au', np.asarray([-8.87803, -0.53645, 0.19258])),
            ('Au', np.asarray([-7.06365, 1.76129, -0.29539])),
            ('Au', np.asarray([-5.40391, 4.08452, -0.96449])),
            ('Au', np.asarray([-4.63949, 0.4575, 0.77061])),
            ('Au', np.asarray([-6.28023, -1.92323, 0.60467])),
            ('Au', np.asarray([-4.53662, 1.38337, -1.96234])),
            ('Au', np.asarray([-6.86736, -0.56702, -2.01609])),
            ('Au', np.asarray([-5.52414, 3.03586, 1.87345])),
            ('Au', np.asarray([-7.07995, 0.4345, 2.36911])),
            ('Au', np.asarray([-3.2648, -2.06975, 1.24991])),
            ('Au', np.asarray([-4.18824, -1.56643, -1.50541])),
            ('Au', np.asarray([-2.0273, 0.15485, -0.53999])),
            ('Au', np.asarray([0., 0., 6.])),
            ('Au', np.asarray([-8.87803, -0.53645, 6.19258])),
            ('Au', np.asarray([-7.06365, 1.76129, 5.29539])),
            ('Au', np.asarray([-5.40391, 4.08452, 5.96449])),
            ('Au', np.asarray([-4.63949, 0.4575, 6.77061])),
            ('Au', np.asarray([-6.28023, -1.92323, 6.60467])),
            ('Au', np.asarray([-4.53662, 1.38337, 4.96234])),
            ('Au', np.asarray([-6.86736, -0.56702, 4.01609])),
            ('Au', np.asarray([-5.52414, 3.03586, 7.87345])),
            ('Au', np.asarray([-7.07995, 0.4345, 8.36911])),
            ('Au', np.asarray([-3.2648, -2.06975, 7.24991])),
            ('Au', np.asarray([-4.18824, -1.56643, 4.50541])),
            ('Au', np.asarray([-2.0273, 0.15485, 5.53999]))
            ]

Training=NeuralNetworkUtilities.AtomicNeuralNetInstance()
Training.prepare_evaluation("/home/afuchs/Git/NeuralNetworks/Au_test2",atom_types=["Au"],nr_atoms_per_type=[52])
durations=[]
for i in range(0,5):
    start=time.time()
    Training.force_for_geometry(start_geom)
    print("Total time : "+str(time.time()-start)+" s")


