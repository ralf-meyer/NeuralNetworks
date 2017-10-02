from NeuralNetworks.SymmetryFunctionSet_new import SymmetryFunctionSet as SymFunSet_py, SymmetryFunctionSet_new as SymFunSet_cpp
from NeuralNetworks.SymmetryFunctionSetC import SymmetryFunctionSet as SymFunSet_c
import numpy as np
import matplotlib.pyplot as plt

sfs_cpp = SymFunSet_cpp([0, 1])
sfs_py = SymFunSet_py([0, 1])
sfs_c = SymFunSet_c([0, 1])

pos = np.array([[0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.0, 1.2, 0.0],
                [-1.2, 0.0, 0.0]])
types = [0, 0, 1, 0]

#sfs_cpp.add_TwoBodySymmetryFunction(0,0,0,[1.0,0.0],1,7.0)
#sfs_cpp.add_TwoBodySymmetryFunction(0,1,0,[1.0,0.0],1,7.0)
#sfs_cpp.add_TwoBodySymmetryFunction(1,0,0,[1.0,0.0],1,7.0)
#sfs_cpp.add_TwoBodySymmetryFunction(1,1,0,[1.0,0.0],1,7.0)
sfs_cpp.add_radial_functions([0.0],[1.0])
sfs_cpp.add_ThreeBodySymmetryFunction(0,0,0,0,[1.0, 1.0, 1.0],1,7.0)
sfs_cpp.add_ThreeBodySymmetryFunction(0,0,1,0,[1.0, 1.0, 1.0],1,7.0)
sfs_cpp.add_ThreeBodySymmetryFunction(0,1,1,0,[1.0, 1.0, 1.0],1,7.0)
sfs_cpp.add_ThreeBodySymmetryFunction(1,0,0,0,[1.0, 1.0, 1.0],1,7.0)
sfs_cpp.add_ThreeBodySymmetryFunction(1,0,1,0,[1.0, 1.0, 1.0],1,7.0)
sfs_cpp.add_ThreeBodySymmetryFunction(1,1,1,0,[1.0, 1.0, 1.0],1,7.0)
sfs_py.add_radial_functions([0.0],[1.0])
sfs_py.add_angular_functions([1.0],[1.0],[1.0])
sfs_c.add_radial_functions([0.0],[1.0])
sfs_c.add_angular_functions([1.0],[1.0],[1.0])

out_cpp = sfs_cpp.eval(types, pos)
geo = [(t, p) for t,p in zip(types, pos)]
out_py = sfs_py.eval_geometry(geo).flatten()

print "Evalutation difference smaller 1e-6: ", all(out_cpp - out_py < 1e-6)

analytical_derivatives = sfs_cpp.eval_derivatives(types, pos)
## Calculate numerical derivatives
numerical_derivatives = np.zeros((len(out_cpp), pos.size))
dx = 0.00001
for i in xrange(pos.size):
    dpos = np.zeros(pos.shape)
    dpos[np.unravel_index(i,dpos.shape)] += dx
    numerical_derivatives[:,i] = (sfs_cpp.eval(types, pos+dpos)
        - sfs_cpp.eval(types, pos-dpos))/(2*dx)

print "Derivatives difference smaller 1e-6: ", all(numerical_derivatives.flatten() - analytical_derivatives.flatten() < 1e-6)

#fig, ax = plt.subplots(ncols = 2, figsize = (10,5));
#p1 = ax[0].pcolormesh(numerical_derivatives)
#plt.colorbar(p1, ax = ax[0])
#p2 = ax[1].pcolormesh(analytical_derivatives)
#plt.colorbar(p2, ax = ax[1])
#plt.show()
