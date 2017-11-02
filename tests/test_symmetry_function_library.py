import unittest

from NeuralNetworks.SymmetryFunctionSet import SymmetryFunctionSet as SymFunSet_cpp
import numpy as np

class LibraryTest(unittest.TestCase):

    def test_derivaties(self):
        sfs_cpp = SymFunSet_cpp(["Ni", "Au"], cutoff = 10.)
        pos = np.array([[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0,-1.0, 0.0]])
        types = ["Ni", "Ni", "Au", "Ni", "Au"]
        geo = [(t, p) for t,p in zip(types, pos)]

        rss = [0.0]
        etas = [np.log(2.0)]

        sfs_cpp.add_radial_functions(rss, etas)
        sfs_cpp.add_angular_functions([1.0], [1.0], etas)

        out_cpp = sfs_cpp.eval_geometry(geo)
        analytical_derivatives = sfs_cpp.eval_derivatives(types, pos)
        numerical_derivatives = np.zeros((len(out_cpp), out_cpp[0].size, pos.size))
        dx = 0.00001
        for i in xrange(pos.size):
            dpos = np.zeros(pos.shape)
            dpos[np.unravel_index(i,dpos.shape)] += dx
            numerical_derivatives[:,:,i] = (np.array(sfs_cpp.eval(types, pos+dpos))
                             - np.array(sfs_cpp.eval(types, pos-dpos)))/(2*dx)

        self.assertTrue(all(abs(numerical_derivatives.flatten() -
            np.array(analytical_derivatives).flatten()) < 1e-6))
