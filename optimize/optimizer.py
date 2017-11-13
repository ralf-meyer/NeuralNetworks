from scipy.optimize import minimize
import numpy as np

def get_type_and_xyz(geom):
    xyz=[]
    types=[]
    for atom in geom:
        types.append(atom[0])
        xyz.append(atom[1])

    return types,np.asarray(xyz)


class Optimizer(object):

    def __init__(self,Net,start_geom):
        self.Net=Net
        self.start_geom=start_geom
        self.types,self.x0=get_type_and_xyz(start_geom)
        self.nr_atoms=len(self.types)

    def to_nn_input(self,x):
        nn_geom=[]
        reshaped_x=np.asarray(x).reshape(self.nr_atoms,3)
        for i,type in enumerate(self.types):
            nn_geom.append((type,reshaped_x[i]))

        return nn_geom

    def fun(self,x):

        return self.Net.energy_for_geometry(self.to_nn_input(x))

    def der_fun(self,x):

        force=np.asarray(self.Net.force_for_geometry(self.to_nn_input(x)))
        grad=-force.flatten()

        return grad


    def start_bfgs(self):

        res=minimize(self.fun,jac=self.der_fun, x0=self.x0,method='BFGS')
        return self.to_nn_input(res.x)

    def start_nelder_mead(self):

        res=minimize(self.fun,self.x0,method='nelder-mead')
        return self.to_nn_input(res.x)