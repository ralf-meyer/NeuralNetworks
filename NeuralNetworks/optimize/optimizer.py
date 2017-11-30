import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from scipy.optimize import approx_fprime

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as _np

def get_type_and_xyz(geom):
    xyz=[]
    types=[]
    for atom in geom:
        types.append(atom[0])
        xyz.append(atom[1])

    return types,_np.asarray(xyz)


class Optimizer(object):


    def __init__(self,Net,start_geom):
        self.Net=Net
        self.start_geom=start_geom
        self.types,self.x0=get_type_and_xyz(start_geom)
        self.nr_atoms=len(self.types)
        self.plot=False
        self.fig = None
        self.ax = None
        self.scat = None
        self.update_with_energy=False

    def init_plot(self,x):
        """Initialize the scatter plot"""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scat = self.ax.scatter( x[:,0] ,
                                     x[:,1] ,
                                     x[:,2] ,
                                     animated=False , marker='o' , alpha=None , s=150)
        plt.show(block=False)


    def update_plot(self,x):
        """Update the scatter plot."""
        reshaped_x = _np.asarray(x).reshape(self.nr_atoms, 3)
        self.scat._offsets3d = (_np.ma.ravel(reshaped_x[:, 0]),
                                _np.ma.ravel(reshaped_x[:, 1]),
                                _np.ma.ravel(reshaped_x[:, 2])
                                )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return self.scat,



    def to_nn_input(self,x):
        """Converts the raw geometry into a
        ("type",_np.array(geometry)) geometry"""

        nn_geom=[]
        reshaped_x=_np.asarray(x).reshape(self.nr_atoms,3)
        for i,type in enumerate(self.types):
            nn_geom.append((type,reshaped_x[i]))

        return nn_geom

    def fun(self,x):
        """Wrapper for energy evaluation"""
        if self.update_with_energy:
            self.update_plot(x)
        return self.Net.energy_for_geometry(self.to_nn_input(x))

    def der_fun(self,x):
        """TODO: Fix force calculation"""
        if not(self.update_with_energy):
            self.update_plot(x)
        force=_np.asarray(self.Net.force_for_geometry(self.to_nn_input(x)))
        grad=-force.flatten()

        return grad

    def der_fun_approx(self,x):

        return approx_fprime(x.flatten(),self.fun,epsilon=1e-7)


    def check_gradient(self):
        if self.plot:
            self.init_plot(self.x0)
        analytic=self.der_fun(self.x0)
        approx=self.der_fun_approx(self.x0)
        print([analytic,approx])
        print([_np.abs(analytic-approx)])

    def start_bfgs(self):
        """Starts a geometry optimization using the BFGS method"""
        if self.plot:
            self.init_plot(self.x0)
        #res=minimize(self.fun,jac=self.der_fun, x0=self.x0,tol=1e-6,method='BFGS',options={'disp': True, 'gtol': 1e-6})
        [res,fopt,gopt,Bopt, func_calls, grad_calls, warnflg]=fmin_bfgs(self.fun,
                                                                        self.x0,
                                                                        self.der_fun,
                                                                        gtol=1e-09,
                                                                        full_output=True,
                                                                        disp=True)
        print("E optimal = "+str(fopt))
        print("F optimal = "+str(gopt))
        return self.to_nn_input(res)

    def start_nelder_mead(self):
        """Starts a geometry optimization using the nelder-mead method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,method='nelder-mead',tol=1e-3,options={'disp': True})
        return self.to_nn_input(res.x)