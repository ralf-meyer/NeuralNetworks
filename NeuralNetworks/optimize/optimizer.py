import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from scipy.optimize import approx_fprime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as _np
import os as _os
from numpy import linalg as la

def get_type_and_xyz(geom):
    xyz=[]
    types=[]
    diff_types=[]
    for atom in geom:
        types.append(atom[0])
        xyz.append(atom[1])
        if atom[0] not in diff_types:
            diff_types.append(atom[0])

    return diff_types,types,_np.asarray(xyz)


class Optimizer(object):


    def __init__(self,Net,start_geom):
        self.Net=Net
        self.start_geom=start_geom
        self.diff_types,self.types,self.x0=get_type_and_xyz(start_geom)
        self.nr_atoms=len(self.types)
        self.plot=False
        self.fig = None
        self.ax = None
        self.scat = None
        self.update_with_energy=False
        self.save_png=False
        self.png_path=""
        self.counter=0

    def init_plot(self,x):
        """Initialize the scatter plot"""

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        #Get coloring for atoms
        my_colors=[]
        for i,atom in enumerate(self.start_geom):
            type=atom[0]
            if type=="Ni":
                my_colors.append('b')
            elif type=="Au":
                my_colors.append('y')
            else:
                my_colors.append('k')

        self.scat = self.ax.scatter( x[:,0] ,
                                     x[:,1] ,
                                     x[:,2] ,
                                     animated=False , marker='o' , alpha=None , s=150,c=my_colors)
        if self.save_png:
            if not _os.path.exists(self.png_path):
                _os.makedirs(self.png_path)

            self.fig.savefig(_os.path.join(self.png_path,"pic_"+str(self.counter)))
        self.counter+=1

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

        if self.save_png:
            self.fig.savefig(_os.path.join(self.png_path, "pic_" + str(self.counter)))
        self.counter+=1

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

    def isPD(self,B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    def nearestPD(self,A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """
        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = _np.dot(V.T, _np.dot(_np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = _np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = _np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = _np.min(_np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    def hessian_approx(self,x,epsilon=1.e-5):
        """
        A numerical approximation to the Hessian matrix of cost function at
        location x0 (hopefully, the minimum)
        """
        # ``calculate_cost_function`` is the cost function implementation
        # The next line calculates an approximation to the first
        # derivative
        f1 = self.der_fun(x)

        # Allocate space for the hessian
        n = x.shape[0]
        hessian = _np.zeros((n, n))
        # The next loop fill in the matrix
        xx = x
        for j in xrange(n):
            xx0 = xx[j]  # Store old value
            xx[j] = xx0 + epsilon  # Perturb with finite difference
            # Recalculate the partial derivatives for this new point
            f2 = self.der_fun(xx)
            hessian[:, j] = (f2 - f1) / epsilon  # scale...
            xx[j] = xx0  # Restore initial value of x0


        hessian=self.nearestPD(hessian)

        return hessian


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
        """Starts a geometry optimization using the Nelder-Mead method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='nelder-mead',
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_powell(self):
        """Starts a geometry optimization using the Powell method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='powell',
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_conjugate_gradient(self):
        """Starts a geometry optimization using the conjugate gradient method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='cg',
                     jac=self.der_fun,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_newton_cg(self):
        """Starts a geometry optimization using the Newton conjugate gradient method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='Newton-CG',
                     jac=self.der_fun,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_l_bfgs_b(self):
        """Starts a geometry optimization using the L-BFGS-B method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='L-BFGS-B',
                     jac=self.der_fun,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_tnc(self):
        """Starts a geometry optimization using the TNC method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='TNC',
                     jac=self.der_fun,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_cobyla(self):
        """Starts a geometry optimization using the COBYLA method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='COBYLA',
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_slsqp(self):
        """Starts a geometry optimization using the SLSQP method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='SLSQP',
                     jac=self.der_fun,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_dogleg(self):
        """Starts a geometry optimization using the dogleg method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,method='dogleg',
                     jac=self.der_fun,
                     hess=self.hessian_approx,tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_trust_ncg(self):
        """Starts a geometry optimization using the trust-ncg method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='trust-ncg',
                     jac=self.der_fun,
                     hess=self.hessian_approx,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_trust_exact(self):
        """Starts a geometry optimization using the trust-exact method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='trust-exact',
                     jac=self.der_fun,
                     hess=self.hessian_approx,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)

    def start_trust_krylov(self):
        """Starts a geometry optimization using the trust-krylov method"""
        if self.plot:
            self.init_plot(self.x0)
            self.update_with_energy=True
        res=minimize(self.fun,self.x0,
                     method='trust-krylov',
                     jac=self.der_fun,
                     hess=self.hessian_approx,
                     tol=1e-3,
                     options={'disp': True})
        return self.to_nn_input(res.x)