from NeuralNetworks import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import time
import os
plt.ion()


class PES(object):

    def __init__(self,model_name,Instance=None,resolution=50):
        self.geoms=[]
        self.figure=plt.figure()
        self.ax = Axes3D(self.figure)
        #self.ax.set_zlim(bottom=-1,top=5)
        plt.title(model_name)
        self.resolution=resolution
        self.theta=np.linspace(10*np.pi/180,350*np.pi/180,self.resolution)
        self.rhs=np.linspace(1,8,self.resolution)
        self.meshX,self.meshY=np.meshgrid(self.rhs,self.theta*180/np.pi)
        meshZ=np.zeros((len(self.theta),len(self.rhs)))
        self.surf=self.ax.plot_surface(self.meshX,self.meshY,meshZ,cmap=cm.magma,animated=True)
        self.model=model_name
        self.create_geoms()
        self.Evaluation = NN.AtomicNeuralNetInstance()
        self.Evaluation.CalcDatasetStatistics=False
        self.Evaluation.TextOutput=False
        if Instance!=None:
            self.Evaluation._IsFromCheck=True
            self.Evaluation.Rs=Instance.Rs
            self.Evaluation.R_Etas=Instance.R_Etas
            self.Evaluation.NumberOfRadialFunctions=Instance.NumberOfRadialFunctions
            self.Evaluation.Etas=Instance.Etas
            self.Evaluation.Lambs=Instance.Lambs
            self.Evaluation.Zetas=Instance.Zetas
        self.Data=None
        self.ZData=[]


    def create_geoms(self):

        ct = 0
        geom = []
        xs = []
        ys = []

        for w in self.theta:
            for r in self.rhs:
                for i in range(2):
                    if ct == 0:
                        x = r
                        y = 0
                        z = 0
                        ct = ct + 1
                        geom.append(("X", np.zeros(3)))
                        h1 = [x, y, z]
                        geom.append(("Y", np.asarray(h1)))
                    else:
                        x = r * np.cos(w)
                        y = r * np.sin(w)
                        z = 0
                        h2 = [x, y, z]
                        geom.append(("Y", np.asarray(h2)))
                        self.geoms.append(geom)
                        ct = 0
                        geom = []

    def pes_check(self,atom_types=["X","Y"],nr_atoms_per_type=[2,1],show_plot=True):
        self.Evaluation.prepare_evaluation(self.model,nr_atoms_per_type=nr_atoms_per_type,atom_types=atom_types)
        if self.Data ==None:
            self.Data = self.Evaluation.create_eval_data(self.geoms)
        out=self.Evaluation.eval_dataset_energy(self.Data,0)
        out=out-np.min(out)
        meshZ=out.reshape(len(self.theta),len(self.rhs))
        self.ZData.append(meshZ)
        #self.surf.remove()
        self.surf=self.ax.plot_surface(self.meshX,self.meshY,meshZ,cmap=cm.magma)
        self.ax.set_xlabel(r'$r / \AA$')
        self.ax.set_ylabel(r'$\phi / \degree$')
        self.ax.set_zlabel("Neuron output")
        if show_plot:
            plt.show(block=False)



if __name__ == "__main__":
    plt.ioff()
    #file="/home/afuchs/Git/NeuralNetworks/NeuralNetworks/training/save_no_name"
    file="/home/afuchs/Documents/NiAu_Training/multi_no_angular_force"
    MyCheck1=PES(file)
    MyCheck1.pes_check(atom_types=["X","Y"],nr_atoms_per_type=[1,2],show_plot=False)
    MyCheck2 = PES(file)
    MyCheck2.pes_check(atom_types=["Y", "X"], nr_atoms_per_type=[2, 1],show_plot=False)
    plt.show()