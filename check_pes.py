from NeuralNetworks import NeuralNetworkUtilities as NN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()



def pes_check(model_name):

    theta=np.linspace(0.1*np.pi/180,360*np.pi/180,40)
    rhs=np.linspace(2,5,40)

    ct=0
    geoms=[]
    geom=[]
    xs=[]
    ys=[]

    for w in theta:
        for r in rhs:
            for i in range(2):
                if ct==0:
                    x=r
                    y=0
                    z=0
                    ct=ct+1
                    geom.append(("Au",np.zeros(3)))
                    h1=[x,y,z]
                    geom.append(("Au",np.asarray(h1)))
                else:
                    x=r*np.cos(w)
                    y=r*np.sin(w)
                    z=0
                    h2=[x,y,z]
                    geom.append(("Au",np.asarray(h2)))
                    geoms.append(geom)
                    ct=0
                    geom=[]

    Evaluation=NN.AtomicNeuralNetInstance()
    Evaluation.prepare_evaluation(model_name,atom_types=["Au"],nr_atoms_per_type=[3])
    Evaluation.create_eval_data(geoms)
    out=Evaluation.eval_dataset_energy(Evaluation.EvalData,0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    meshX,meshY=np.meshgrid(rhs,theta*180/np.pi)
    meshZ=out.reshape(len(theta),len(rhs))
    ax.plot_surface(meshX,meshY,meshZ)
    m=np.min(meshZ)
    L=meshZ==m
    print(str(meshX[L])+" Angstroem")
    print(str(meshY[L])+" grad")
    plt.show()

model="/home/afuchs/Documents/Au_training/Au_test"
pes_check(model_name=model)