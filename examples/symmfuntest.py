import numpy as np
import matplotlib.pyplot as plt
import NeuralNetworkUtilities as NN


def cos(a,b,x):
    return np.cos(a*x+b)

def d_cos(a,b,x):
    return a*np.sin(a*x+b)

def gaussian(a,b,x):
    
    return np.exp(-a*(x-b)**2)*np.cos(np.sqrt(3)*a*(x-b)/2)

#x=np.linspace(-np.pi/2,np.pi/2,1000)
#my_a=[1.5,3,6,9]
#my_b=[-np.pi/2,0,np.pi/2,np.pi]
#
#fig=plt.figure()
#
#for a in my_a:
#    for b in my_b:
#        #plt.plot(x,gaussian(a,b,x))
#        plt.plot(x,cos(a,b,x))
#        #plt.plot(x,d_cos(a,b,x))

#Load first trainings data 
Training=NN.AtomicNeuralNetInstance()

Training.XYZfile="2ClusterNiAu_data.xyz"
Training.Logfile="2cluster.md"
Training.Atomtypes=["1","2"]
Training.NumberOfRadialFunctions=0
#angular symmetry function settings
Training.Lambs=[1.0,-1.0]
Training.Zetas=[1,1.5,2,3,5,10,18]
Training.Etas=[0.1,0.2,0.5,1]
Training.Rs=[0,1,1.5,2,2.5,3,4,5,6]


Training.read_files()
Training.make_training_and_validation_data(100,70,30)
NrNi=12
NrAu=14

Training.Structures.append([Training.SizeOfInputs[0],10,1])
Training.NumberOfSameNetworks.append(NrNi)
Training.Structures.append([Training.SizeOfInputs[1],10,1])
Training.NumberOfSameNetworks.append(NrAu)
Training.HiddenType="truncated_normal"
Training.ActFun="elu"
Training.LearningRate=0.01
Training.Epochs=1500
Training.MakePlots=True
Training.OptimizerType="Adam"
Training.Regularization="L1"
Training.CostFunType="Adaptive_2"
Training.LearningRateType="exponential_decay"
Training.SavingDirectory="save_weights"
Training.LearningDecayEpochs=100
Training.RegularizationParam=0.2
#Training.expand_existing_net()
Training.make_and_initialize_network()

#Start training

Training.start_batch_training(True)
