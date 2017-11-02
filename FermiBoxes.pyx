
#!python
#cython: cdivision=True
import warnings
import numpy as np

cdef extern from "math.h":
    double cos(double m)
    double sin(double m)
    double exp(double m)
    double tanh(double m)
    double cosh(double m)
    double sqrt(double m)
    double M_PI
    double atan2(double m)
    double pow(double m,double n)
         
    
cdef double eval_fermi_box(double [:,:] xyz,double [:,:,:,:] centers,double width,int N_atoms,double alpha,int n_dims):
    
    cdef double [:,:,:] out = np.zeros((n_dims,n_dims,n_dims))
    cdef double xi = 0
    cdef double yi = 0
    cdef double zi = 0
    cdef double x0 = 0
    cdef double y0 = 0
    cdef double z0 = 0
    cdef double center_x = 0
    cdef double center_y = 0
    cdef double center_z = 0
    
    #loop over every atom
    for i in range(N_atoms):
        #loop over every box 
        for j in range(n_dims):
            for k in range(n_dims):
                for l in range(n_dims):
                    xi=xyz[i,0]
                    yi=xyz[i,1]
                    zi=xyz[i,2]
                    center_x=centers[j,k,l,0]
                    center_y=centers[j,k,l,1]
                    center_z=centers[j,k,l,2]
                    #x border
                    if xi >= center_x:
                        x0 = center_x+width/2
                    else:
                        x0= center_x-width/2
                    #y border    
                    if yi >= center_y:
                        y0 = center_y+width/2
                    else:
                        y0= center_y-width/2
                    #z border    
                    if yi >= center_y:
                        z0 = center_z+width/2
                    else:
                        z0= center_z-width/2
                        
                    out[j,k,l]+=1/(exp((xi-x0)/alpha)+1)*1/(exp((yi-y0)/alpha)+1)*1/(exp((zi-z0)/alpha)+1)
            
            
        
def create_centers(n_boxes,box_width,start_center):
    
    centers=np.zeros((n_boxes,n_boxes,n_boxes))
    cdef double [:] vec_x = np.arange(start_center[0],start_center[0]+(n_boxes+1)*box_width,box_width)
    cdef double [:] vec_y= np.arange(start_center[1],start_center[1]+(n_boxes+1)*box_width,box_width)
    cdef double [:] vec_z = np.arange(start_center[2],start_center[2]+(n_boxes+1)*box_width,box_width)
    cdef double* xptr = &vec_x[0]
    cdef double* yptr = &vec_y[0]
    cdef double* zptr = &vec_z[0]
    cdef double [:,:,:,:] out  = np.zeros((3,n_boxes,n_boxes,n_boxes))
    
    for i in range(0,len(vec_x)):
        for j in range(0,len(vec_y)):
            for k in range(0,len(vec_z)):
                out[i,j,k,0]=xptr[i]
                out[i,j,k,1]=yptr[j]
                out[i,j,k,2]=zptr[k]
                
    return out 
    
    
    
    
    
class FermiBox(object):
    
   
    def __init__(self,n_boxes_per_dim=10,box_width=5,alpha=0.1,start_center=[-25,-25,-25]):
           
        self.box_vals=[]
        self.n_boxes_per_dim=n_boxes_per_dim
        self.box_width=box_width
        self.alpha=alpha
        self.start_center=start_center
        self.centers=create_centers(self.n_boxes_per_dim,self.box_width,self.start_center)
        
    def evaluate(self,xyz):
        
        self.box_vals=eval_fermi_box(xyz,self.centers,self.box_width,len(xyz[:,0]),self.alpha,self.n_boxes_per_dim)
        return self.box_vals
        
        