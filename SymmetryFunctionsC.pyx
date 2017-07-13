#!python
#cython: cdivision=True
import warnings

cdef extern from "math.h":
    double cos(double m)
    double sin(double m)
    double exp(double m)
    double tanh(double m)
    double cosh(double m)
    double sqrt(double m)
    double M_PI

cdef class CutoffFunction(object):
    
    def __init__(self, cut):
        if cut <= 0.:
            warnings.warn("Cutoff must be larger than 0")
            cut = 7.
        self.cut = cut 
    
cdef class CosCutoffFunction(CutoffFunction):
    
    cdef public double evaluate(self, double r):
        return (r <= self.cut)*0.5*(cos(M_PI*r/self.cut)+1)
        
    cdef public double derivative(self, double r):
        return (r <= self.cut)*(-0.5*M_PI*sin(M_PI*r/self.cut)/self.cut)
        
        
cdef class TanhCutoffFunction(CutoffFunction):
    
    cdef public double evaluate(self, double r):
        return (r <= self.cut)*tanh(1.-r/self.cut)**3      
        
    cdef public double derivative(self, double r):
        return (r <= self.cut)*(-3.0*tanh(1-r/self.cut)**2 /
                (cosh(1-r/self.cut)**2*self.cut))
    
    
cdef class RadialSymmetryFunction(object):
    
    def __init__(self, rs, eta, cut, cutoff_type = "cos"):
        self.rs = rs
        self.eta = eta
        self.cut = cut
        if cutoff_type == "cos":
            self.cut_fun = CosCutoffFunction(cut)
            self.cut_type=1
        elif cutoff_type == "tanh":
            warnings.warn("TanhCutoffFunction disabled in C")
            self.cut_fun = TanhCutoffFunction(cut)
            self.cut_type=1#2
        else:
            warnings.warn("{} not recognized, switching to 'cos'".format(cutoff_type),
                          UserWarning)
            self.cut_fun = CosCutoffFunction(cut)
        
    cdef public double evaluate(self, double r):
        return exp(-self.eta*(r-self.rs)**2)*self.cut_fun.evaluate(r)
        
    cdef public double derivative(self, double r):
        return (-2.0*self.eta*(r-self.rs)*self.evaluate(r) +
                exp(-self.eta*(r-self.rs)**2)*self.cut_fun.derivative(r))

cdef class AngularSymmetryFunction(object):
    
    def __init__(self, eta, zeta, lamb, cut, cutoff_type = "cos"):
        self.eta = eta
        self.zeta = zeta
        self.lamb = lamb
        self.cut=cut
        if cutoff_type == "cos":
            self.cut_fun = CosCutoffFunction(cut)
            self.cut_type=1
        elif cutoff_type == "tanh":
            warnings.warn("TanhCutoffFunction disabled in C")
            self.cut_fun = TanhCutoffFunction(cut)
            self.cut_type=1#2
        else:
            warnings.warn("{} not recognized, switching to 'cos'".format(cutoff_type),
                          UserWarning)
            self.cut_fun = CosCutoffFunction(cut)
        
    cdef public double evaluate(self, double rij, double rik, double costheta):
        return 2**(1-self.zeta)* ((1 + self.lamb*costheta)**self.zeta * 
            exp(-self.eta*(rij**2+rik**2)) * self.cut_fun.evaluate(rij)*self.cut_fun.evaluate(rik))
            
    cdef public double derivative(self, double rij, double rik, double costheta):
        # Not nice but a simple workaround for very small negative numbers in sqrt
        sintheta = sqrt(abs(1.-costheta**2))
        return 2**(1-self.zeta)*(-self.lamb * self.zeta * sintheta *
            (1 + self.lamb*costheta)**(self.zeta-1) * 
            exp(-self.eta*(rij**2+rik**2)) * self.cut_fun.evaluate(rij)*self.cut_fun.evaluate(rik))
        
#class AngularSymmetryFunctionNew(object):
#    def __init__(self, eta, zeta, lamb, rs, cut, cutoff_type = "cos"):
#        self.eta = eta
#        self.zeta = zeta
#        self.lamb = lamb
#        if cutoff_type == "cos":
#            self.cut_fun = CosCutoffFunction(cut)
#        elif cutoff_type == "tanh":
#            self.cut_fun = TanhCutoffFunction(cut)
#        else:
#            warnings.warn("{} not recognized, switching to 'cos'".format(cutoff_type),
#                          UserWarning)
#            self.cut_fun = CosCutoffFunction(cut)
#        
#    def __call__(self, rij, rik, costheta):
#        return 2**(1-self.zeta)* ((1 + self.lamb*costheta)**self.zeta * 
#            _np.exp(-self.eta*((rij-self.rs)**2+(rik-self.rs)**2)) * self.cut_fun.evaluate(rij)*self.cut_fun.evaluate(rik))
#            
#    def derivative(self, rij, rik, costheta):
#        sintheta = _np.sqrt(1-costheta**2)
#        return 2**(1-self.zeta)*(-self.lamb * self.zeta * sintheta *
#            (1 + self.lamb*costheta)**(self.zeta-1) * 
#            _np.exp(-self.eta*((rij-self.rs)**2+(rik-self.rs)**2)) * self.cut_fun.evaluate(rij)*self.cut_fun.evaluate(rik))

#def radial_function(rij, rs, eta, cut):
#    return _np.exp(-eta*(rij-rs)**2)*cutoff_function(rij,cut)
    
#def angular_function(rij, rik, costheta, eta, zeta, lamb,  cut):
#    return 2**(1-zeta)* ((1 + lamb*costheta) * _np.exp(-eta*(rij**2+rik**2)) *
#        cutoff_function(rij,cut)*cutoff_function(rik,cut))
        
#def cutoff_function(r,cut):
#    # Weird formulation allows for calls using vectors
#    return (r <= cut)*0.5*(_np.cos(_np.pi*r/cut)+1)
