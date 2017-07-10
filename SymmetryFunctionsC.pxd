cdef class CutoffFunction(object):
    cdef double cut
    
cdef class CosCutoffFunction(CutoffFunction):
    
    cdef public double evaluate(self, double r)      
    cdef public double derivative(self, double r)
        
cdef class TanhCutoffFunction(CutoffFunction):
    
    cdef public double evaluate(self, double r)
    cdef public double derivative(self, double r)
    
cdef class RadialSymmetryFunction(object):
    cdef double rs
    cdef double eta
    cdef CosCutoffFunction cut_fun;
    
    cdef public double evaluate(self, double r)
    cdef public double derivative(self, double r)

cdef class AngularSymmetryFunction(object):
    cdef double eta
    cdef double zeta
    cdef double lamb
    cdef CosCutoffFunction cut_fun
    
    cdef public double evaluate(self, double rij, double rik, double costheta)
    cdef public double derivative(self, double rij, double rik, double costheta)