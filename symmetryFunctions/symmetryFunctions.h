#include "cutoffFunctions.h"

// Figure out how abstract classes work
class SymmetryFunction
{
    public:
        SymmetryFunction(double* pmrs_i, CutoffFunction cutfun_i);
        double eval(double* args);
        double derivative(double* args);
    protected:
        double* prms;
        CutoffFunction cutfun;
};

class TwoBodySymmetryFunction: public SymmetryFunction
{
    public:
        double eval(double r);
        double derivative(double r);
};

class ThreeBodySymmetryFunction: public SymmetryFunction
{
    public:
        double eval(double rij, double rik, double theta);
        double derivative(double rij, double rik, double theta);
};

class BehlerG2: public TwoBodySymmetryFunction
{
    public:
        //BehlerG2(double* prms_i, CutoffFunction cutfun_i);
        double eval(double r);
        double derivative(double r);
};
