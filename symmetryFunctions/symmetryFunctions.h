#include "cutoffFunctions.h"

// Figure out how abstract classes work
class SymmetryFunction
{
    public:
        SymmetryFunction(int num_prms, double* pmrs_i, CutoffFunction* cutfun_i);
        ~SymmetryFunction();
    protected:
        double* prms;
        CutoffFunction* cutfun;
};

class TwoBodySymmetryFunction: public SymmetryFunction
{
    public:
        TwoBodySymmetryFunction(int num_prms, double* prms_i, CutoffFunction* cutfun_i):
          SymmetryFunction(num_prms, prms_i, cutfun_i){};
        virtual double eval(double rij) = 0;
        virtual double drij(double rij) = 0;
};

// Start of custom TwoBodySymFuns

class BehlerG2: public TwoBodySymmetryFunction
{
    public:
        BehlerG2(int num_prms, double* prms_i, CutoffFunction* cutfun_i):
          TwoBodySymmetryFunction(num_prms, prms_i, cutfun_i){};
        double eval(double rij);
        double drij(double rij);
};
// End of custom TwoBodySymFuns

class ThreeBodySymmetryFunction: public SymmetryFunction
{
    public:
      ThreeBodySymmetryFunction(int num_prms, double* prms, CutoffFunction* cutfun_i):
        SymmetryFunction(num_prms, prms, cutfun_i){};
      virtual double eval(double rij, double rik, double theta) = 0;
      virtual double drij(double rij, double rik, double theta) = 0;
      virtual double drik(double rij, double rik, double theta) = 0;
      virtual double dtheta(double rij, double rik, double theta) = 0;
};

// Start of custom ThreeBodySymFuns

class BehlerG4: public ThreeBodySymmetryFunction
{
  public:
    BehlerG4(int num_prms, double* prms, CutoffFunction* cutfun_i):
      ThreeBodySymmetryFunction(num_prms, prms, cutfun_i){};
    double eval(double rij, double rik, double theta);
    double drij(double rij, double rik, double theta);
    double drik(double rij, double rik, double theta);
    double dtheta(double rij, double rik, double theta);
};
// End of custom ThreeBodySymFuns

/*class BehlerG2: public TwoBodySymmetryFunction
{
    public:
        //BehlerG2(double* prms_i, CutoffFunction cutfun_i);
        BehlerG2(int num_prms, double* prms_i, CutoffFunction* cutfun_i):
          TwoBodySymmetryFunction(num_prms, prms_i, cutfun_i){};
        double eval(double rij);
        double drij(double rij);
};

class BehlerG4: public ThreeBodySymmetryFunction
{
  public:
    BehlerG4(int num_prms, double* prms, CutoffFunction* cutfun_i):
      ThreeBodySymmetryFunction(num_prms, prms, cutfun_i){};
    double eval(double rij, double rik, double theta);
    double drij(double rij, double rik, double theta);
    double drik(double rij, double rik, double theta);
    double dtheta(double rij, double rik, double theta);
};*/
