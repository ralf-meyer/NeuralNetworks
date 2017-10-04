#include "cutoffFunctions.h"
#include <memory>

class SymmetryFunction
{
    public:
        SymmetryFunction(int num_prms, double* pmrs_i, std::shared_ptr<CutoffFunction> cutfun_i);
        ~SymmetryFunction();
    protected:
        double* prms;
        std::shared_ptr<CutoffFunction> cutfun;
};

class TwoBodySymmetryFunction: public SymmetryFunction
{
    public:
        TwoBodySymmetryFunction(int num_prms, double* prms_i, std::shared_ptr<CutoffFunction> cutfun_i):
          SymmetryFunction(num_prms, prms_i, cutfun_i){};
        virtual double eval(double rij) = 0;
        virtual double drij(double rij) = 0;
};

// AUTOMATIC Start of custom TwoBodySymFuns

class BehlerG2: public TwoBodySymmetryFunction
{
    public:
        BehlerG2(int num_prms, double* prms_i, std::shared_ptr<CutoffFunction> cutfun_i):
          TwoBodySymmetryFunction(num_prms, prms_i, cutfun_i){};
        double eval(double rij);
        double drij(double rij);
};
// AUTOMATIC End of custom TwoBodySymFuns

class ThreeBodySymmetryFunction: public SymmetryFunction
{
    public:
      ThreeBodySymmetryFunction(int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun_i):
        SymmetryFunction(num_prms, prms, cutfun_i){};
      virtual double eval(double rij, double rik, double theta) = 0;
      virtual double drij(double rij, double rik, double theta) = 0;
      virtual double drik(double rij, double rik, double theta) = 0;
      virtual double dtheta(double rij, double rik, double theta) = 0;
};

// AUTOMATIC Start of custom ThreeBodySymFuns

class BehlerG4: public ThreeBodySymmetryFunction
{
  public:
    BehlerG4(int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun_i):
      ThreeBodySymmetryFunction(num_prms, prms, cutfun_i){};
    double eval(double rij, double rik, double theta);
    double drij(double rij, double rik, double theta);
    double drik(double rij, double rik, double theta);
    double dtheta(double rij, double rik, double theta);
};
// AUTOMATIC End of custom ThreeBodySymFuns

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
