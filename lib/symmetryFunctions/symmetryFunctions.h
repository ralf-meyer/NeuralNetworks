/*

CAUTION: Part of this file is written by the python script generateSymFuns.py.
These parts are marked by comment lines starting with AUTOMATIC. Do not alter
anything between these tags.
*/

#include "cutoffFunctions.h"
#include <memory>

class SymmetryFunction
{
    public:
        SymmetryFunction(int num_prms, double* pmrs_i,
          std::shared_ptr<CutoffFunction> cutfun_i);
        ~SymmetryFunction();
    protected:
        double* prms;
        std::shared_ptr<CutoffFunction> cutfun;
};

class TwoBodySymmetryFunction: public SymmetryFunction
{
    public:
        TwoBodySymmetryFunction(int num_prms, double* prms_i,
          std::shared_ptr<CutoffFunction> cutfun_i):
          SymmetryFunction(num_prms, prms_i, cutfun_i){};
        virtual double eval(double rij) = 0;
        virtual double drij(double rij) = 0;
};

// AUTOMATIC Start of custom TwoBodySymFuns

class BehlerG2: public TwoBodySymmetryFunction
{
    public:
        BehlerG2(int num_prms, double* prms_i,
          std::shared_ptr<CutoffFunction> cutfun_i):
          TwoBodySymmetryFunction(num_prms, prms_i, cutfun_i){};
        double eval(double rij);
        double drij(double rij);
};
// AUTOMATIC End of custom TwoBodySymFuns

class ThreeBodySymmetryFunction: public SymmetryFunction
{
    public:
      ThreeBodySymmetryFunction(int num_prms, double* prms,
        std::shared_ptr<CutoffFunction> cutfun_i):
        SymmetryFunction(num_prms, prms, cutfun_i){};
      virtual double eval(double rij, double rik, double costheta) = 0;
      virtual double drij(double rij, double rik, double costheta) = 0;
      virtual double drik(double rij, double rik, double costheta) = 0;
      virtual double dcostheta(double rij, double rik, double costheta) = 0;
      virtual void derivatives(double rij, double rik, double costheta,
        double &drij, double &drik, double &dcostheta) = 0;
};

// AUTOMATIC Start of custom ThreeBodySymFuns

class BehlerG4: public ThreeBodySymmetryFunction
{
  public:
    BehlerG4(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun_i):
      ThreeBodySymmetryFunction(num_prms, prms, cutfun_i){};
    double eval(double rij, double rik, double costheta);
    double drij(double rij, double rik, double costheta);
    double drik(double rij, double rik, double costheta);
    double dcostheta(double rij, double rik, double costheta);
    void derivatives(double rij, double rik, double costheta,
      double &drij, double &drik, double &dcostheta);
};
// AUTOMATIC End of custom ThreeBodySymFuns

std::shared_ptr<CutoffFunction> switch_CutFun(
  int cutoff_type, double cutoff);
std::shared_ptr<TwoBodySymmetryFunction> switch_TwoBodySymFun(
  int funtype, int num_prms, double* prms,
  std::shared_ptr<CutoffFunction> cutfun);
std::shared_ptr<ThreeBodySymmetryFunction> switch_ThreeBodySymFun(
  int funtype, int num_prms, double* prms,
  std::shared_ptr<CutoffFunction> cutfun);
int get_CutFun_by_name(const char* name);
int get_TwoBodySymFun_by_name(const char* name);
int get_ThreeBodySymFun_by_name(const char* name);
void available_symFuns();
