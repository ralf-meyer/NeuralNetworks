/*

CAUTION: Part of this file is written by the python script generateSymFuns.py.
These parts are marked by comment lines starting with AUTOMATIC. Do not alter
anything between these tags.
*/

#include "symmetryFunctions.h"
#include <stdio.h>
#include <math.h>
#include <limits>

SymmetryFunction::SymmetryFunction(int num_prms, double* prms_i,
  std::shared_ptr<CutoffFunction> cutfun_i):cutfun(cutfun_i)
{
  prms = new double[num_prms];
  for (int i = 0; i < num_prms; i++)
  {
    prms[i] = prms_i[i];
  }
};

SymmetryFunction::~SymmetryFunction()
{
  delete[] prms;
};

// AUTOMATIC Start of custom TwoBodySymFuns

double BehlerG2::eval(double rij)
{
  return cutfun->eval(rij)*exp(-prms[0]*pow(-prms[1] + rij, 2));
};

double BehlerG2::drij(double rij)
{
  return -prms[0]*(-2*prms[1] + 2*rij)*cutfun->eval(rij)*exp(-prms[0]*pow(-prms[1] + rij, 2)) + cutfun->derivative(rij)*exp(-prms[0]*pow(-prms[1] + rij, 2));
};
// AUTOMATIC End of custom TwoBodySymFuns

// AUTOMATIC Start of custom ThreeBodySymFuns

double BehlerG4::eval(double rij, double rik, double theta)
{
  return pow(2.0, -prms[1] + 1)*pow(prms[0]*cos(theta) + 1, prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::drij(double rij, double rik, double theta)
{
  return -2*pow(2.0, -prms[1] + 1)*prms[2]*rij*pow(prms[0]*cos(theta) + 1, prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2))) + pow(2.0, -prms[1] + 1)*pow(prms[0]*cos(theta) + 1, prms[1])*cutfun->derivative(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::drik(double rij, double rik, double theta)
{
  return -2*pow(2.0, -prms[1] + 1)*prms[2]*rik*pow(prms[0]*cos(theta) + 1, prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2))) + pow(2.0, -prms[1] + 1)*pow(prms[0]*cos(theta) + 1, prms[1])*cutfun->derivative(rik)*cutfun->eval(rij)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::dtheta(double rij, double rik, double theta)
{
  return -pow(2.0, -prms[1] + 1)*prms[0]*prms[1]*pow(prms[0]*cos(theta) + 1, prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)))*sin(theta)/(prms[0]*cos(theta) + 1);
};
// AUTOMATIC End of custom ThreeBodySymFuns
