#include "symmetryFunctions.h"
#include <stdio.h>
#include <math.h>
#include <limits>

SymmetryFunction::SymmetryFunction(int num_prms, double* prms_i, std::shared_ptr<CutoffFunction> cutfun_i):
cutfun(cutfun_i){
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
/*
double BehlerG2::eval(double rij)
{
  return exp(-prms[0]*pow(rij-prms[1],2))*cutfun->eval(rij);
};

double BehlerG2::drij(double rij)
{
  return -2*prms[0]*(rij-prms[1])*cutfun->eval(rij) +
    exp(-prms[0]*pow(rij-prms[1],2))*cutfun->derivative(rij);
};*/

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

/*
double BehlerG4::eval(double rij, double rik, double theta)
{
  return pow(2.0, 1 - prms[1])*pow((1 + prms[0]*cos(theta)), prms[1]) *
    exp(-prms[2]*(pow(rij,2)+pow(rik,2)))*cutfun->eval(rij)*cutfun->eval(rik);
};

double BehlerG4::drij(double rij, double rik, double theta)
{
  //-2*2**(-zeta + 1)*eta*rij*(lambda*cos(theta) + 1)**zeta*cutfun_ij(rij)*cutfun_ik(rik)*exp(-eta*(rij**2 + rik**2)) 
  //+ 2**(-zeta + 1)*(lambda*cos(theta) + 1)**zeta*cutfun_ik(rik)*exp(-eta*(rij**2 + rik**2))*Derivative(cutfun_ij(rij), rij)
  return -2.0*pow(2.0,1-prms[1])*prms[2]*rij*pow(1 + prms[0]*cos(theta), prms[1])*exp(-prms[2]*(pow(rij,2)+pow(rik,2)))*cutfun->eval(rij)*cutfun->eval(rik) +
  pow(2.0, 1 - prms[1])*pow((1 + prms[0]*cos(theta)), prms[1]) * exp(-prms[2]*(pow(rij,2)+pow(rik,2)))*cutfun->derivative(rij)*cutfun->eval(rik);
};

double BehlerG4::drik(double rij, double rik, double theta)
{
  return -2.0*pow(2.0,1-prms[1])*prms[2]*rik*pow(1 + prms[0]*cos(theta), prms[1])*exp(-prms[2]*(pow(rij,2)+pow(rik,2)))*cutfun->eval(rij)*cutfun->eval(rik) +
  pow(2.0, 1 - prms[1])*pow((1 + prms[0]*cos(theta)), prms[1]) * exp(-prms[2]*(pow(rij,2)+pow(rik,2)))*cutfun->eval(rij)*cutfun->derivative(rik);
};

double BehlerG4::dtheta(double rij, double rik, double theta)
{
  //-2**(-zeta + 1)*lamb*zeta*(lamb*cos(theta) + 1)**zeta*exp(-eta*(rij**2 + rik**2))*sin(theta)/(lamb*cos(theta) + 1)
  return -pow(2.0, 1 - prms[1])*prms[0]*prms[1]*pow(1 + prms[0]*cos(theta), prms[1])*
    exp(-prms[2]*(pow(rij,2)+pow(rik,2)))*sin(theta)/(prms[0]*cos(theta) + 1 + std::numeric_limits<double>::epsilon())*
    cutfun->eval(rij)*cutfun->eval(rik);
};*/
