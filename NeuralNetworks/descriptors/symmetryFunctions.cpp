/*

CAUTION: Part of this file is written by the python script generateSymFuns.py.
These parts are marked by comment lines starting with AUTOMATIC. Do not alter
anything between these tags.
*/

#include "symmetryFunctions.h"
#include <stdio.h>
#include <math.h>
#include <limits>
#include <string.h>

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
  return (2*prms[0]*(prms[1] - rij)*cutfun->eval(rij) + cutfun->derivative(rij))*exp(-prms[0]*pow(prms[1] - rij, 2));
};

void BehlerG2::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  auto x1 = prms[1] - rij;
  auto x2 = exp(-prms[0]*pow(x1, 2));
  G = x0*x2;
  dGdrij = x2*(2*prms[0]*x0*x1 + cutfun->derivative(rij));
};

double OneOverR6::eval(double rij)
{
  return cutfun->eval(rij)/pow(rij, 6);
};

double OneOverR6::drij(double rij)
{
  return (rij*cutfun->derivative(rij) - 6*cutfun->eval(rij))/pow(rij, 7);
};

void OneOverR6::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  G = x0/pow(rij, 6);
  dGdrij = (rij*cutfun->derivative(rij) - 6*x0)/pow(rij, 7);
};

double OneOverR8::eval(double rij)
{
  return cutfun->eval(rij)/pow(rij, 8);
};

double OneOverR8::drij(double rij)
{
  return (rij*cutfun->derivative(rij) - 8*cutfun->eval(rij))/pow(rij, 9);
};

void OneOverR8::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  G = x0/pow(rij, 8);
  dGdrij = (rij*cutfun->derivative(rij) - 8*x0)/pow(rij, 9);
};

double OneOverR10::eval(double rij)
{
  return cutfun->eval(rij)/pow(rij, 10);
};

double OneOverR10::drij(double rij)
{
  return (rij*cutfun->derivative(rij) - 10*cutfun->eval(rij))/pow(rij, 11);
};

void OneOverR10::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  G = x0/pow(rij, 10);
  dGdrij = (rij*cutfun->derivative(rij) - 10*x0)/pow(rij, 11);
};
// AUTOMATIC End of custom TwoBodySymFuns

// AUTOMATIC Start of custom ThreeBodySymFuns

double BehlerG4::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rij*cutfun->eval(rij) + cutfun->derivative(rij))*exp2(-prms[1] + 1)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rik*cutfun->eval(rik) + cutfun->derivative(rik))*exp2(-prms[1] + 1)*cutfun->eval(rij)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::dcostheta(double rij, double rik, double costheta)
{
  return prms[0]*prms[1]*pow(costheta*prms[0] + 1, prms[1] - 1)*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

void BehlerG4::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = costheta*prms[0] + 1;
  auto x1 = pow(x0, prms[1]);
  auto x2 = cutfun->eval(rij);
  auto x3 = cutfun->eval(rik);
  auto x4 = exp2(-prms[1] + 1);
  auto x5 = exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
  auto x6 = x2*x3*x4*x5;
  auto x7 = 2*prms[2];
  auto x8 = x1*x4*x5;
  G = x1*x6;
  dGdrij = x3*x8*(-rij*x2*x7 + cutfun->derivative(rij));
  dGdrik = x2*x8*(-rik*x3*x7 + cutfun->derivative(rik));
  dGdcostheta = prms[0]*prms[1]*pow(x0, prms[1] - 1)*x6;
};

void BehlerG4::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = costheta*prms[0] + 1;
  auto x1 = pow(x0, prms[1]);
  auto x2 = 2*prms[2];
  auto x3 = cutfun->eval(rij);
  auto x4 = cutfun->eval(rik);
  auto x5 = exp2(-prms[1] + 1);
  auto x6 = exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
  auto x7 = x4*x5*x6;
  dGdrij = x1*x7*(-rij*x2*x3 + cutfun->derivative(rij));
  dGdrik = x1*x3*x5*x6*(-rik*x2*x4 + cutfun->derivative(rik));
  dGdcostheta = prms[0]*prms[1]*pow(x0, prms[1] - 1)*x3*x7;
};
// AUTOMATIC End of custom ThreeBodySymFuns

std::shared_ptr<CutoffFunction> switch_CutFun(
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun;
  switch (cutoff_type) {
    case 0:
      cutfun = std::make_shared<ConstCutoffFunction>(cutoff);
      break;
    case 1:
      cutfun = std::make_shared<CosCutoffFunction>(cutoff);
      break;
    case 2:
      cutfun = std::make_shared<TanhCutoffFunction>(cutoff);
      break;
    case 3:
      cutfun = std::make_shared<ShortRangeCutoffFunction>(cutoff);
      break;
    case 4:
      cutfun = std::make_shared<LongRangeCutoffFunction>(cutoff);
      break;
  }
  return cutfun;
}

std::shared_ptr<TwoBodySymmetryFunction> switch_TwoBodySymFun(int funtype,
  int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<TwoBodySymmetryFunction> symFun;
  switch (funtype){
// AUTOMATIC TwoBodySymmetryFunction switch start
    case 0:
      symFun = std::make_shared<BehlerG2>(num_prms, prms, cutfun);
      break;
    case 1:
      symFun = std::make_shared<OneOverR6>(num_prms, prms, cutfun);
      break;
    case 2:
      symFun = std::make_shared<OneOverR8>(num_prms, prms, cutfun);
      break;
    case 3:
      symFun = std::make_shared<OneOverR10>(num_prms, prms, cutfun);
      break;
// AUTOMATIC TwoBodySymmetryFunction switch end
    default:
      printf("No function type %d\n", funtype);
  }
  return symFun;
}

std::shared_ptr<ThreeBodySymmetryFunction> switch_ThreeBodySymFun(int funtype,
  int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<ThreeBodySymmetryFunction> symFun;
  switch (funtype){
// AUTOMATIC ThreeBodySymmetryFunction switch start
    case 0:
      symFun = std::make_shared<BehlerG4>(num_prms, prms, cutfun);
      break;
// AUTOMATIC ThreeBodySymmetryFunction switch end
    default:
      printf("No function type %d\n", funtype);
  }
  return symFun;
}

int get_CutFun_by_name(const char* name)
{
  int id = -1;
  if (strcmp(name, "const") == 0)
  {
    id = 0;
  } else if (strcmp(name, "cos") == 0)
  {
    id = 1;
  } else if (strcmp(name, "tanh") == 0)
  {
    id = 2;
  } else if (strcmp(name, "shortRange") == 0)
  {
    id = 3;
  } else if (strcmp(name, "longRange") == 0)
  {
    id = 4;
  }
  return id;
}

int get_TwoBodySymFun_by_name(const char* name)
{
  int id = -1;
// AUTOMATIC get_TwoBodySymFuns start
  if (strcmp(name, "BehlerG2") == 0)
  {
    id = 0;
  }
  if (strcmp(name, "OneOverR6") == 0)
  {
    id = 1;
  }
  if (strcmp(name, "OneOverR8") == 0)
  {
    id = 2;
  }
  if (strcmp(name, "OneOverR10") == 0)
  {
    id = 3;
  }
// AUTOMATIC get_TwoBodySymFuns end
  return id;
}

int get_ThreeBodySymFun_by_name(const char* name)
{
  int id = -1;
// AUTOMATIC get_ThreeBodySymFuns start
  if (strcmp(name, "BehlerG4") == 0)
  {
    id = 0;
  }
// AUTOMATIC get_ThreeBodySymFuns end
  return id;
}

void available_symFuns()
{
// AUTOMATIC available_symFuns start
  printf("TwoBodySymmetryFunctions: (key: name, # of parameters)\n");
  printf("0: BehlerG2, 2\n");
  printf("1: OneOverR6, 0\n");
  printf("2: OneOverR8, 0\n");
  printf("3: OneOverR10, 0\n");
  printf("ThreeBodySymmetryFunctions: (key: name, # of parameters)\n");
  printf("0: BehlerG4, 3\n");
// AUTOMATIC available_symFuns end
}
