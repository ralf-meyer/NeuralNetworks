#include "cutoffFunctions.h"
#include <math.h>

CutoffFunction::CutoffFunction(double cutoff_i)
{
    cutoff = cutoff_i;
};

CutoffFunction::CutoffFunction(){};

double CutoffFunction::eval(double r)
{
  return 0.0;
};

double ConstCutoffFunction::eval(double r)
{
    return 1.0;
};

double ConstCutoffFunction::derivative(double r)
{
    return 0.0;
};

double CosCutoffFunction::eval(double r)
{
    if (r <= cutoff) return 0.5*(cos(M_PI*r/cutoff)+1.0);
    else return 0.0;
};

double CosCutoffFunction::derivative(double r)
{
    if (r <= cutoff) return -0.5*(M_PI*sin(M_PI*r/cutoff))/cutoff;
    else return 0.0;
};

double TanhCutoffFunction::eval(double r)
{
    if (r <= cutoff) return pow(tanh(1.0-r/cutoff),3);
    else return 0.0;
};

double TanhCutoffFunction::derivative(double r)
{
    if (r <= cutoff) return -(3.0*pow(sinh(1.0-r/cutoff),2))/
        (cutoff*pow(cosh(1.0-r/cutoff),4));
    else return 0.0;
};
