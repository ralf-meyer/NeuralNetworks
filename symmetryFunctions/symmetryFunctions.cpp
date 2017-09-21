#include "symmetryFunctions.h"
#include <math.h>

SymmetryFunction::SymmetryFunction(double* prms_i, CutoffFunction cutfun_i)
{
    prms = prms_i;
    cutfun = cutfun_i;
};
/*
BehlerG2::BehlerG2(double* prms_i, CutoffFunction cutfun_i)
{
    prms = prms_i;
    cutfun = cutfun_i;
};*/

double BehlerG2::eval(double r)
{
    return exp(-prms[0]*pow(r-prms[1],2))*cutfun.eval(r);
};
