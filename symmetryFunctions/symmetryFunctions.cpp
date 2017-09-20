#include "symmetryFunctions.h"
#include <math.h>

CutoffFunction::CutoffFunction(double cutoff_i)
{
    cutoff = cutoff_i;
};

/*double CutoffFunction::eval(double r)
{
    return 0.0;
};
*/

double CosCutoffFunction::eval(double r)
{
    if (r <= cutoff) return 0.5*(cos(M_PI*r/cutoff)+1.0);
    else return 0.0;
};

double CosCutoffFunction::derivative(double r)
{
    if (r <= cutoff) return 0.5*(cos(M_PI*r/cutoff)+1.0);
    else return 0.0;
};


/*SymmetryFunction::SymmetryFunction()
{

};


BehlerG2cut::BehlerG2cut(double rs_i, double eta_i, double cutoff, int cutoff_type)
{
    rs = rs_i;
    eta = eta_i;
    if (cutoff_type == 1)
        cutFun = new CosCutoffFunction(cutoff);
    else
        cutFun = new CosCutoffFunction(cutoff);
};

double BehlerG2cut::eval(double* r)
{
    double rij = r[0];
    return exp(-eta*pow(rij-rs,2))*cutFun->eval(rij);
};

double BehlerG2cut::derivative(double* r)
{
    double rij = r[0];
    return exp(-eta*pow(rij-rs,2))*cutFun->eval(rij);
};
*/

RadialSymmetryFunction::RadialSymmetryFunction(double rs_i, double eta_i, double cutoff, int cutoff_type)
{
    rs = rs_i;
    eta = eta_i;
    if (cutoff_type == 1)
        cutFun = new CosCutoffFunction(cutoff);
    else
        cutFun = new CosCutoffFunction(cutoff);
};

double RadialSymmetryFunction::eval(double r)
{
   return exp(-eta*pow(r-rs,2))*cutFun->eval(r);
};


AngularSymmetryFunction::AngularSymmetryFunction(double eta_i, double zeta_i, double lambda_i, double cutoff, int cutoff_type)
{
    eta = eta_i;
    zeta = zeta_i;
    lambda = lambda_i;
    if (cutoff_type == 1)
        cutFun = new CosCutoffFunction(cutoff);
    else
        cutFun = new CosCutoffFunction(cutoff);
};

double AngularSymmetryFunction::eval(double rij, double rik, double costheta)
{
   return pow(2.0, 1-zeta)*pow(1.0 + lambda*costheta, zeta)*exp(-eta*(pow(rij,2)+pow(rik,2)))*cutFun->eval(rij)*cutFun->eval(rik);
};

