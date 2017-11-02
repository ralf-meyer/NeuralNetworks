#include "symmetryFunctions.h"
#include <math.h>

CutoffFunction::CutoffFunction(double cutoff_i)
{
    cutoff = cutoff_i;
};

double CutoffFunction::eval(double r)
{
    return 0.0;
};

double CosCutoffFunction::eval(double r)
{
    if (r <= cutoff) return 0.5*(cos(M_PI*r/cutoff)+1.0);
    else return 0.0;
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
}


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
//Start of section for custom symmetry functions
double cutoffSymmetryFunction::cutoffSymmetryFunction(double cut_i,int cutoff_type)
	cut = cut_i;
	if (cutoff_type == 0)
		cutFun = new CosCutoffFunction(cutoff);
	else if(cutoff_type == 1)
		cutFun = new TanhCutoffFunction(cutoff);
	else
		cutFun = 1;
};
double cutoffSymmetryFunction::eval(double* r)
{
	return = 1.0e-16*cut + 1.0e-16*sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2)) + 1;
};
double cutoffSymmetryFunction::derivative_rij(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_rik(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_costheta(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_xi(double* r)
{
	double tmp0 = r[0] - r[1];
	return = 1.0e-16*tmp0/sqrt(pow(tmp0, 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
};
double cutoffSymmetryFunction::derivative_yi(double* r)
{
	double tmp0 = r[2] - r[3];
	return = 1.0e-16*tmp0/sqrt(pow(tmp0, 2) + pow(r[0] - r[1], 2) + pow(r[4] - r[5], 2));
};
double cutoffSymmetryFunction::derivative_zi(double* r)
{
	double tmp0 = r[4] - r[5];
	return = 1.0e-16*tmp0/sqrt(pow(tmp0, 2) + pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2));
};
double cutoffSymmetryFunction::derivative_xj(double* r)
{
	return = 1.0e-16*(-r[0] + r[1])/sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
};
double cutoffSymmetryFunction::derivative_yj(double* r)
{
	return = 1.0e-16*(-r[2] + r[3])/sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
};
double cutoffSymmetryFunction::derivative_zj(double* r)
{
	return = 1.0e-16*(-r[4] + r[5])/sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
};
double cutoffSymmetryFunction::derivative_xk(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_yk(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_zk(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::cutoffSymmetryFunction(double cut_i,int cutoff_type)
	cut = cut_i;
	if (cutoff_type == 0)
		cutFun = new CosCutoffFunction(cutoff);
	else if(cutoff_type == 1)
		cutFun = new TanhCutoffFunction(cutoff);
	else
		cutFun = 1;
};
double cutoffSymmetryFunction::eval(double* r)
{
	return = 1.0e-16*cut + 1.0e-16*r[0] + 1;
};
double cutoffSymmetryFunction::derivative_rij(double* r)
{
	return = 1e-16;
};
double cutoffSymmetryFunction::derivative_rik(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_costheta(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_xi(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_yi(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_zi(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_xj(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_yj(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_zj(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_xk(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_yk(double* r)
{
	return = 0.0;
};
double cutoffSymmetryFunction::derivative_zk(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::angularSymmetryFunction(double eta_i, double lamb_i, double zeta_i,int cutoff_type)
	eta = eta_i;
	lamb = lamb_i;
	zeta = zeta_i;
	if (cutoff_type == 0)
		cutFun = new CosCutoffFunction(cutoff);
	else if(cutoff_type == 1)
		cutFun = new TanhCutoffFunction(cutoff);
	else
		cutFun = 1;
};
double angularSymmetryFunction::eval(double* r)
{
	double tmp0 = pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp1 = -r[2];
	double tmp2 = -r[5];
	double tmp3 = -r[8];
	double tmp4 = pow(tmp1 + r[0], 2) + pow(tmp2 + r[3], 2) + pow(tmp3 + r[6], 2);
	double tmp5 = tmp0 + tmp4;
	return = pow(2, -zeta + 1)*pow((1.0L/2.0L)*lamb*(tmp5 - pow(tmp1 + r[1], 2) - pow(tmp2 + r[4], 2) - pow(tmp3 + r[7], 2))/(sqrt(tmp0)*sqrt(tmp4)) + 1, zeta)*exp(-eta*tmp5);
};
double angularSymmetryFunction::derivative_rij(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_rik(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_costheta(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_xi(double* r)
{
	double tmp0 = 4*r[0] - 2*r[1] - 2*r[2];
	double tmp1 = pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp2 = -r[2];
	double tmp3 = -r[5];
	double tmp4 = -r[8];
	double tmp5 = pow(tmp2 + r[0], 2) + pow(tmp3 + r[3], 2) + pow(tmp4 + r[6], 2);
	double tmp6 = tmp1 + tmp5;
	double tmp7 = pow(tmp1, -1.0L/2.0L);
	double tmp8 = pow(tmp5, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp7*tmp8;
	double tmp10 = tmp6 - pow(tmp2 + r[1], 2) - pow(tmp3 + r[4], 2) - pow(tmp4 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp6);
	double tmp13 = -r[0];
	double tmp14 = (1.0L/2.0L)*lamb*tmp10;
	return = -eta*tmp0*tmp12 + tmp12*zeta*(tmp0*tmp9 + tmp14*tmp7*(tmp13 + r[2])/pow(tmp5, 3.0L/2.0L) + tmp14*tmp8*(tmp13 + r[1])/pow(tmp1, 3.0L/2.0L))/tmp11;
};
double angularSymmetryFunction::derivative_yi(double* r)
{
	double tmp0 = 4*r[3] - 2*r[4] - 2*r[5];
	double tmp1 = pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp2 = -r[2];
	double tmp3 = -r[5];
	double tmp4 = -r[8];
	double tmp5 = pow(tmp2 + r[0], 2) + pow(tmp3 + r[3], 2) + pow(tmp4 + r[6], 2);
	double tmp6 = tmp1 + tmp5;
	double tmp7 = pow(tmp1, -1.0L/2.0L);
	double tmp8 = pow(tmp5, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp7*tmp8;
	double tmp10 = tmp6 - pow(tmp2 + r[1], 2) - pow(tmp3 + r[4], 2) - pow(tmp4 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp6);
	double tmp13 = -r[3];
	double tmp14 = (1.0L/2.0L)*lamb*tmp10;
	return = -eta*tmp0*tmp12 + tmp12*zeta*(tmp0*tmp9 + tmp14*tmp7*(tmp13 + r[5])/pow(tmp5, 3.0L/2.0L) + tmp14*tmp8*(tmp13 + r[4])/pow(tmp1, 3.0L/2.0L))/tmp11;
};
double angularSymmetryFunction::derivative_zi(double* r)
{
	double tmp0 = 4*r[6] - 2*r[7] - 2*r[8];
	double tmp1 = pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp2 = -r[2];
	double tmp3 = -r[5];
	double tmp4 = -r[8];
	double tmp5 = pow(tmp2 + r[0], 2) + pow(tmp3 + r[3], 2) + pow(tmp4 + r[6], 2);
	double tmp6 = tmp1 + tmp5;
	double tmp7 = pow(tmp1, -1.0L/2.0L);
	double tmp8 = pow(tmp5, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp7*tmp8;
	double tmp10 = tmp6 - pow(tmp2 + r[1], 2) - pow(tmp3 + r[4], 2) - pow(tmp4 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp6);
	double tmp13 = -r[6];
	double tmp14 = (1.0L/2.0L)*lamb*tmp10;
	return = -eta*tmp0*tmp12 + tmp12*zeta*(tmp0*tmp9 + tmp14*tmp7*(tmp13 + r[8])/pow(tmp5, 3.0L/2.0L) + tmp14*tmp8*(tmp13 + r[7])/pow(tmp1, 3.0L/2.0L))/tmp11;
};
double angularSymmetryFunction::derivative_xj(double* r)
{
	double tmp0 = -2*r[0];
	double tmp1 = r[0] - r[1];
	double tmp2 = pow(tmp1, 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp3 = -r[2];
	double tmp4 = -r[5];
	double tmp5 = -r[8];
	double tmp6 = pow(tmp3 + r[0], 2) + pow(tmp4 + r[3], 2) + pow(tmp5 + r[6], 2);
	double tmp7 = tmp2 + tmp6;
	double tmp8 = pow(tmp6, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp8/sqrt(tmp2);
	double tmp10 = tmp7 - pow(tmp3 + r[1], 2) - pow(tmp4 + r[4], 2) - pow(tmp5 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp7);
	return = -eta*tmp12*(tmp0 + 2*r[1]) + tmp12*zeta*((1.0L/2.0L)*lamb*tmp1*tmp10*tmp8/pow(tmp2, 3.0L/2.0L) + tmp9*(tmp0 + 2*r[2]))/tmp11;
};
double angularSymmetryFunction::derivative_yj(double* r)
{
	double tmp0 = -2*r[3];
	double tmp1 = r[3] - r[4];
	double tmp2 = pow(tmp1, 2) + pow(r[0] - r[1], 2) + pow(r[6] - r[7], 2);
	double tmp3 = -r[2];
	double tmp4 = -r[5];
	double tmp5 = -r[8];
	double tmp6 = pow(tmp3 + r[0], 2) + pow(tmp4 + r[3], 2) + pow(tmp5 + r[6], 2);
	double tmp7 = tmp2 + tmp6;
	double tmp8 = pow(tmp6, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp8/sqrt(tmp2);
	double tmp10 = tmp7 - pow(tmp3 + r[1], 2) - pow(tmp4 + r[4], 2) - pow(tmp5 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp7);
	return = -eta*tmp12*(tmp0 + 2*r[4]) + tmp12*zeta*((1.0L/2.0L)*lamb*tmp1*tmp10*tmp8/pow(tmp2, 3.0L/2.0L) + tmp9*(tmp0 + 2*r[5]))/tmp11;
};
double angularSymmetryFunction::derivative_zj(double* r)
{
	double tmp0 = -2*r[6];
	double tmp1 = r[6] - r[7];
	double tmp2 = pow(tmp1, 2) + pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2);
	double tmp3 = -r[2];
	double tmp4 = -r[5];
	double tmp5 = -r[8];
	double tmp6 = pow(tmp3 + r[0], 2) + pow(tmp4 + r[3], 2) + pow(tmp5 + r[6], 2);
	double tmp7 = tmp2 + tmp6;
	double tmp8 = pow(tmp6, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp8/sqrt(tmp2);
	double tmp10 = tmp7 - pow(tmp3 + r[1], 2) - pow(tmp4 + r[4], 2) - pow(tmp5 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp7);
	return = -eta*tmp12*(tmp0 + 2*r[7]) + tmp12*zeta*((1.0L/2.0L)*lamb*tmp1*tmp10*tmp8/pow(tmp2, 3.0L/2.0L) + tmp9*(tmp0 + 2*r[8]))/tmp11;
};
double angularSymmetryFunction::derivative_xk(double* r)
{
	double tmp0 = -2*r[0];
	double tmp1 = pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp2 = -r[2];
	double tmp3 = tmp2 + r[0];
	double tmp4 = -r[5];
	double tmp5 = -r[8];
	double tmp6 = pow(tmp3, 2) + pow(tmp4 + r[3], 2) + pow(tmp5 + r[6], 2);
	double tmp7 = tmp1 + tmp6;
	double tmp8 = pow(tmp1, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp8/sqrt(tmp6);
	double tmp10 = tmp7 - pow(tmp2 + r[1], 2) - pow(tmp4 + r[4], 2) - pow(tmp5 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp7);
	return = -eta*tmp12*(tmp0 + 2*r[2]) + tmp12*zeta*((1.0L/2.0L)*lamb*tmp10*tmp3*tmp8/pow(tmp6, 3.0L/2.0L) + tmp9*(tmp0 + 2*r[1]))/tmp11;
};
double angularSymmetryFunction::derivative_yk(double* r)
{
	double tmp0 = -2*r[3];
	double tmp1 = pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp2 = -r[2];
	double tmp3 = -r[5];
	double tmp4 = tmp3 + r[3];
	double tmp5 = -r[8];
	double tmp6 = pow(tmp4, 2) + pow(tmp2 + r[0], 2) + pow(tmp5 + r[6], 2);
	double tmp7 = tmp1 + tmp6;
	double tmp8 = pow(tmp1, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp8/sqrt(tmp6);
	double tmp10 = tmp7 - pow(tmp2 + r[1], 2) - pow(tmp3 + r[4], 2) - pow(tmp5 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp7);
	return = -eta*tmp12*(tmp0 + 2*r[5]) + tmp12*zeta*((1.0L/2.0L)*lamb*tmp10*tmp4*tmp8/pow(tmp6, 3.0L/2.0L) + tmp9*(tmp0 + 2*r[4]))/tmp11;
};
double angularSymmetryFunction::derivative_zk(double* r)
{
	double tmp0 = -2*r[6];
	double tmp1 = pow(r[0] - r[1], 2) + pow(r[3] - r[4], 2) + pow(r[6] - r[7], 2);
	double tmp2 = -r[2];
	double tmp3 = -r[5];
	double tmp4 = -r[8];
	double tmp5 = tmp4 + r[6];
	double tmp6 = pow(tmp5, 2) + pow(tmp2 + r[0], 2) + pow(tmp3 + r[3], 2);
	double tmp7 = tmp1 + tmp6;
	double tmp8 = pow(tmp1, -1.0L/2.0L);
	double tmp9 = (1.0L/2.0L)*lamb*tmp8/sqrt(tmp6);
	double tmp10 = tmp7 - pow(tmp2 + r[1], 2) - pow(tmp3 + r[4], 2) - pow(tmp4 + r[7], 2);
	double tmp11 = tmp10*tmp9 + 1;
	double tmp12 = pow(2, -zeta + 1)*pow(tmp11, zeta)*exp(-eta*tmp7);
	return = -eta*tmp12*(tmp0 + 2*r[8]) + tmp12*zeta*((1.0L/2.0L)*lamb*tmp10*tmp5*tmp8/pow(tmp6, 3.0L/2.0L) + tmp9*(tmp0 + 2*r[7]))/tmp11;
};
double angularSymmetryFunction::angularSymmetryFunction(double eta_i, double lamb_i, double zeta_i,int cutoff_type)
	eta = eta_i;
	lamb = lamb_i;
	zeta = zeta_i;
	if (cutoff_type == 0)
		cutFun = new CosCutoffFunction(cutoff);
	else if(cutoff_type == 1)
		cutFun = new TanhCutoffFunction(cutoff);
	else
		cutFun = 1;
};
double angularSymmetryFunction::eval(double* r)
{
	return = pow(2, -zeta + 1)*pow(r[0]*lamb + 1, zeta)*exp(-eta*(pow(r[1], 2) + pow(r[2], 2)));
};
double angularSymmetryFunction::derivative_rij(double* r)
{
	return = -2*pow(2, -zeta + 1)*eta*r[1]*pow(r[0]*lamb + 1, zeta)*exp(-eta*(pow(r[1], 2) + pow(r[2], 2)));
};
double angularSymmetryFunction::derivative_rik(double* r)
{
	return = -2*pow(2, -zeta + 1)*eta*r[2]*pow(r[0]*lamb + 1, zeta)*exp(-eta*(pow(r[1], 2) + pow(r[2], 2)));
};
double angularSymmetryFunction::derivative_costheta(double* r)
{
	double tmp0 = r[0]*lamb + 1;
	return = pow(2, -zeta + 1)*lamb*pow(tmp0, zeta)*zeta*exp(-eta*(pow(r[1], 2) + pow(r[2], 2)))/tmp0;
};
double angularSymmetryFunction::derivative_xi(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_yi(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_zi(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_xj(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_yj(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_zj(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_xk(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_yk(double* r)
{
	return = 0.0;
};
double angularSymmetryFunction::derivative_zk(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::radialSymmetryFunction(double eta_i, double rs_i,int cutoff_type)
	eta = eta_i;
	rs = rs_i;
	if (cutoff_type == 0)
		cutFun = new CosCutoffFunction(cutoff);
	else if(cutoff_type == 1)
		cutFun = new TanhCutoffFunction(cutoff);
	else
		cutFun = 1;
};
double radialSymmetryFunction::eval(double* r)
{
	double tmp0 = -rs + sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
	return = exp(-eta*pow(tmp0, 2))*cos(0.5*sqrt(3)*eta*tmp0);
};
double radialSymmetryFunction::derivative_rij(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_rik(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_costheta(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_xi(double* r)
{
	double tmp0 = sqrt(3);
	double tmp1 = r[0] - r[1];
	double tmp2 = sqrt(pow(tmp1, 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
	double tmp3 = -rs + tmp2;
	double tmp4 = 0.5*eta*tmp0*tmp3;
	double tmp5 = tmp1*exp(-eta*pow(tmp3, 2))/tmp2;
	return = -0.5*eta*tmp0*tmp5*sin(tmp4) - 2*eta*tmp3*tmp5*cos(tmp4);
};
double radialSymmetryFunction::derivative_yi(double* r)
{
	double tmp0 = sqrt(3);
	double tmp1 = r[2] - r[3];
	double tmp2 = sqrt(pow(tmp1, 2) + pow(r[0] - r[1], 2) + pow(r[4] - r[5], 2));
	double tmp3 = -rs + tmp2;
	double tmp4 = 0.5*eta*tmp0*tmp3;
	double tmp5 = tmp1*exp(-eta*pow(tmp3, 2))/tmp2;
	return = -0.5*eta*tmp0*tmp5*sin(tmp4) - 2*eta*tmp3*tmp5*cos(tmp4);
};
double radialSymmetryFunction::derivative_zi(double* r)
{
	double tmp0 = sqrt(3);
	double tmp1 = r[4] - r[5];
	double tmp2 = sqrt(pow(tmp1, 2) + pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2));
	double tmp3 = -rs + tmp2;
	double tmp4 = 0.5*eta*tmp0*tmp3;
	double tmp5 = tmp1*exp(-eta*pow(tmp3, 2))/tmp2;
	return = -0.5*eta*tmp0*tmp5*sin(tmp4) - 2*eta*tmp3*tmp5*cos(tmp4);
};
double radialSymmetryFunction::derivative_xj(double* r)
{
	double tmp0 = sqrt(3);
	double tmp1 = sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
	double tmp2 = -rs + tmp1;
	double tmp3 = 0.5*eta*tmp0*tmp2;
	double tmp4 = (-r[0] + r[1])*exp(-eta*pow(tmp2, 2))/tmp1;
	return = -0.5*eta*tmp0*tmp4*sin(tmp3) - 2*eta*tmp2*tmp4*cos(tmp3);
};
double radialSymmetryFunction::derivative_yj(double* r)
{
	double tmp0 = sqrt(3);
	double tmp1 = sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
	double tmp2 = -rs + tmp1;
	double tmp3 = 0.5*eta*tmp0*tmp2;
	double tmp4 = (-r[2] + r[3])*exp(-eta*pow(tmp2, 2))/tmp1;
	return = -0.5*eta*tmp0*tmp4*sin(tmp3) - 2*eta*tmp2*tmp4*cos(tmp3);
};
double radialSymmetryFunction::derivative_zj(double* r)
{
	double tmp0 = sqrt(3);
	double tmp1 = sqrt(pow(r[0] - r[1], 2) + pow(r[2] - r[3], 2) + pow(r[4] - r[5], 2));
	double tmp2 = -rs + tmp1;
	double tmp3 = 0.5*eta*tmp0*tmp2;
	double tmp4 = (-r[4] + r[5])*exp(-eta*pow(tmp2, 2))/tmp1;
	return = -0.5*eta*tmp0*tmp4*sin(tmp3) - 2*eta*tmp2*tmp4*cos(tmp3);
};
double radialSymmetryFunction::derivative_xk(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_yk(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_zk(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::radialSymmetryFunction(double eta_i, double rs_i,int cutoff_type)
	eta = eta_i;
	rs = rs_i;
	if (cutoff_type == 0)
		cutFun = new CosCutoffFunction(cutoff);
	else if(cutoff_type == 1)
		cutFun = new TanhCutoffFunction(cutoff);
	else
		cutFun = 1;
};
double radialSymmetryFunction::eval(double* r)
{
	double tmp0 = r[0] - rs;
	return = exp(-eta*pow(tmp0, 2))*cos(0.5*sqrt(3)*eta*tmp0);
};
double radialSymmetryFunction::derivative_rij(double* r)
{
	double tmp0 = 0.5*sqrt(3)*eta;
	double tmp1 = r[0] - rs;
	double tmp2 = exp(-eta*pow(tmp1, 2));
	double tmp3 = tmp0*tmp1;
	return = -eta*tmp2*(2*r[0] - 2*rs)*cos(tmp3) - tmp0*tmp2*sin(tmp3);
};
double radialSymmetryFunction::derivative_rik(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_costheta(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_xi(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_yi(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_zi(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_xj(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_yj(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_zj(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_xk(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_yk(double* r)
{
	return = 0.0;
};
double radialSymmetryFunction::derivative_zk(double* r)
{
	return = 0.0;
};
//End of section for custom symmetry functions
