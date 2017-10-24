class CutoffFunction
{
    public:
        CutoffFunction(double cutoff_i);
        CutoffFunction();
        virtual double eval(double r);
        virtual double derivative(double r);
    protected:
        double cutoff;
};

class CosCutoffFunction: public CutoffFunction
{
    public:
        CosCutoffFunction(double cutoff_i):CutoffFunction(cutoff_i){};
        double eval(double r);
        double derivative(double r);
};

class TanhCutoffFunction: public CutoffFunction
{
    public:
        TanhCutoffFunction(double cutoff_i):CutoffFunction(cutoff_i){};
        double eval(double r);
        double derivative(double r);
};

// Abstract base class for all symmetry functions
class SymmetryFunction
{
    public:
        virtual double eval(double* args);
        virtual double derivative(double* args);
};

class BehlerG2cut: public SymmetryFunction
{
    double rs;
    double eta;
    CutoffFunction *cutFun;
    public:
        BehlerG2cut(double rs_i, double eta_i, double cutoff, int cutoff_type);
        double eval(double* r);
        double derivative(double* r);        
};

class RadialSymmetryFunction
{
    double rs;
    double eta;
    CutoffFunction *cutFun;
    public:
        RadialSymmetryFunction(double rs_i, double eta_i, double cutoff, int cutoff_type);
        double eval(double r);
        double derivative(double r);        
};

class AngularSymmetryFunction
{
    double eta;
    double zeta;
    double lambda;
    CutoffFunction *cutFun;
    public:
        AngularSymmetryFunction(double eta_i, double zeta_i, double lambda_i, double cutoff, int cutoff_type);
        double eval(double rij, double rik, double costheta);
        double derivative(double rij, double rik, double costheta); 
};
//Start of section for custom symmetry functions
class cutoffSymmetryFunction:: public Symmetry Function
{
	double cut;
	CutoffFunction *cutFun
	public:
		cutoffSymmetryFunction(double cut);
		double eval(double* r);
		double derivative_rij(double* r);
		double derivative_rik(double* r);
		double derivative_costheta(double* r);
		double derivative_xi(double* r);
		double derivative_yi(double* r);
		double derivative_zi(double* r);
		double derivative_xj(double* r);
		double derivative_yj(double* r);
		double derivative_zj(double* r);
		double derivative_xk(double* r);
		double derivative_yk(double* r);
		double derivative_zk(double* r);
};
class cutoffSymmetryFunction:: public Symmetry Function
{
	double cut;
	CutoffFunction *cutFun
	public:
		cutoffSymmetryFunction(double cut);
		double eval(double* r);
		double derivative_rij(double* r);
		double derivative_rik(double* r);
		double derivative_costheta(double* r);
		double derivative_xi(double* r);
		double derivative_yi(double* r);
		double derivative_zi(double* r);
		double derivative_xj(double* r);
		double derivative_yj(double* r);
		double derivative_zj(double* r);
		double derivative_xk(double* r);
		double derivative_yk(double* r);
		double derivative_zk(double* r);
};
class angularSymmetryFunction:: public Symmetry Function
{
	double eta;
	double lamb;
	double zeta;
	CutoffFunction *cutFun
	public:
		angularSymmetryFunction(double eta, double lamb, double zeta);
		double eval(double* r);
		double derivative_rij(double* r);
		double derivative_rik(double* r);
		double derivative_costheta(double* r);
		double derivative_xi(double* r);
		double derivative_yi(double* r);
		double derivative_zi(double* r);
		double derivative_xj(double* r);
		double derivative_yj(double* r);
		double derivative_zj(double* r);
		double derivative_xk(double* r);
		double derivative_yk(double* r);
		double derivative_zk(double* r);
};
class angularSymmetryFunction:: public Symmetry Function
{
	double eta;
	double lamb;
	double zeta;
	CutoffFunction *cutFun
	public:
		angularSymmetryFunction(double eta, double lamb, double zeta);
		double eval(double* r);
		double derivative_rij(double* r);
		double derivative_rik(double* r);
		double derivative_costheta(double* r);
		double derivative_xi(double* r);
		double derivative_yi(double* r);
		double derivative_zi(double* r);
		double derivative_xj(double* r);
		double derivative_yj(double* r);
		double derivative_zj(double* r);
		double derivative_xk(double* r);
		double derivative_yk(double* r);
		double derivative_zk(double* r);
};
class radialSymmetryFunction:: public Symmetry Function
{
	double eta;
	double rs;
	CutoffFunction *cutFun
	public:
		radialSymmetryFunction(double eta, double rs);
		double eval(double* r);
		double derivative_rij(double* r);
		double derivative_rik(double* r);
		double derivative_costheta(double* r);
		double derivative_xi(double* r);
		double derivative_yi(double* r);
		double derivative_zi(double* r);
		double derivative_xj(double* r);
		double derivative_yj(double* r);
		double derivative_zj(double* r);
		double derivative_xk(double* r);
		double derivative_yk(double* r);
		double derivative_zk(double* r);
};
class radialSymmetryFunction:: public Symmetry Function
{
	double eta;
	double rs;
	CutoffFunction *cutFun
	public:
		radialSymmetryFunction(double eta, double rs);
		double eval(double* r);
		double derivative_rij(double* r);
		double derivative_rik(double* r);
		double derivative_costheta(double* r);
		double derivative_xi(double* r);
		double derivative_yi(double* r);
		double derivative_zi(double* r);
		double derivative_xj(double* r);
		double derivative_yj(double* r);
		double derivative_zj(double* r);
		double derivative_xk(double* r);
		double derivative_yk(double* r);
		double derivative_zk(double* r);
};
//End of section for custom symmetry functions

