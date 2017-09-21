/* TODO figure out why abstract classes do not seem to work with the ctypes implementation*/
class CutoffFunction
{
    public:
        CutoffFunction(double cutoff_i);
        CutoffFunction();
        double eval(double r);
        double derivative(double r);
    protected:
        double cutoff;
};

class ConstCutoffFunction: public CutoffFunction
{
    public:
        ConstCutoffFunction(double cutoff_i):CutoffFunction(cutoff_i){};
        double eval(double r);
        double derivative(double r);
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
