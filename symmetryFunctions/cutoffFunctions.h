class CutoffFunction
{
    public:
      CutoffFunction(double cutoff_i);
      CutoffFunction();
      virtual ~CutoffFunction();
      virtual double eval(double r) = 0;
      virtual double derivative(double r) = 0;
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