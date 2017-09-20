#include <stdio.h>
#include"symmetryFunctions.h"
#include <vector>

class SymmetryFunctionSet{
    public:
        SymmetryFunctionSet() {
    		printf("Constructor called\n");
                
	}
        void foo(){
            printf("Hello World\n");
        }        
        double add(double a, double b){
            double sum = a+b;
            printf("Got a = %f, b = %f  sum equal a+b=%f\n",a, b, sum);
            return a + b;
        }
        void add_radial_function(double rs, double eta, double cutoff){
            radSymFuns.push_back(RadialSymmetryFunction(rs, eta, cutoff, 1));
        };
        /*void add_BehlerG2cut(double rs, double eta, double cutoff){
            symFuns.push_back(BehlerG2cut(rs, eta, cutoff, 1));
	};*/
        void add_angular_function(double eta, double zeta, double lambda, double cutoff){
            angSymFuns.push_back(AngularSymmetryFunction(eta, zeta, lambda, cutoff, 1));
        };
        void eval_geometry(double& out){
        };
    private:
        std::vector <RadialSymmetryFunction> radSymFuns;  
        std::vector <AngularSymmetryFunction> angSymFuns; 
        //std::vector <SymmetryFunction> symFuns;
};

// Wrap the C++ classes for C usage in python ctypes:
extern "C" {
    SymmetryFunctionSet* SymmetryFunctionSet_new(){ return new SymmetryFunctionSet(); }
    void SymmetryFunctionSet_foo(SymmetryFunctionSet* symFunSet){ symFunSet->foo(); }
    double SymmetryFunctionSet_add(SymmetryFunctionSet* symFunSet, double a, double b){ symFunSet->add(a,b); }
    void SymmetryFunctionSet_add_radial_function(SymmetryFunctionSet* symFunSet, double rs, double eta, double cutoff) {symFunSet->add_radial_function(rs, eta, cutoff);}
    void SymmetryFunctionSet_add_angular_function(SymmetryFunctionSet* symFunSet, double eta, double zeta, double lambda, double cutoff) {symFunSet->add_angular_function(eta, zeta, lambda, cutoff);}
    void SymmetryFunctionSet_eval_geometry(SymmetryFunctionSet* symFunSet, double& out){symFunSet->eval_geometry(out);}
}
