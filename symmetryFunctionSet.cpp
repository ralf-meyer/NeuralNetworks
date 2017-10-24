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
        void add_angular_function(double eta, double zeta, double lambda, double cutoff){
            angSymFuns.push_back(AngularSymmetryFunction(eta, zeta, lambda, cutoff, 1));
        };
//Start of section for adding custom symmetry functions
	void add_cutoff(double cut,int cutoff_type) {
		cutoffSymFuns.push_back(cutoffSymmetryFunction(cut,cutoff_type));
	};
	void add_angular(double eta, double lamb, double zeta,int cutoff_type) {
		angularSymFuns.push_back(angularSymmetryFunction(eta,lamb,zeta,cutoff_type));
	};
	void add_radial(double eta, double rs,int cutoff_type) {
		radialSymFuns.push_back(radialSymmetryFunction(eta,rs,cutoff_type));
	};
//End of section for adding custom symmetry functions
        void eval_geometry(double& out){
        };
    private:
        std::vector <RadialSymmetryFunction> radSymFuns;  
        std::vector <AngularSymmetryFunction> angSymFuns;  
//Start of section custom symmetry functions vector
//End of section custom symmetry functions vector     
};

// Wrap the C++ classes for C usage in python ctypes:
extern "C" {
    SymmetryFunctionSet* SymmetryFunctionSet_new(){ return new SymmetryFunctionSet(); }
//Start of custom extern declaration
	void SymmetryFunctionSet_add_cutoff(SymmetryFunctionSet* symFunSet, double cut,int cutoff_type){symFunSet->add_cutoff(cut,cutoff_type);}
	void SymmetryFunctionSet_add_angular(SymmetryFunctionSet* symFunSet, double eta, double lamb, double zeta,int cutoff_type){symFunSet->add_angular(eta,lamb,zeta,cutoff_type);}
	void SymmetryFunctionSet_add_radial(SymmetryFunctionSet* symFunSet, double eta, double rs,int cutoff_type){symFunSet->add_radial(eta,rs,cutoff_type);}
//End of custom extern declaration

    void SymmetryFunctionSet_foo(SymmetryFunctionSet* symFunSet){ symFunSet->foo(); }
    double SymmetryFunctionSet_add(SymmetryFunctionSet* symFunSet, double a, double b){ symFunSet->add(a,b); }
    void SymmetryFunctionSet_add_radial_function(SymmetryFunctionSet* symFunSet, double rs, double eta, double cutoff) {symFunSet->add_radial_function(rs, eta, cutoff);}
    void SymmetryFunctionSet_add_angular_function(SymmetryFunctionSet* symFunSet, double eta, double zeta, double lambda, double cutoff) {symFunSet->add_angular_function(eta, zeta, lambda, cutoff);}
}
