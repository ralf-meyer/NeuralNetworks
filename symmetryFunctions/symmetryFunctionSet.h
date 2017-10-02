#include "symmetryFunctions.h"
#include <vector>

class SymmetryFunctionSet
{
  public:
    SymmetryFunctionSet(int num_atomtypes_i);
    ~SymmetryFunctionSet();
    void add_TwoBodySymmetryFunction(int type1, int type2, int funtype, int num_prms,
      double* prms, int cutoff_type, double cutoff);
    void add_ThreeBodySymmetryFunction(int type1, int type2, int type3,
      int funtype, int num_prms, double* prms, int cutoff_type, double cutoff);
    int get_G_vector_size(int num_atoms, int* types);
    void eval(int num_atoms, int* types, double* xyzs, double* G_vector);
    void eval_derivatives(int num_atoms, int* types, double* xyzs, double* dG_tensor);
  private:
    int num_atomtypes, num_atomtypes2;
    int* num_symFuns;
    std::vector<TwoBodySymmetryFunction*>* twoBodySymFuns;
    int* pos_twoBody;
    std::vector<ThreeBodySymmetryFunction*>* threeBodySymFuns;
    int* pos_threeBody;

    CutoffFunction* switch_CutFun(int cutoff_type, double cutoff);
    TwoBodySymmetryFunction* switch_TwoBodySymFun(int funtype, int num_prms,
      double* prms, CutoffFunction* cutfun);
    ThreeBodySymmetryFunction* switch_ThreeBodySymFun(int funtype, int num_prms,
      double* prms, CutoffFunction* cutfun);
};

// Wrap the C++ classes for C usage in python ctypes:
extern "C" {
  SymmetryFunctionSet* create_SymmetryFunctionSet(int num_atomtypes)
  {
    return new SymmetryFunctionSet(num_atomtypes);
  }
  void SymmetryFunctionSet_add_TwoBodySymmetryFunction(SymmetryFunctionSet* symFunSet,
  int type1, int type2, int funtype, int num_prms, double* prms, int cutoff_type, double cutoff)
  {
    symFunSet->add_TwoBodySymmetryFunction(type1, type2, funtype, num_prms, prms, cutoff_type,
      cutoff);
  }
  void SymmetryFunctionSet_add_ThreeBodySymmetryFunction(SymmetryFunctionSet* symFunSet,
  int type1, int type2, int type3, int funtype, int num_prms, double* prms, int cutoff_type, double cutoff)
  {
    symFunSet->add_ThreeBodySymmetryFunction(type1, type2, type3, funtype, num_prms, prms,
    cutoff_type, cutoff);
  }
  //void SymmetryFunctionSet_add_radial_function(SymmetryFunctionSet* symFunSet, double rs, double eta, double cutoff) {symFunSet->add_radial_function(rs, eta, cutoff);}
  //void SymmetryFunctionSet_add_angular_function(SymmetryFunctionSet* symFunSet, double eta, double zeta, double lambda, double cutoff) {symFunSet->add_angular_function(eta, zeta, lambda, cutoff);}
  int SymmetryFunctionSet_get_G_vector_size(SymmetryFunctionSet* symFunSet,
    int num_atoms, int* types)
  {
    symFunSet->get_G_vector_size(num_atoms, types);
  }
  void SymmetryFunctionSet_eval(SymmetryFunctionSet* symFunSet, int num_atoms,
    int* types, double* xyzs, double* out)
  {
    symFunSet->eval(num_atoms, types, xyzs, out);
  }
  void SymmetryFunctionSet_eval_derivatives(SymmetryFunctionSet* symFunSet,
    int num_atoms, int* types, double* xyzs, double* out)
  {
    symFunSet->eval_derivatives(num_atoms, types, xyzs, out);
  }
};
