#include <stdio.h>
#include "symmetryFunctionSet.h"

/* Compile using "g++ -g -std=c++11 test_for_leaks.cpp -I/path/to/NeuralNetworks/symmetryFunctions
-L/path/to/NeuralNetworks/symmetryFunctions -lSymFunSet -o test_for_leaks"*/

int main()
{
  printf("Programm started\n");
  SymmetryFunctionSet* sfs = new SymmetryFunctionSet(2);
  double* prms = new double[2];
  prms[0] = 0.0;
  prms[1] = 1.0;
  double* prms3 = new double[3];
  prms3[0] = 1.0;
  prms3[1] = 1.0;
  prms3[2] = 1.0;
  sfs->add_TwoBodySymmetryFunction(0, 0, 0, 2, prms, 0, 7.0);
  sfs->add_ThreeBodySymmetryFunction(0, 0, 0, 0, 3, prms3, 0, 7.0);
  printf("SymFunSet created\n");

  delete[] prms;
  delete[] prms3;
  delete sfs;
  return 0;
}