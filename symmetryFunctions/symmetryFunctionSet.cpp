#include "symmetryFunctionSet.h"
#include <stdio.h>
#include <limits>
#include <math.h>

SymmetryFunctionSet::SymmetryFunctionSet(int num_atomtypes_i)
{
  num_atomtypes = num_atomtypes_i;
  num_atomtypes2 = pow(num_atomtypes, 2);
  num_symFuns = new int[2*num_atomtypes]();
  pos_twoBody = new int[num_atomtypes2]();
  pos_threeBody = new int[num_atomtypes2*num_atomtypes]();
  twoBodySymFuns = new std::vector<TwoBodySymmetryFunction*>[num_atomtypes2]();
  threeBodySymFuns = new std::vector<ThreeBodySymmetryFunction*>[num_atomtypes2*num_atomtypes]();
  printf("Constructor called with %d atom types\n",num_atomtypes);
}

SymmetryFunctionSet::~SymmetryFunctionSet()
{
  delete pos_twoBody;
  delete pos_threeBody;
  delete twoBodySymFuns;
  delete threeBodySymFuns;
  delete num_symFuns;
}

void SymmetryFunctionSet::add_TwoBodySymmetryFunction(int type1, int type2, int funtype, int num_prms,
double* prms, int cutoff_type, double cutoff)
{
  CutoffFunction* cutfun = switch_CutFun(cutoff_type, cutoff);
  TwoBodySymmetryFunction* symfun = switch_TwoBodySymFun(funtype, num_prms, prms, cutfun);

  twoBodySymFuns[type1*num_atomtypes+type2].push_back(symfun);
  num_symFuns[2*type1]++;
  for (int i = type2+1; i < num_atomtypes; i++){
    pos_twoBody[num_atomtypes*type1 + i]++;
  }
}

void SymmetryFunctionSet::add_ThreeBodySymmetryFunction(int type1, int type2, int type3,
int funtype, int num_prms, double* prms, int cutoff_type, double cutoff)
{
  CutoffFunction* cutfun = switch_CutFun(cutoff_type, cutoff);
  ThreeBodySymmetryFunction* symfun = switch_ThreeBodySymFun(funtype, num_prms, prms, cutfun);
  threeBodySymFuns[num_atomtypes2*type1 + num_atomtypes*type2 + type3].push_back(symfun);
  num_symFuns[2*type1+1]++;
  for (int j = type3+1; j < num_atomtypes; j++)
  {
    pos_threeBody[num_atomtypes2*type1 + num_atomtypes*type2 + j]++;
  }
  for (int i = type2+1; i < num_atomtypes; i++)
  {
    for (int j = 0; j < num_atomtypes; j++)
    {
      pos_threeBody[num_atomtypes2*type1 + num_atomtypes*i + j]++;
    }
  }
}

int SymmetryFunctionSet::get_G_vector_size(int num_atoms, int* types)
{
  int G_vector_size = 0;
  for (int i = 0; i < num_atoms; i++)
  {
    G_vector_size += num_symFuns[2*types[i]] + num_symFuns[2*types[i]+1];
  }
  return G_vector_size;
}

/* Evaluates the symmetryFunctionSet for the given atomic geometry.
   Could probably be rewritten a bit faster if the loops would not span
   the full N^3 but only the upper (or lower) triangle of the rij matrix
   and rijk tensor.*/
void SymmetryFunctionSet::eval(int num_atoms, int* types, double* xyzs, double* G_vector)
{
  double rij, rik, theta;
  int counter = 0;
  int i, j, k, twoBody_i, three_Body_i;

  for (i = 0; i < num_atoms; i++)
  {
    for (j = 0; j < num_atoms; j++)
    {
      if (i == j)
      {
        continue;
      } else
      {
        rij = sqrt(pow(xyzs[3*i]-xyzs[3*j], 2) +
                  pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                  pow(xyzs[3*i+2]-xyzs[3*j+2], 2));
        for (twoBody_i = 0; twoBody_i < twoBodySymFuns[types[i]*num_atomtypes+types[j]].size(); twoBody_i++)
        {
          G_vector[counter + pos_twoBody[num_atomtypes*types[i]+types[j]] + twoBody_i] +=
            twoBodySymFuns[types[i]*num_atomtypes+types[j]][twoBody_i]->eval(rij);
        }
        for (k = 0; k < num_atoms; k++)
        {
          if (i == k || j == k)
          {
            continue;
          } else
          {
            rik = sqrt(pow(xyzs[3*i]-xyzs[3*k], 2) +
                      pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                      pow(xyzs[3*i+2]-xyzs[3*k+2], 2));
            // Calculate the angle between rij and rik
            theta = acos(((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
              (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
              (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]))/(rij*rik));
            for (three_Body_i = 0; three_Body_i < threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]].size(); three_Body_i++)
            {
              G_vector[counter + num_symFuns[2*types[i]] + pos_threeBody[num_atomtypes2*types[i] +
                num_atomtypes*types[j] + types[k]]+ three_Body_i] +=
                threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]][three_Body_i]->eval(rij, rik, theta);
            }
          }
        }
      }
    }
    counter += num_symFuns[2*types[i]] + num_symFuns[2*types[i]+1];
  }
}

void SymmetryFunctionSet::eval_derivatives(int num_atoms, int* types, double* xyzs, double* dG_tensor)
{
  double rij, rij2, rik, rik2, theta, dGdr, dGdrij, dGdrik, dGdtheta, dot;
  int counter = 0;
  int i, j, k, twoBody_i, three_Body_i, coord, index_base;

  for (i = 0; i < num_atoms; i++)
  {
    for (j = 0; j < num_atoms; j++)
    {
      if (i == j)
      {
        continue;
      } else
      {
        rij2 = pow(xyzs[3*i]-xyzs[3*j], 2) + pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                  pow(xyzs[3*i+2]-xyzs[3*j+2], 2);
        rij = sqrt(rij2);
        for (twoBody_i = 0; twoBody_i < twoBodySymFuns[types[i]*num_atomtypes+types[j]].size(); twoBody_i++)
        {
          dGdr = twoBodySymFuns[types[i]*num_atomtypes+types[j]][twoBody_i]->drij(rij);
          for (coord = 0; coord < 3; coord++){
            // dG/dx is calculated as product of dG/dr * dr/dx
            dG_tensor[3*num_atoms*(counter + pos_twoBody[num_atomtypes*types[i]+types[j]] + twoBody_i) +
              3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
            dG_tensor[3*num_atoms*(counter + pos_twoBody[num_atomtypes*types[i]+types[j]] + twoBody_i) +
              3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
          }
        }
        for (k = 0; k < num_atoms; k++)
        {
          if (i == k || j == k)
          {
            continue;
          } else
          {
            rik2 = pow(xyzs[3*i]-xyzs[3*k], 2) + pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                      pow(xyzs[3*i+2]-xyzs[3*k+2], 2);
            rik = sqrt(rik2);

            dot = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
              (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
              (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]));
            theta = acos(dot/(rij*rik));
            for (three_Body_i = 0; three_Body_i < threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]].size(); three_Body_i++)
            {
              dGdrij = threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]][three_Body_i]->drij(rij, rik, theta);
              dGdrik = threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]][three_Body_i]->drik(rij, rik, theta);
              dGdtheta = threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]][three_Body_i]->dtheta(rij, rik, theta);

              index_base = 3*num_atoms*(counter + num_symFuns[2*types[i]] +
                pos_threeBody[num_atomtypes2*types[i] + num_atomtypes*types[j] + types[k]]+ three_Body_i);
              for (coord = 0; coord < 3; coord++){
                // Derivative with respect to rij
                dG_tensor[index_base + 3*i + coord] += dGdrij*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
                dG_tensor[index_base + 3*j + coord] += dGdrij*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
                // Derivative with respect to rik
                dG_tensor[index_base + 3*i + coord] += dGdrik*(xyzs[3*i+coord]-xyzs[3*k+coord])/rik;
                dG_tensor[index_base + 3*k + coord] += dGdrik*(-xyzs[3*i+coord]+xyzs[3*k+coord])/rik;

                // Derivative with respect to theta
                dG_tensor[index_base + 3*i + coord] += dGdtheta*(dot*rik2*(xyzs[3*i + coord]-xyzs[3*j+coord]) +
                  dot*rij2*(xyzs[3*i+coord]-xyzs[3*k+coord]) + rij2*rik2*(xyzs[3*j + coord]+xyzs[3*k + coord]-2*xyzs[3*i + coord]))/
                  (rij*rij2*sqrt(1-pow(dot,2)/pow(rij*rik,2))*rik*rik2+std::numeric_limits<double>::epsilon());

                dG_tensor[index_base + 3*j + coord] += dGdtheta*(rij2*(xyzs[3*i+coord]-xyzs[3*k+coord])-dot*(xyzs[3*i + coord]-xyzs[3*j+coord]))/
                  (rij*rij2*sqrt(1-pow(dot,2)/pow(rij*rik,2))*rik+std::numeric_limits<double>::epsilon());

                dG_tensor[index_base + 3*k + coord] += dGdtheta*(rik2*(xyzs[3*i + coord]-xyzs[3*j+coord])-dot*(xyzs[3*i+coord]-xyzs[3*k+coord]))/
                  (rij*sqrt(1-pow(dot,2)/pow(rij*rik,2))*rik*rik2+std::numeric_limits<double>::epsilon());
              }
            }
          }
        }
      }
    }
    counter += num_symFuns[2*types[i]] + num_symFuns[2*types[i]+1];
  }
}

CutoffFunction* SymmetryFunctionSet::switch_CutFun(int cutoff_type, double cutoff)
{
  CutoffFunction* cutfun;
  switch (cutoff_type) {
    case 0:
      cutfun = new ConstCutoffFunction(cutoff);
      break;
    case 1:
      cutfun = new CosCutoffFunction(cutoff);
      break;
    case 2:
      cutfun = new TanhCutoffFunction(cutoff);
      break;
  }
  return cutfun;
}

TwoBodySymmetryFunction* SymmetryFunctionSet::switch_TwoBodySymFun(int funtype, int num_prms, double* prms, CutoffFunction* cutfun)
{
  TwoBodySymmetryFunction* symFun;
  switch (funtype){
    case 0:
      symFun = new BehlerG2(num_prms, prms, cutfun);
      break;
  }
  return symFun;
}

ThreeBodySymmetryFunction* SymmetryFunctionSet::switch_ThreeBodySymFun(int funtype, int num_prms, double* prms, CutoffFunction* cutfun)
{
  ThreeBodySymmetryFunction* symFun;
  switch (funtype){
    case 0:
      symFun = new BehlerG4(num_prms, prms, cutfun);
      break;
  }
  return symFun;
}
