/*

CAUTION: Part of this file is written by the python script generateSymFuns.py.
These parts are marked by comment lines starting with AUTOMATIC. Do not alter
anything between these tags.
*/

#include "symmetryFunctionSet.h"
#include <stdio.h>
#include <limits>
#include <math.h>
#include <algorithm>
#include <string.h>

SymmetryFunctionSet::SymmetryFunctionSet(int num_atomtypes_i):
twoBodySymFuns(num_atomtypes_i*num_atomtypes_i),
threeBodySymFuns(num_atomtypes_i*num_atomtypes_i*num_atomtypes_i)
{
  num_atomtypes = num_atomtypes_i;
  num_atomtypes2 = pow(num_atomtypes, 2);
  num_symFuns = new int[2*num_atomtypes]();
  pos_twoBody = new int[num_atomtypes2]();
  pos_threeBody = new int[num_atomtypes2*num_atomtypes]();
  printf("Constructor called with %d atom types\n",num_atomtypes);
}

SymmetryFunctionSet::~SymmetryFunctionSet()
{
  printf("Destructor called\n");
  for (int i = 0; i < num_atomtypes; i++)
  {
    for (int j = 0; j < num_atomtypes; j++)
    {
      twoBodySymFuns[num_atomtypes*i + j].clear();
    }
  }
  delete[] pos_twoBody;
  delete[] pos_threeBody;
  delete[] num_symFuns;
  printf("Destructor finished\n");
}

void SymmetryFunctionSet::add_TwoBodySymmetryFunction(
  int type1, int type2, int funtype, int num_prms, double* prms,
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun = switch_CutFun(cutoff_type, cutoff);
  std::shared_ptr<TwoBodySymmetryFunction> symfun = switch_TwoBodySymFun(
    funtype, num_prms, prms, cutfun);

  twoBodySymFuns[type1*num_atomtypes+type2].push_back(symfun);
  num_symFuns[2*type1]++;
  for (int i = type2+1; i < num_atomtypes; i++){
    pos_twoBody[num_atomtypes*type1 + i]++;
  }
}

void SymmetryFunctionSet::add_ThreeBodySymmetryFunction(
  int type1, int type2, int type3, int funtype, int num_prms, double* prms,
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun = switch_CutFun(cutoff_type, cutoff);
  std::shared_ptr<ThreeBodySymmetryFunction> symfun = switch_ThreeBodySymFun(
    funtype, num_prms, prms, cutfun);
  // Atomtype2 and atomtype3 are sorted to maintain symmetry
  threeBodySymFuns[num_atomtypes2*type1 + num_atomtypes*std::min(type2,type3) +
    std::max(type2,type3)].push_back(symfun);
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

int SymmetryFunctionSet::get_CutFun_by_name(const char* name)
{
  int id = -1;
  if (strcmp(name, "const") == 0)
  {
    id = 0;
  } else if (strcmp(name, "cos") == 0)
  {
    id = 1;
  } else if (strcmp(name, "tanh") == 0)
  {
    id = 2;
  }
  return id;
}

int SymmetryFunctionSet::get_TwoBodySymFun_by_name(const char* name)
{
  int id = -1;

  if (strcmp(name, "BehlerG2") == 0)
  {
    id = 0;
  }
  return id;
}

int SymmetryFunctionSet::get_ThreeBodySymFun_by_name(const char* name)
{
  int id = -1;

  if (strcmp(name, "BehlerG4") == 0)
  {
    id = 0;
  }
  return id;
}

void SymmetryFunctionSet::print_symFuns()
{
  printf("Number of atom types: %d\n", num_atomtypes);
  //printf("TwoBodySymmetryFunctions:\n");
  for (int ti = 0; ti < num_atomtypes; ti++)
  {
    printf("ti = %d\n", ti);
    printf("Number of TwoBodySymmetryFunction for atom type %d is %d\n",
      ti, num_symFuns[2*ti]);
    printf("Number of ThreeBodySymmetryFunction for atom type %d is %d\n",
      ti, num_symFuns[2*ti+1]);
  }
}

void SymmetryFunctionSet::available_symFuns()
{
// AUTOMATIC available_symFuns start
  printf("TwoBodySymmetryFunctions: (key: name, # of parameters)\n");
  printf("0: BehlerG2, 2\n");
  printf("ThreeBodySymmetryFunctions: (key: name, # of parameters)\n");
  printf("0: BehlerG4, 3\n");
// AUTOMATIC available_symFuns end
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
void SymmetryFunctionSet::eval_old(
  int num_atoms, int* types, double* xyzs, double* G_vector)
{
  double rij, rik, theta;
  int counter = 0;
  int i, j, k, two_Body_i, three_Body_i;

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
        for (two_Body_i = 0;
          two_Body_i < twoBodySymFuns[types[i]*num_atomtypes + types[j]].size();
          two_Body_i++)
        {
          G_vector[counter + pos_twoBody[num_atomtypes*types[i] + types[j]] +
            two_Body_i] += twoBodySymFuns[types[i]*num_atomtypes + types[j]]
            [two_Body_i]->eval(rij);
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

            for (three_Body_i = 0;
              three_Body_i < threeBodySymFuns[num_atomtypes2*types[i] +
              num_atomtypes*types[j]+types[k]].size(); three_Body_i++)
            {
              if (types[j] == types[k])
              {
                G_vector[counter + num_symFuns[2*types[i]] + pos_threeBody[num_atomtypes2*types[i] +
                  num_atomtypes*types[j] + types[k]]+ three_Body_i] += 0.5*
                  threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]][three_Body_i]->eval(rij, rik, theta);
              } else
              {
                G_vector[counter + num_symFuns[2*types[i]] + pos_threeBody[num_atomtypes2*types[i] +
                  num_atomtypes*types[j] + types[k]]+ three_Body_i] +=
                  threeBodySymFuns[num_atomtypes2*types[i]+num_atomtypes*types[j]+types[k]][three_Body_i]->eval(rij, rik, theta);
              }
            }
          }
        }
      }
    }
    counter += num_symFuns[2*types[i]] + num_symFuns[2*types[i]+1];
  }
}

void SymmetryFunctionSet::eval(
  int num_atoms, int* types, double* xyzs, double* G_vector)
{
  double rij, rik, rjk, theta_i, theta_j, theta_k;
  int i, j, k, two_Body_i, three_Body_i, type_ij, type_ji, type_ijk, type_jki,
    type_kij;

  // Figure out the positions of symmetry functions centered on atom i and
  // save in pos_atoms
  int counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_symFuns[2*types[i]] + num_symFuns[2*types[i] + 1];
  }

  // Actual evaluation of the symmetry functions. The sum over other atoms
  // is done for all symmetry functions simultaniously.
  for (i = 0; i < num_atoms; i++)
  {
    // Loop over other atoms. To reduce computational effort symmetry functions
    // centered on atom j are also evaluated here. This allows to restrict the
    // loop to values j = i + 1.
    for (j = i + 1; j < num_atoms; j++)
    {
      rij = sqrt(pow(xyzs[3*i]-xyzs[3*j], 2) +
                pow(xyzs[3*i + 1]-xyzs[3*j+1], 2) +
                pow(xyzs[3*i + 2]-xyzs[3*j+2], 2));
      // Add to two body symmetry functions centered on atom i
      type_ij = types[i]*num_atomtypes+types[j];
      for (two_Body_i = 0; two_Body_i < twoBodySymFuns[type_ij].size();
        two_Body_i++)
      {
        G_vector[pos_atoms[i] + pos_twoBody[type_ij] + two_Body_i] +=
          twoBodySymFuns[type_ij][two_Body_i]->eval(rij);
      }
      // Add to two body symmetry functions centered on atom j
      type_ji = types[j]*num_atomtypes+types[i];
      for (two_Body_i = 0; two_Body_i < twoBodySymFuns[type_ji].size();
        two_Body_i++)
      {
        G_vector[pos_atoms[j] + pos_twoBody[type_ji] + two_Body_i] +=
          twoBodySymFuns[type_ji][two_Body_i]->eval(rij);
      }
      for (k = j + 1; k < num_atoms; k++)
      {
        rik = sqrt(pow(xyzs[3*i]-xyzs[3*k], 2) +
                  pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                  pow(xyzs[3*i+2]-xyzs[3*k+2], 2));
        rjk = sqrt(pow(xyzs[3*j]-xyzs[3*k], 2) +
                  pow(xyzs[3*j+1]-xyzs[3*k+1], 2) +
                  pow(xyzs[3*j+2]-xyzs[3*k+2], 2));
        // Calculate the angle between rij, rik and rjk
        theta_i = acos(((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
          (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
          (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]))/(rij*rik));
        theta_j = acos(((xyzs[3*j]-xyzs[3*k])*(xyzs[3*j]-xyzs[3*i]) +
          (xyzs[3*j+1]-xyzs[3*k+1])*(xyzs[3*j+1]-xyzs[3*i+1]) +
          (xyzs[3*j+2]-xyzs[3*k+2])*(xyzs[3*j+2]-xyzs[3*i+2]))/(rjk*rij));
        theta_k = acos(((xyzs[3*k]-xyzs[3*i])*(xyzs[3*k]-xyzs[3*j]) +
          (xyzs[3*k+1]-xyzs[3*i+1])*(xyzs[3*k+1]-xyzs[3*j+1]) +
          (xyzs[3*k+2]-xyzs[3*i+2])*(xyzs[3*k+2]-xyzs[3*j+2]))/(rik*rjk));

        // As described in add_ThreeBodySymmetryFunction() the type of the three
        // body symmetry function consists of the atom type of the atom the
        // function is centered on an the sorted pair of atom types of the two
        // remaining atoms.

        // Add to three body symmetry functions centered on atom i.
        type_ijk = num_atomtypes2*types[i] +
          num_atomtypes*std::min(types[j], types[k]) +
          std::max(types[j], types[k]);
        for (three_Body_i = 0; three_Body_i < threeBodySymFuns[type_ijk].size();
          three_Body_i++)
        {
          G_vector[pos_atoms[i] + num_symFuns[2*types[i]] +
            pos_threeBody[type_ijk] + three_Body_i] +=
            threeBodySymFuns[type_ijk][three_Body_i]->eval(rij, rik, theta_i);
        }

        // Add to three body symmetry functions centered on atom j.
        type_jki = num_atomtypes2*types[j] +
          num_atomtypes*std::min(types[i],types[k]) +
          std::max(types[i],types[k]);
        for (three_Body_i = 0; three_Body_i < threeBodySymFuns[type_jki].size();
          three_Body_i++)
        {
          G_vector[pos_atoms[j] + num_symFuns[2*types[j]] +
            pos_threeBody[type_jki] + three_Body_i] +=
            threeBodySymFuns[type_jki][three_Body_i]->eval(rij, rjk, theta_j);
        }

        // Add to three body symmetry functions centered on atom k.
        type_kij = num_atomtypes2*types[k] +
          num_atomtypes*std::min(types[i],types[j]) +
          std::max(types[i],types[j]);
        for (three_Body_i = 0; three_Body_i < threeBodySymFuns[type_kij].size();
          three_Body_i++)
        {
          G_vector[pos_atoms[k] + num_symFuns[2*types[k]] +
            pos_threeBody[type_kij] + three_Body_i] +=
            threeBodySymFuns[type_kij][three_Body_i]->eval(rjk, rik, theta_k);
        }
      }
    }
  }
  delete[] pos_atoms;
}

void SymmetryFunctionSet::eval_derivatives_old(
  int num_atoms, int* types, double* xyzs, double* dG_tensor)
{
  double rij, rij2, rik, rik2, theta, dGdr, dGdrij, dGdrik, dGdtheta, dot;
  int counter = 0;
  int i, j, k, two_Body_i, three_Body_i, coord, index_base;

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
        for (two_Body_i = 0; two_Body_i < twoBodySymFuns[types[i]*num_atomtypes+types[j]].size(); two_Body_i++)
        {
          dGdr = twoBodySymFuns[types[i]*num_atomtypes+types[j]][two_Body_i]->drij(rij);
          // Loop over the three cartesian coordinates
          for (coord = 0; coord < 3; coord++){
            // dG/dx is calculated as product of dG/dr * dr/dx
            dG_tensor[3*num_atoms*(counter + pos_twoBody[num_atomtypes*types[i]+types[j]] + two_Body_i) +
              3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
            dG_tensor[3*num_atoms*(counter + pos_twoBody[num_atomtypes*types[i]+types[j]] + two_Body_i) +
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

void SymmetryFunctionSet::eval_derivatives(
  int num_atoms, int* types, double* xyzs, double* dG_tensor)
{
  double rij, rij2, rik, rik2, rjk, rjk2, theta_i, theta_j, theta_k,
    dGdr, dGdrij, dGdrik, dGdtheta, dot_i, dot_j, dot_k;
  int i, j, k, two_Body_i, three_Body_i, coord, index_base, type_ij, type_ji,
    type_ijk, type_jki, type_kij;

  int counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_symFuns[2*types[i]] + num_symFuns[2*types[i]+1];
  }

  for (i = 0; i < num_atoms; i++)
  {
    for (j = i + 1; j < num_atoms; j++)
    {

      rij2 = pow(xyzs[3*i]-xyzs[3*j], 2) + pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                pow(xyzs[3*i+2]-xyzs[3*j+2], 2);
      rij = sqrt(rij2);
      // dG/dx is calculated as product of dG/dr * dr/dx
      // Add to two body symmetry functions centered on atom i
      type_ij = types[i]*num_atomtypes+types[j];
      for (two_Body_i = 0; two_Body_i < twoBodySymFuns[type_ij].size();
        two_Body_i++)
      {
        dGdr = twoBodySymFuns[type_ij][two_Body_i]->drij(rij);
        // Loop over the three cartesian coordinates
        for (coord = 0; coord < 3; coord++){
          dG_tensor[3*num_atoms*(pos_atoms[i] + pos_twoBody[type_ij] +
            two_Body_i) + 3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
          dG_tensor[3*num_atoms*(pos_atoms[i] + pos_twoBody[type_ij] + two_Body_i) +
            3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
        }
      }
      // Add to two body symmetry functions centered on atom j
      type_ji = types[j]*num_atomtypes+types[i];
      for (two_Body_i = 0; two_Body_i < twoBodySymFuns[type_ji].size();
        two_Body_i++)
      {
        dGdr = twoBodySymFuns[type_ji][two_Body_i]->drij(rij);
        // Loop over the three cartesian coordinates
        for (coord = 0; coord < 3; coord++){
          dG_tensor[3*num_atoms*(pos_atoms[j] + pos_twoBody[type_ji] + two_Body_i) +
            3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
          dG_tensor[3*num_atoms*(pos_atoms[j] + pos_twoBody[type_ji] + two_Body_i) +
            3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
        }
      }

      for (k = j + 1; k < num_atoms; k++)
      {
        rik2 = pow(xyzs[3*i]-xyzs[3*k], 2) + pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                  pow(xyzs[3*i+2]-xyzs[3*k+2], 2);
        rik = sqrt(rik2);
        rjk2 = pow(xyzs[3*j]-xyzs[3*k], 2) + pow(xyzs[3*j+1]-xyzs[3*k+1], 2) +
                  pow(xyzs[3*j+2]-xyzs[3*k+2], 2);
        rjk = sqrt(rjk2);

        dot_i = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
          (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
          (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]));
        dot_j = ((xyzs[3*j]-xyzs[3*k])*(xyzs[3*j]-xyzs[3*i]) +
          (xyzs[3*j+1]-xyzs[3*k+1])*(xyzs[3*j+1]-xyzs[3*i+1]) +
          (xyzs[3*j+2]-xyzs[3*k+2])*(xyzs[3*j+2]-xyzs[3*i+2]));
        dot_k = ((xyzs[3*k]-xyzs[3*i])*(xyzs[3*k]-xyzs[3*j]) +
          (xyzs[3*k+1]-xyzs[3*i+1])*(xyzs[3*k+1]-xyzs[3*j+1]) +
          (xyzs[3*k+2]-xyzs[3*i+2])*(xyzs[3*k+2]-xyzs[3*j+2]));
        // Calculate the angle between rij and rik
        theta_i = acos(dot_i/(rij*rik));
        theta_j = acos(dot_j/(rjk*rij));
        theta_k = acos(dot_k/(rik*rjk));

        // Add to three body symmetry functions centered on atom i.
        type_ijk = num_atomtypes2*types[i] +
          num_atomtypes*std::min(types[j],types[k]) +
          std::max(types[j],types[k]);
        for (three_Body_i = 0; three_Body_i < threeBodySymFuns[type_ijk].size();
          three_Body_i++)
        {
          dGdrij = threeBodySymFuns[type_ijk][three_Body_i]->drij(
            rij, rik, theta_i);
          dGdrik = threeBodySymFuns[type_ijk][three_Body_i]->drik(
            rij, rik, theta_i);
          dGdtheta = threeBodySymFuns[type_ijk][three_Body_i]->dtheta(
            rij, rik, theta_i);

          index_base = 3*num_atoms*(pos_atoms[i] + num_symFuns[2*types[i]] +
            pos_threeBody[type_ijk]+ three_Body_i);
          for (coord = 0; coord < 3; coord++){
            // Derivative with respect to rij
            dG_tensor[index_base + 3*i + coord] += dGdrij*
              (xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
            dG_tensor[index_base + 3*j + coord] += dGdrij*
              (-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
            // Derivative with respect to rik
            dG_tensor[index_base + 3*i + coord] += dGdrik*
              (xyzs[3*i+coord]-xyzs[3*k+coord])/rik;
            dG_tensor[index_base + 3*k + coord] += dGdrik*
              (-xyzs[3*i+coord]+xyzs[3*k+coord])/rik;

            // Derivative with respect to theta
            dG_tensor[index_base + 3*i + coord] += dGdtheta*
              (dot_i*rik2*(xyzs[3*i + coord]-xyzs[3*j + coord]) +
              dot_i*rij2*(xyzs[3*i + coord]-xyzs[3*k + coord]) +
              rij2*rik2*(xyzs[3*j + coord]+xyzs[3*k + coord] -
              2*xyzs[3*i + coord])) /
              (rij*rij2*sqrt(1-pow(dot_i,2)/pow(rij*rik,2))*rik*rik2 +
              std::numeric_limits<double>::epsilon());

            dG_tensor[index_base + 3*j + coord] += dGdtheta*
              (rij2*(xyzs[3*i + coord]-xyzs[3*k + coord]) -
              dot_i*(xyzs[3*i + coord]-xyzs[3*j + coord]))/
              (rij*rij2*sqrt(1-pow(dot_i,2)/pow(rij*rik,2))*rik +
              std::numeric_limits<double>::epsilon());

            dG_tensor[index_base + 3*k + coord] += dGdtheta*
              (rik2*(xyzs[3*i + coord]-xyzs[3*j + coord]) -
              dot_i*(xyzs[3*i + coord]-xyzs[3*k + coord]))/
              (rij*sqrt(1-pow(dot_i,2)/pow(rij*rik,2))*rik*rik2 +
              std::numeric_limits<double>::epsilon());
          }
        }

        // Add to three body symmetry functions centered on atom j.
        type_jki = num_atomtypes2*types[j] +
          num_atomtypes*std::min(types[k],types[i]) +
          std::max(types[k],types[i]);
        for (three_Body_i = 0; three_Body_i < threeBodySymFuns[type_jki].size();
          three_Body_i++)
        {
          dGdrij = threeBodySymFuns[type_jki][three_Body_i]->drij(
            rjk, rij, theta_j);
          dGdrik = threeBodySymFuns[type_jki][three_Body_i]->drik(
            rjk, rij, theta_j);
          dGdtheta = threeBodySymFuns[type_jki][three_Body_i]->dtheta(
            rjk, rij, theta_j);

          index_base = 3*num_atoms*(pos_atoms[j] + num_symFuns[2*types[j]] +
            pos_threeBody[type_jki]+ three_Body_i);
          for (coord = 0; coord < 3; coord++)
          {
            // Derivative with respect to rij
            dG_tensor[index_base + 3*j + coord] += dGdrij*
              (xyzs[3*j+coord]-xyzs[3*k+coord])/rjk;
            dG_tensor[index_base + 3*k + coord] += dGdrij*
              (-xyzs[3*j+coord]+xyzs[3*k+coord])/rjk;
            // Derivative with respect to rik
            dG_tensor[index_base + 3*j + coord] += dGdrik*
              (xyzs[3*j+coord]-xyzs[3*i+coord])/rij;
            dG_tensor[index_base + 3*i + coord] += dGdrik*
              (-xyzs[3*j+coord]+xyzs[3*i+coord])/rij;

            // Derivative with respect to theta
            dG_tensor[index_base + 3*j + coord] += dGdtheta*
              (dot_j*rij2*(xyzs[3*j + coord]-xyzs[3*k + coord]) +
              dot_j*rjk2*(xyzs[3*j + coord]-xyzs[3*i + coord]) +
              rjk2*rij2*(xyzs[3*k + coord]+xyzs[3*i + coord] -
              2*xyzs[3*j + coord])) /
              (rjk*rjk2*sqrt(1-pow(dot_j,2)/pow(rjk*rij,2))*rij*rij2 +
              std::numeric_limits<double>::epsilon());

            dG_tensor[index_base + 3*k + coord] += dGdtheta*
              (rjk2*(xyzs[3*j + coord]-xyzs[3*i + coord]) -
              dot_j*(xyzs[3*j + coord]-xyzs[3*k + coord]))/
              (rjk*rjk2*sqrt(1-pow(dot_j,2)/pow(rjk*rij,2))*rij +
              std::numeric_limits<double>::epsilon());

            dG_tensor[index_base + 3*i + coord] += dGdtheta*
              (rij2*(xyzs[3*j + coord]-xyzs[3*k + coord]) -
              dot_j*(xyzs[3*j + coord]-xyzs[3*i + coord]))/
              (rjk*sqrt(1-pow(dot_j,2)/pow(rjk*rij,2))*rij*rij2 +
              std::numeric_limits<double>::epsilon());
          }
        }

        // Add to three body symmetry functions centered on atom k.
        type_kij = num_atomtypes2*types[k] +
          num_atomtypes*std::min(types[i],types[j]) +
          std::max(types[i],types[j]);
        for (three_Body_i = 0; three_Body_i < threeBodySymFuns[type_kij].size();
          three_Body_i++)
        {
          dGdrij = threeBodySymFuns[type_kij][three_Body_i]->drij(
            rik, rjk, theta_k);
          dGdrik = threeBodySymFuns[type_kij][three_Body_i]->drik(
            rik, rjk, theta_k);
          dGdtheta = threeBodySymFuns[type_kij][three_Body_i]->dtheta(
            rik, rjk, theta_k);

          index_base = 3*num_atoms*(pos_atoms[k] + num_symFuns[2*types[k]] +
            pos_threeBody[type_kij] + three_Body_i);
          for (coord = 0; coord < 3; coord++)
          {
            // Derivative with respect to rij
            dG_tensor[index_base + 3*k + coord] += dGdrij*
              (xyzs[3*k+coord]-xyzs[3*i+coord])/rik;
            dG_tensor[index_base + 3*i + coord] += dGdrij*
              (-xyzs[3*k+coord]+xyzs[3*i+coord])/rik;
            // Derivative with respect to rik
            dG_tensor[index_base + 3*k + coord] += dGdrik*
              (xyzs[3*k+coord]-xyzs[3*j+coord])/rjk;
            dG_tensor[index_base + 3*j + coord] += dGdrik*
              (-xyzs[3*k+coord]+xyzs[3*j+coord])/rjk;


            // Derivative with respect to theta
            dG_tensor[index_base + 3*k + coord] += dGdtheta*
              (dot_k*rjk2*(xyzs[3*k + coord]-xyzs[3*i + coord]) +
              dot_k*rik2*(xyzs[3*k + coord]-xyzs[3*j + coord]) +
              rik2*rjk2*(xyzs[3*i + coord]+xyzs[3*j + coord] -
              2*xyzs[3*k + coord])) /
              (rik*rik2*sqrt(1-pow(dot_k,2)/pow(rik*rjk,2))*rjk*rjk2 +
              std::numeric_limits<double>::epsilon());

            dG_tensor[index_base + 3*i + coord] += dGdtheta*
              (rik2*(xyzs[3*k + coord]-xyzs[3*j + coord]) -
              dot_k*(xyzs[3*k + coord]-xyzs[3*i + coord]))/
              (rik*rik2*sqrt(1-pow(dot_k,2)/pow(rik*rjk,2))*rjk +
              std::numeric_limits<double>::epsilon());

            dG_tensor[index_base + 3*j + coord] += dGdtheta*
              (rjk2*(xyzs[3*k + coord]-xyzs[3*i + coord]) -
              dot_k*(xyzs[3*k + coord]-xyzs[3*j + coord]))/
              (rik*sqrt(1-pow(dot_k,2)/pow(rik*rjk,2))*rjk*rjk2 +
              std::numeric_limits<double>::epsilon());
          }
        }

      }
    }
  }
  delete[] pos_atoms;
}

std::shared_ptr<CutoffFunction> SymmetryFunctionSet::switch_CutFun(
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun;
  switch (cutoff_type) {
    case 0:
      cutfun = std::make_shared<ConstCutoffFunction>(cutoff);
      break;
    case 1:
      cutfun = std::make_shared<CosCutoffFunction>(cutoff);
      break;
    case 2:
      cutfun = std::make_shared<TanhCutoffFunction>(cutoff);
      break;
  }
  return cutfun;
}

std::shared_ptr<TwoBodySymmetryFunction> SymmetryFunctionSet::switch_TwoBodySymFun(
  int funtype, int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<TwoBodySymmetryFunction> symFun;
  switch (funtype){
// AUTOMATIC TwoBodySymmetryFunction switch start
    case 0:
      symFun = std::make_shared<BehlerG2>(num_prms, prms, cutfun);
      break;
// AUTOMATIC TwoBodySymmetryFunction switch end
    default:
      printf("No function type %d\n", funtype);
  }
  return symFun;
}

std::shared_ptr<ThreeBodySymmetryFunction> SymmetryFunctionSet::switch_ThreeBodySymFun(
  int funtype, int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<ThreeBodySymmetryFunction> symFun;
  switch (funtype){
// AUTOMATIC ThreeBodySymmetryFunction switch start
    case 0:
      symFun = std::make_shared<BehlerG4>(num_prms, prms, cutfun);
      break;
// AUTOMATIC ThreeBodySymmetryFunction switch end
    default:
      printf("No function type %d\n", funtype);
  }
  return symFun;
}
