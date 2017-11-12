import sympy as _sp
from sympy.parsing.sympy_parser import parse_expr

rij, rik, costheta = _sp.symbols("rij rik costheta")

header_twoBody = """
class {0}: public TwoBodySymmetryFunction
{{
    public:
        {0}(int num_prms, double* prms_i,
          std::shared_ptr<CutoffFunction> cutfun_i):
          TwoBodySymmetryFunction(num_prms, prms_i, cutfun_i){{}};
        double eval(double rij);
        double drij(double rij);
}};
"""

header_threeBody = """
class {0}: public ThreeBodySymmetryFunction
{{
  public:
    {0}(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun_i):
      ThreeBodySymmetryFunction(num_prms, prms, cutfun_i){{}};
    double eval(double rij, double rik, double costheta);
    double drij(double rij, double rik, double costheta);
    double drik(double rij, double rik, double costheta);
    double dcostheta(double rij, double rik, double costheta);
}};
"""

method_twoBody = """
double {}::{}(double rij)
{{
  return {};
}};
"""

method_threeBody = """
double {}::{}(double rij, double rik, double costheta)
{{
  return {};
}};
"""

user_funs = {"fcut":"cutfun->eval", "dfcut":"cutfun->derivative"}

def format_prms(num_prms, s):
    # Replace prm0 with prms[0]
    for i in range(num_prms):
        s = s.replace("prm{:d}".format(i), "prms[{:d}]".format(i))
    return s

def format_py(s):
    return s

# Read custom symmetry function file
twoBodySymFuns = []
threeBodySymFuns = []
with open("customSymFuns.txt", "r") as fin:
    for line in fin:
        if line.startswith("TwoBodySymFun"):
            sp = line.split()
            twoBodySymFuns.append([sp[1], int(sp[2]), " ".join(sp[3::])])
        if line.startswith("ThreeBodySymFun"):
            sp = line.split()
            threeBodySymFuns.append([sp[1], int(sp[2]), " ".join(sp[3::])])

with open("symmetryFunctions.h", "r") as fin:
    lines = fin.readlines()

lines = (lines[0:(lines.index("// AUTOMATIC Start of custom TwoBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom TwoBodySymFuns\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC Start of custom ThreeBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom ThreeBodySymFuns\n")::])

with open("symmetryFunctions.h", "w") as fout:
    for line in lines:
        fout.write(line)
        if line.startswith("// AUTOMATIC Start of custom TwoBodySymFuns"):
            for symfun in twoBodySymFuns:
                fout.write(header_twoBody.format(symfun[0]))
        if line.startswith("// AUTOMATIC Start of custom ThreeBodySymFuns"):
            for symfun in threeBodySymFuns:
                fout.write(header_threeBody.format(symfun[0]))

with open("symmetryFunctions.cpp", "r") as fin:
    lines = fin.readlines()

lines = (lines[0:(lines.index("// AUTOMATIC Start of custom TwoBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom TwoBodySymFuns\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC Start of custom ThreeBodySymFuns\n")+1)] +
    lines[lines.index("// AUTOMATIC End of custom ThreeBodySymFuns\n")::])

with open("symmetryFunctions.cpp", "w") as fout:
    for line in lines:
        fout.write(line)
        if line.startswith("// AUTOMATIC Start of custom TwoBodySymFuns"):
            for symfun in twoBodySymFuns:
                fout.write(method_twoBody.format(symfun[0],"eval",
                    format_prms(symfun[1],_sp.ccode(symfun[2],
                    user_functions = user_funs))))
                deriv = str(_sp.simplify(
                    _sp.Derivative(parse_expr(symfun[2]), rij).doit()))
                deriv = deriv.replace("Derivative(fcut(rij), rij)", "dfcut(rij)")
                fout.write(method_twoBody.format(symfun[0],"drij",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))
        elif line.startswith("// AUTOMATIC Start of custom ThreeBodySymFuns"):
            for symfun in threeBodySymFuns:
                fout.write(method_threeBody.format(symfun[0],"eval",
                    format_prms(symfun[1],_sp.ccode(symfun[2],
                    user_functions = user_funs))))
                # Derivative with respect to rij
                deriv = str(_sp.simplify(
                    _sp.Derivative(parse_expr(symfun[2]), rij).doit()))
                deriv = deriv.replace("Derivative(fcut(rij), rij)", "dfcut(rij)")
                deriv = deriv.replace("Derivative(fcut(rik), rik)", "dfcut(rik)")
                fout.write(method_threeBody.format(symfun[0],"drij",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))
                # Derivative with respect to rik
                deriv = str(_sp.simplify(
                    _sp.Derivative(parse_expr(symfun[2]), rik).doit()))
                deriv = deriv.replace("Derivative(fcut(rij), rij)", "dfcut(rij)")
                deriv = deriv.replace("Derivative(fcut(rik), rik)", "dfcut(rik)")
                fout.write(method_threeBody.format(symfun[0],"drik",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))
                # Derivative with respect to costheta
                deriv = str(_sp.simplify(
                    _sp.Derivative(parse_expr(symfun[2]), costheta).doit()))
                fout.write(method_threeBody.format(symfun[0],"dcostheta",
                    format_prms(symfun[1],_sp.ccode(deriv,
                    user_functions = user_funs))))

with open("symmetryFunctionSet.cpp", "r") as fin:
    lines = fin.readlines()

lines = (lines[0:(lines.index("// AUTOMATIC TwoBodySymmetryFunction switch start\n")+1)] +
    lines[lines.index("// AUTOMATIC TwoBodySymmetryFunction switch end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC ThreeBodySymmetryFunction switch start\n")+1)] +
    lines[lines.index("// AUTOMATIC ThreeBodySymmetryFunction switch end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC available_symFuns start\n")+1)] +
    lines[lines.index("// AUTOMATIC available_symFuns end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC get_TwoBodySymFuns start\n")+1)] +
    lines[lines.index("// AUTOMATIC get_TwoBodySymFuns end\n")::])
lines = (lines[0:(lines.index("// AUTOMATIC get_ThreeBodySymFuns start\n")+1)] +
    lines[lines.index("// AUTOMATIC get_ThreeBodySymFuns end\n")::])

case_string = """    case {}:
      symFun = std::make_shared<{}>(num_prms, prms, cutfun);
      break;
"""

switch_string = """  if (strcmp(name, "{}") == 0)
  {{
    id = {};
  }}
"""

with open("symmetryFunctionSet.cpp", "w") as fout:
    for line in lines:
        fout.write(line)
        if line.startswith("// AUTOMATIC available_symFuns start"):
            fout.write('  printf("TwoBodySymmetryFunctions: (key: name, # of parameters)\\n");\n')
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(i, symfun[0], symfun[1]))
            fout.write('  printf("ThreeBodySymmetryFunctions: (key: name, # of parameters)\\n");\n')
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write('  printf("{}: {}, {}\\n");\n'.format(i, symfun[0], symfun[1]))

        elif line.startswith("// AUTOMATIC TwoBodySymmetryFunction switch start"):
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write(case_string.format(i, symfun[0]))
        elif line.startswith("// AUTOMATIC ThreeBodySymmetryFunction switch start"):
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write(case_string.format(i, symfun[0]))
        elif line.startswith("// AUTOMATIC get_TwoBodySymFuns start"):
            for i, symfun in enumerate(twoBodySymFuns):
                fout.write(switch_string.format(symfun[0], i))
        elif line.startswith("// AUTOMATIC get_ThreeBodySymFuns start"):
            for i, symfun in enumerate(threeBodySymFuns):
                fout.write(switch_string.format(symfun[0], i))
