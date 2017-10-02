import sympy as _sp
from sympy.parsing.sympy_parser import parse_expr

rij, rik, theta = _sp.symbols("rij rik theta")

header_twoBody = """
class {0}: public TwoBodySymmetryFunction
{{
    public:
        {0}(int num_prms, double* prms_i, CutoffFunction* cutfun_i):
          TwoBodySymmetryFunction(num_prms, prms_i, cutfun_i){{}};
        double eval(double rij);
        double drij(double rij);
}};
"""

header_threeBody = """
class {0}: public ThreeBodySymmetryFunction
{{
  public:
    {0}(int num_prms, double* prms, CutoffFunction* cutfun_i):
      ThreeBodySymmetryFunction(num_prms, prms, cutfun_i){{}};
    double eval(double rij, double rik, double theta);
    double drij(double rij, double rik, double theta);
    double drik(double rij, double rik, double theta);
    double dtheta(double rij, double rik, double theta);
}};
"""

method_twoBody = """
double {}::{}(double rij)
{{
  return {};
}};
"""

method_threeBody = """
double {}::{}(double rij, double rik, double theta)
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

lines = (lines[0:(lines.index("// Start of custom TwoBodySymFuns\n")+1)] +
    lines[lines.index("// End of custom TwoBodySymFuns\n")::])
lines = (lines[0:(lines.index("// Start of custom ThreeBodySymFuns\n")+1)] +
    lines[lines.index("// End of custom ThreeBodySymFuns\n")::])

with open("symmetryFunctions.h", "w") as fout:
    for line in lines:
        fout.write(line)
        if line.startswith("// Start of custom TwoBodySymFuns"):
            for symfun in twoBodySymFuns:
                fout.write(header_twoBody.format(symfun[0]))
        if line.startswith("// Start of custom ThreeBodySymFuns"):
            for symfun in threeBodySymFuns:
                fout.write(header_threeBody.format(symfun[0]))

with open("symmetryFunctions.cpp", "r") as fin:
    lines = fin.readlines()

lines = (lines[0:(lines.index("// Start of custom TwoBodySymFuns\n")+1)] +
    lines[lines.index("// End of custom TwoBodySymFuns\n")::])
lines = (lines[0:(lines.index("// Start of custom ThreeBodySymFuns\n")+1)] +
    lines[lines.index("// End of custom ThreeBodySymFuns\n")::])

with open("symmetryFunctions.cpp", "w") as fout:
    for line in lines:
        fout.write(line)
        if line.startswith("// Start of custom TwoBodySymFuns"):
            for symfun in twoBodySymFuns:
                fout.write(method_twoBody.format(symfun[0],"eval",
                    format_prms(symfun[1],_sp.ccode(symfun[2], user_functions = user_funs))))
                deriv = str(_sp.Derivative(parse_expr(symfun[2]), rij).doit())
                deriv = deriv.replace("Derivative(fcut(rij), rij)", "dfcut(rij)")
                fout.write(method_twoBody.format(symfun[0],"drij",
                    format_prms(symfun[1],_sp.ccode(deriv, user_functions = user_funs))))
        if line.startswith("// Start of custom ThreeBodySymFuns"):
            for symfun in threeBodySymFuns:
                fout.write(method_threeBody.format(symfun[0],"eval",
                    format_prms(symfun[1],_sp.ccode(symfun[2], user_functions = user_funs))))
                # Derivative with respect to rij
                deriv = str(_sp.Derivative(parse_expr(symfun[2]), rij).doit())
                deriv = deriv.replace("Derivative(fcut(rij), rij)", "dfcut(rij)")
                deriv = deriv.replace("Derivative(fcut(rik), rik)", "dfcut(rik)")
                fout.write(method_threeBody.format(symfun[0],"drij",
                    format_prms(symfun[1],_sp.ccode(deriv, user_functions = user_funs))))
                # Derivative with respect to rik
                deriv = str(_sp.Derivative(parse_expr(symfun[2]), rik).doit())
                deriv = deriv.replace("Derivative(fcut(rij), rij)", "dfcut(rij)")
                deriv = deriv.replace("Derivative(fcut(rik), rik)", "dfcut(rik)")
                fout.write(method_threeBody.format(symfun[0],"drik",
                    format_prms(symfun[1],_sp.ccode(deriv, user_functions = user_funs))))
                # Derivative with respect to theta
                deriv = str(_sp.Derivative(parse_expr(symfun[2]), theta).doit())
                fout.write(method_threeBody.format(symfun[0],"dtheta",
                    format_prms(symfun[1],_sp.ccode(deriv, user_functions = user_funs))))


for symfun in twoBodySymFuns:
    print _sp.Derivative(parse_expr(symfun[2]), rij).doit()
