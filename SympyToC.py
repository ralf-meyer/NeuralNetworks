
from sympy import Symbol, cse, ccode
from sympy.utilities import numbered_symbols

# Inspired by http://stackoverflow.com/questions/22665990/optimize-code-generated-by-sympy
def sympyToC(symname, symfunc):

    if type(symfunc)!=int and type(symfunc)!=float:
        tmpsyms = numbered_symbols("tmp")
        parse_string=""
        arg_string=""
        symbols, simple = cse(symfunc, symbols=tmpsyms)
        symbolslist = map(lambda x:str(x), list(symfunc.atoms(Symbol)) )
        symbolslist.sort()
        varstring=",".join( " double "+x for x in symbolslist )[1:]
        c_code = "\n// Needed GCC 4.7+\n\n"
        c_code += "#pragma GCC push_options\n"
        c_code += "#pragma GCC optimize (\"Ofast\")\n\n"
        #c_code += "#include <Python.h>\n"
        c_code += "#include \"" + symname + ".h\"\n"
        c_code += "#include <math.h>\n\n"
        #Part for calling function from python
#        c_code += "int main(int argc, char **argv){\n"
#        c_code += "  Py_SetProgramName(argv[0]);\n"
#        c_code += "  Py_Initialize();\n"
#        c_code += "  init"+str(symname)+"();\n"
#        c_code += "  return 1;\n"
#        c_code += "};\n"        
#        
#        c_code += "static PyObject *"+str(symname)+"_system(PyObject *self, PyObject *args)\n"
#        c_code += "{\n"
#        ct=0
#        for x in symbolslist:
#            c_code += "  double *"+str(x)+";\n"
#            parse_string += ","+str(x)
#            if ct==0:
#                arg_string += str(x)
#            else:
#                arg_string += ","+str(x)
#            ct+=1
#        c_code += "double result;\n"
#        c_code += "  if (!PyArg_ParseTuple(args, \"d\""+parse_string+"))\n"
#        c_code += "    return NULL;\n"
#        c_code += "  result = "+str(symname)+"("+arg_string+");\n"
#        c_code += "  return Py_BuildValue(\"d\", result);\n"
#        c_code += "}\n"  
#            
#        c_code += "static PyMethodDef "+str(symname)+"_Methods[] = {\n"
#        c_code += "  ...\n"
#        c_code += "  {\""+str(symname)+"\",  "+str(symname)+"_system, METH_VARARGS,\n"
#        c_code += "  \""+str(symname)+"\"}\n"
#        c_code += "  ...\n"
#        c_code += "  {NULL, NULL, 0, NULL}        /* Sentinel */"
#        c_code += "};\n"        
#            
#        c_code += "PyMODINIT_FUNC init"+str(symname)+"(void)\n"
#        c_code += "  {\n"
#        c_code += "  (void) Py_InitModule(\""+str(symname)+"\", "+str(symname)+"_Methods);\n"
#        c_code += "}\n"               
        c_code += "double "+str(symname)+"("+varstring+") {\n"
        c_code +=  "  double r;\n"
        for s in symbols:
            c_code +=  "  double " +ccode(s[0]) + " = " + ccode(s[1]) + ";\n"
        
        mystr=str(symfunc)
        if "Heaviside" in mystr:
            #get argument of Heaviside function
            heavi_idx=mystr.index("Heaviside")
            arg_start=mystr.index("(",heavi_idx)
            arg_end=mystr.index(")",heavi_idx)
            arg=mystr[arg_start+1:arg_end]
            c_code += "if(("+ccode(arg)+") < 0) {\n"
            c_code += "  r = 0; }\n"
            c_code += "else{"
            c_code +=  "  r = " + ccode(mystr[:heavi_idx-1])+";\n }"
        else:
            c_code +=  "  r = " + ccode(simple[0])+";\n"
        c_code +=  "  return r;\n"
        c_code += "}\n\n"
        c_code += "#pragma GCC pop_options\n"
    else:
        c_code = "\n// Needed GCC 4.7+\n\n"
        c_code += "#pragma GCC push_options\n"
        c_code += "#pragma GCC optimize (\"Ofast\")\n\n"
        c_code += "#include \"" + symname + ".h\"\n"
        c_code += "#include <math.h>\n\n"
        c_code += "double "+str(symname)+"() {\n"
        c_code +=  "  double r;\n"
        c_code +=  "  r = " +str(symfunc) +";\n"
        c_code +=  "  return r;\n"
        c_code += "}\n\n"
        c_code += "#pragma GCC pop_options\n"


    return c_code
    
def sympyToH(symname, symfunc):
    h_code = "\n"
    h_code += "#ifndef PROJECT__" + symname.upper() + "__H\n"
    h_code += "#define PROJECT__" + symname.upper() + "__H\n"
    h_code += "#ifdef __cplusplus\n"
    h_code += "extern \"C\" {\n"
    h_code += "#endif\n"

    if type(symfunc)!=int and type(symfunc)!=float:
        symbolslist = map(lambda x:str(x), list(symfunc.atoms(Symbol)) )
        symbolslist.sort()
        varstring=",".join( " double "+x for x in symbolslist )[1:]
    else:
        varstring=""

    h_code += "double "+str(symname)+"("+varstring+");\n"
    h_code += "#ifdef __cplusplus\n"
    h_code += "}\n"
    h_code += "#endif\n"
    h_code += "#endif\n"
    return h_code
