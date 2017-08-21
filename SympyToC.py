import re
from sympy import Symbol, cse, ccode,diff
from sympy.utilities import numbered_symbols

def is_variable(symbol,variables_list):
    out=False
    if symbol in variables_list:
        out=True
    return out


def addToSymmetryFunctionSet(symname_list, symfunc_list,variables_list,remove=True):
    
    with open("symmetryFunctionSet.cpp","r") as file:
        cpp_code=file.read()
        #find insert position
        file.close()
    with open("symmetryFunctionSet.cpp","w+") as file:
        try:
            start_add_idx=cpp_code.index("//Start of section for adding custom symmetry functions\n")\
            +len("//Start of section for adding custom symmetry functions\n")
            end_add_idx=cpp_code.index("//End of section for adding custom symmetry functions\n")
            print("symmetryFunctionSet found insert point!")
        except:
            #in case file is manually modified
            semi_idx=[i.start() for i in re.finditer(';', cpp_code)]
            eval_geom_idx=cpp_code.index("eval_geometry(")
            for i,idx in enumerate(semi_idx):
                if idx > eval_geom_idx:
                    start_add_idx=semi_idx[i-1]+1
                    temp_code="//Start of section for adding custom symmetry functions\n"
                    temp_code+="//End of section for adding custom symmetry functions\n"
                    cpp_code=cpp_code[:start_add_idx]+temp_code+cpp_code[start_add_idx:]
                    start_add_idx=start_add_idx+len("//Start of section for adding custom symmetry functions\n")
                    end_add_idx=cpp_code.index("//End of section for adding custom symmetry functions\n")
                    break 
        #remove old custom add functions
        if remove:
            cpp_code=cpp_code[:start_add_idx]+cpp_code[end_add_idx:]
        
        try:
            start_vec_idx=cpp_code.index("//Start of custom extern declaration\n")+len("//Start of custom extern declaration\n")
            end_vec_idx=cpp_code.index("//End of custom extern declaration\n")
        except:
            bracket_idx=[i.start() for i in re.finditer('}', cpp_code)]
            symfunset_idx=cpp_code.index("return new SymmetryFunctionSet()")
            for i,idx in enumerate(bracket_idx):
                if idx > symfunset_idx:
                    start_vec_idx=idx+1
                    end_vec_idx=start_vec_idx
                    temp_code="\n//Start of custom extern declaration\n"
                    temp_code+="//End of custom extern declaration\n"
                    cpp_code=cpp_code[:start_vec_idx]+temp_code+cpp_code[start_vec_idx:]
                    break 

        #remove old custom add functions
        cpp_code=cpp_code[:start_vec_idx]+cpp_code[end_vec_idx:]
        #insert new code

        for symname,symfuns in zip(symname_list,symfunc_list): 
            for symfunc in symfuns:
                if type(symfunc)!=int and type(symfunc)!=float:
                    #construct code
                    tmpsyms = numbered_symbols("tmp")
                    symbols, simple = cse(symfunc, symbols=tmpsyms)
                    symbolslist = map(lambda x:str(x), list(symfunc.atoms(Symbol)) )
                    symbolslist.sort()
                    varstring=",".join( " double "+x for x in symbolslist if not(is_variable(str(x),variables_list)))[1:]
                    argstring=",".join(x for x in symbolslist if not(is_variable(str(x),variables_list)))[:]
                    temp_code ="\tvoid add_"+str(symname)+"("+varstring+",int cutoff_type) {\n"
                    temp_code +="\t\t"+str(symname)+"SymFuns.push_back("+str(symname)+"SymmetryFunction("+argstring+",cutoff_type));\n"
                    temp_code +="\t};\n"
                    #add to previous
                    cpp_code=cpp_code[:start_add_idx]+temp_code+cpp_code[start_add_idx:]
                else:
                    cpp_code +="\tvoid add_"+str(symname)+"(int cutoff_type) {\n"
                    cpp_code +="\t\t"+str(symname)+"SymFuns.push_back("+str(symname)+"SymmetryFunction(cutoff_type));\n"
                    cpp_code +="};\n"
                    
                try:
                    start_vec_idx=cpp_code.index("//Start of custom extern declaration\n")+len("//Start of custom extern declaration\n")
                except:
                    bracket_idx=[i.start() for i in re.finditer('}', cpp_code)]
                    symfunset_idx=cpp_code.index("return new SymmetryFunctionSet()")
                    for i,idx in enumerate(bracket_idx):
                        if idx > symfunset_idx:
                            start_vec_idx=idx+1
                            break
                temp_code ="\tvoid SymmetryFunctionSet_add_"+str(symname)+"(SymmetryFunctionSet* symFunSet, "\
                +varstring+",int cutoff_type){symFunSet->add_"+str(symname)+"("+argstring+",cutoff_type);}\n"
                #add to previous
                cpp_code=cpp_code[:start_vec_idx]+temp_code+cpp_code[start_vec_idx:]
                break
        file.write(cpp_code)
        file.close()
        
def addToSymmetryFunctionsHeader(symname_list, symfunc_list,variables_list,remove=True):
    
    with open("symmetryFunctions.h","r") as file:
        h_code=file.read()
        file.close()
        
    with open("symmetryFunctions.h","w+") as file:
        #find insert position
        try:
            start_idx=h_code.index("//Start of section for custom symmetry functions\n")\
            +len("//Start of section for custom symmetry functions\n")
            end_idx=h_code.index("//End of section for custom symmetry functions\n")
            print("symmetryFunctions header found insert point!")
        except:
            #in case file is manually modified
            semi_idx=[i.start() for i in re.finditer(';', h_code)]
            ang_symfun_idx=h_code.index("AngularSymmetryFunction")
            for i,idx in enumerate(semi_idx):
                if idx > ang_symfun_idx:
                    start_idx=semi_idx[i]+1
                    temp_code="//Start of section for custom symmetry functions\n"
                    temp_code+="//End of section for custom symmetry functions\n"
                    h_code=h_code[:start_idx]+temp_code+h_code[start_idx:]
                    start_idx=start_idx+len("//Start of section for custom symmetry functions\n")
                    end_idx=h_code.index("//End of section for custom symmetry functions\n")
                    break 

        #remove old custom add functions
        if remove:
            h_code=h_code[:start_idx]+h_code[end_idx:]
        #insert new code

        for symname,symfuns in zip(symname_list,symfunc_list): 
            for symfunc in symfuns:
                #construct code
                tmpsyms = numbered_symbols("tmp")
                symbols, simple = cse(symfunc, symbols=tmpsyms)
                symbolslist_str = map(lambda x:str(x), list(symfunc.atoms(Symbol)))
                symbolslist = map(lambda x:x, list(symfunc.atoms(Symbol)))
                zipped_list=zip(symbolslist_str,symbolslist)
                zipped_list.sort(key=lambda t:t[0])
                symbolslist_str,symbolslist=map(list,zip(*zipped_list))
                conststring=",".join( " double "+str(x) for x in symbolslist if not(is_variable(str(x),variables_list)))[1:]
                temp_code ="class "+str(symname)+"SymmetryFunction:: public Symmetry Function\n"
                temp_code +="{\n"
                for symbol in symbolslist:
                    if not(is_variable(str(symbol),variables_list)):
                        temp_code +="\tdouble "+str(symbol)+";\n"
                temp_code +="\tCutoffFunction *cutFun\n"
                temp_code +="\tpublic:\n"
                temp_code +="\t\t"+str(symname)+"SymmetryFunction("+conststring+");\n"
                temp_code +="\t\tdouble eval(double* r);\n"
                for i in range(0,len(variables_list)):
                        temp_code +="\t\tdouble derivative_"+str(variables_list[i])+"(double* r);\n"
                        
                temp_code +="};\n"
                #add to previous
                h_code=h_code[:start_idx]+temp_code+h_code[start_idx:]
            #write new file
        file.write(h_code)
        file.close()
    
def addToSymmetryFunctions(symname_list, symfunc_list,variables_list,remove=True):
    
    with open("symmetryFunctions.cpp","r") as file:
        cpp_code=file.read()
        file.close()
        
    with open("symmetryFunctions.cpp","w+") as file:
        #find insert position
        try:
            start_idx=cpp_code.index("//Start of section for custom symmetry functions\n")\
            +len("//Start of section for custom symmetry functions\n")
            end_idx=cpp_code.index("//End of section for custom symmetry functions\n")
            print("symmetryFunctions found insert point!")
        except:
            #in case file is manually modified
            semi_idx=[i.start() for i in re.finditer(';', cpp_code)]
            ang_symfun_idx=cpp_code.index("AngularSymmetryFunction")
            for i,idx in enumerate(semi_idx):
                if idx > ang_symfun_idx:
                    start_idx=semi_idx[i]+1
                    temp_code="//Start of section for custom symmetry functions\n"
                    temp_code+="//End of section for custom symmetry functions\n"
                    cpp_code=cpp_code[:start_idx]+temp_code+cpp_code[start_idx:]
                    start_idx=start_idx+len("//Start of section for custom symmetry functions\n")
                    end_idx=cpp_code.index("//End of section for custom symmetry functions\n")
                    break 

        #remove old custom add functions
        if remove:
            cpp_code=cpp_code[:start_idx]+cpp_code[end_idx:]
        #insert new code

        for symname,symfuns in zip(symname_list,symfunc_list): 
            for symfunc in symfuns:
                #construct code
                tmpsyms = numbered_symbols("tmp")
                symbols, simple = cse(symfunc, symbols=tmpsyms)
                symbolslist_str = map(lambda x:str(x), list(symfunc.atoms(Symbol)))
                symbolslist = map(lambda x:x, list(symfunc.atoms(Symbol)))
                zipped_list=zip(symbolslist_str,symbolslist)
                zipped_list.sort(key=lambda t:t[0])
                symbolslist_str,symbolslist=map(list,zip(*zipped_list))
                conststring_i=",".join( " double "+str(x)+"_i" for x in symbolslist if not(is_variable(str(x),variables_list)))[1:]
                cutstring="*".join( "cutFun->eval("+str(x)+")" for x in symbolslist if is_variable(x,variables_list) and "costheta" not in str(x))
                temp_code ="double "+str(symname)+"SymmetryFunction::"+str(symname)+"SymmetryFunction("+conststring_i+",int cutoff_type)\n"
                for symbol in symbolslist:
                    if not(is_variable(str(symbol),variables_list)):
                        temp_code +="\t"+str(symbol)+" = "+str(symbol)+"_i;\n"
                temp_code +="\tif (cutoff_type == 0)\n"
                temp_code +="\t\tcutFun = new CosCutoffFunction(cutoff);\n"
                temp_code +="\telse if(cutoff_type == 1)\n"
                temp_code +="\t\tcutFun = new TanhCutoffFunction(cutoff);\n"
                temp_code +="\telse\n"
                temp_code +="\t\tcutFun = 1;\n"
                temp_code +="};\n"
                for i in range(0,len(variables_list)+1):
                    if i==0:
                        temp_code +="double "+str(symname)+"SymmetryFunction::eval(double* r)\n"
                        fun=symfunc
                    else:
                        temp_code +="double "+str(symname)+"SymmetryFunction::derivative_"+str(variables_list[i-1])+"(double* r)\n"
                        fun=diff(symfunc,variables_list[i-1])
                        tmpsyms = numbered_symbols("tmp")
                        symbols, simple = cse(fun, symbols=tmpsyms)
                    temp_code +="{\n"
                    temp=""
                    for s in symbols:
                        temp +=  "\tdouble " +ccode(s[0]) + " = " + ccode(s[1]) + ";\n"
                    if i>0:
                        try:   
                            const=float(ccode(simple[0]))
                            funstr=str(const)
                        except:
                            funstr=str(ccode(simple[0]))
                    else:
                        funstr=str(ccode(simple[0]))
                        
                    if "Heaviside" in funstr:
                        #get argument of Heaviside function
                        heavi_idx=funstr.index("Heaviside")
                        arg_start=funstr.index("(",heavi_idx)
                        arg_end=funstr.index(")",heavi_idx)
                        arg=funstr[arg_start+1:arg_end]
                        temp += "\tif(("+ccode(arg)+") < 0) {\n"
                        temp += "\t\treturn = 0.0; }\n"
                        temp += "\telse{"
                        temp +=  "\t\treturn = " + funstr[:heavi_idx-1]+"*"+cutstring+";\n }"
                        
                    else:
                        if funstr!="0":
                            temp +=  "\treturn = " + funstr+";\n"
                        else:
                            temp +=  "\treturn = 0.0;\n"
                            
                    #replace variables with pointer
                    ct=0
                    for s in symbolslist:
                        if is_variable(str(s),variables_list):
                            temp=temp.replace(str(s),"r["+str(ct)+"]")
                            ct+=1
                    temp_code += temp
                    temp_code +="};\n"
    
                #add to previous
                cpp_code=cpp_code[:start_idx]+temp_code+cpp_code[start_idx:]
            #write new file
        file.write(cpp_code)
        file.close()
        
def sympyToCpp(symname_list,symfunc_list,variables_list,remove=True):
    
    if len(symname_list)==len(symfunc_list):
        addToSymmetryFunctionSet(symname_list, symfunc_list,variables_list,remove)
        addToSymmetryFunctions(symname_list, symfunc_list,variables_list,remove)
        addToSymmetryFunctionsHeader(symname_list, symfunc_list,variables_list,remove)
    else:
        print("Number of names "+str(len(symname_list))\
              +" does not match number of symmetry functions "+str(len(symname_list))+" specified!")
    
    

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
