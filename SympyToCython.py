
from sympy import Symbol, cse, ccode
from sympy.utilities import numbered_symbols


# Inspired by http://stackoverflow.com/questions/22665990/optimize-code-generated-by-sympy
def sympyToCython(symname_list, symfunc_list):

    pxd_code=""
    cy_code = "#!python\n"
    cy_code += "#cython: boundscheck=False, wraparound=False, cdivision=True\n"
    cy_code += "cdef extern from \"math.h\":\n"
    cy_code += "    double cos(double m)\n"
    cy_code += "    double sin(double m)\n"
    cy_code += "    double exp(double m)\n"
    cy_code += "    double tanh(double m)\n"
    cy_code += "    double cosh(double m)\n"
    cy_code += "    double sqrt(double m)\n"
    cy_code += "    double M_PI\n\n"
    
    for i in range(len(symfunc_list)):
        symfunc= symfunc_list[i]
        symname= symname_list[i]
        if type(symfunc)!=int and type(symfunc)!=float:
            tmpsyms = numbered_symbols("tmp")
            symbols, simple = cse(symfunc, symbols=tmpsyms)
            symbolslist = map(lambda x:str(x), list(symfunc.atoms(Symbol)) )
            symbolslist.sort()
            #Create argument of function
            varstring=",".join( " double "+x for x in symbolslist )[1:]
            #Function header
            cy_code +="cdef double "+str(symname)+"("+varstring+"):\n"
            cy_code += "    cdef double r = 0\n"
            for x in symbols:
                cy_code += "    cdef "+str(x[0])+" = "+str(x[1])+"\n" 
            pxd_code+= "cdef double "+str(symname)+"("+varstring+")\n"
            mystr=str(symfunc)
            if "Heaviside" in mystr:
                #get argument of Heaviside function
                heavi_idx=mystr.index("Heaviside")
                arg_start=mystr.index("(",heavi_idx)
                arg_end=mystr.index(")",heavi_idx)
                arg=mystr[arg_start+1:arg_end]
                cy_code += "    if(("+ccode(arg)+") < 0):\n"
                cy_code += "        r = 0\n"
                cy_code += "    else:"
                cy_code += "        r = " + ccode(mystr[:heavi_idx-1])+"\n"
            else:
                cy_code += "    r = " + ccode(simple[0])+"\n"
            cy_code +=  "    return r\n"
            cy_code += "\n\n"
        else:
            cy_code += "cdef double "+str(symname)+"():\n"
            cy_code +=  "    double r\n"
            cy_code +=  "    r = " +str(symfunc) +"\n"
            cy_code +=  "    return r;\n"
            cy_code += "\n\n"


    return cy_code,pxd_code
    
