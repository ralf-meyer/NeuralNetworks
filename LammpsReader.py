#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:30:28 2017

@author: afuchs
"""

import numpy as np
import random
from lammps import lammps
import re
from NeuralNetworks import DataSet

def give_atom_type(type_list,atoms):
    
    out=[]
    for a_type in type_list:
        out.append(atoms[a_type-1])
    return out

def print_xyz(arr_x,arr_y,arr_z):
    for i in range(0,len(arr_x)):
        print([arr_x[i],arr_y[i],arr_z[i]])
        
def check_forces(fx,fy,fz,thresh):
    Valid=True
    if not(np.alltrue(np.abs(fx)<thresh)):
        Valid=False
    if not(np.alltrue(np.abs(fy)<thresh)):
        Valid=False
    if not(np.alltrue(np.abs(fz)<thresh)):
        Valid=False      
    return Valid

def add_random(arr,r):
    
    for i in range(0,len(arr)):
        arr[i]+=random.uniform(-r,r)

    return arr

def create_random_geometry(N,r,types):
    
    x=list()
    y=list()
    z=list()
    a_type=list()
    for i,Natom in enumerate(N):
        for j in range(0,Natom):
            a_type.append(types[i])
            x.append(random.uniform(-r,r))
            y.append(random.uniform(-r,r))
            z.append(random.uniform(-r,r))
        
    return a_type,x,y,z
    
def create_geometries(N_geoms,N_atoms,r,types):
    
    geoms=list()
    
    for i in range(0,N_geoms):
        geoms.append(create_random_geometry(N_atoms,r,types))
        
    return geoms

def make_files(geoms):
    
    header="&CONTROL\n  calculation  = 'scf'\n  prefix       = 'geom_opt',\n  pseudo_dir   = '/usr/share/espresso/pseudo',\n outdir       = '/home/afuchs/.tempdir'\n/\n&SYSTEM\n ibrav     = 1,\n celldm(1) = 40\n  nat       = $natom$,\n  ntyp      = 2,\n  ecutwfc   = 25.D0,\n  ecutrho   = 100.D0,\n occupations='smearing', smearing='methfessel-paxton',degauss=0.01\n/\n&ELECTRONS\n  conv_thr    = 1.D-5,\n  mixing_beta = 0.15D0,\n  electron_maxstep = 250\n/\nATOMIC_SPECIES\nNi 58.69 Ni.pbe-n-kjpaw_psl.0.1.UPF\nAu 197.00 Au.pbe-dn-kjpaw_psl.0.1.UPF\nATOMIC_POSITIONS {angstrom}\n"
    natoms=len(geoms[0][0])
    header=header.replace("$natom$",str(natoms))
    for i,geom in enumerate(geoms):
        with open('NiAu_'+str(i)+".in",'w') as file:
            file.write(header)
            for k in range(0,len(geom[0])):
                line=str(geom[0][k])+" "+str(geom[1][k])+" "+str(geom[2][k])+" "+str(geom[3][k])
                file.writelines(line+'\n')
            file.write("K_POINTS Gamma")
            file.close()
        


def make_rand_lammps_geoms(N_geom):
    Geometries=[]
    ct=0
    tot_ct=0
    while(ct<N_geom or tot_ct>N_geom*100):
        lmp=lammps()
        with open('template.md') as file1:
            all_str=str(file1.read())#
            all_str=all_str.replace("999",str(random.randint(0,999999)))
            all_str=all_str.replace("888",str(random.randint(0,999999)))
            with open('temp_'+str(ct)+'.md','w') as file2:
                file2.write(all_str)
        file1.close()
        file2.close()
        
        file_name="temp_"+str(ct)+".md"
        print(file_name)
        lmp.file(file_name)
        
        xc = lmp.extract_variable("x","all",1)
        yc = lmp.extract_variable("y","all",1)
        zc = lmp.extract_variable("z","all",1)
        x = np.asarray(xc)
        y = np.asarray(yc)
        z = np.asarray(zc)
        
    
        a_type_int=np.asarray(lmp.extract_variable("type","all",1),dtype=np.int32)
        a_type=give_atom_type(a_type_int,["Ni","Au"])
        fx = np.asarray(lmp.extract_variable("fx","all",1))
        fy = np.asarray(lmp.extract_variable("fy","all",1))
        fz = np.asarray(lmp.extract_variable("fz","all",1))
        
        if check_forces(fx,fy,fz,100):
            ct=ct+1
            print(x[0])
            print(fx[0])
            Geometries.append([a_type,x,y,z])
            print(str(ct/N_geom)+" %")
            
    
    
        
    
        tot_ct=tot_ct+1
        
    print(tot_ct)
    
    
class LammpsReader(object):
    
    def __init__(self):
        self.geometries=[]
        self.file_name=""
        self.file=""
        self.atom_types=[]
        self.nr_atom_types=[]
        self.nr_atoms=13
        self.nr_samples=100
        
    
    def xyz_file_to_qe_inputs(self):
        print(self.file_name)
        ds=DataSet.DataSet()
        ds.read_lammps(self.file_name,"")
        everyN_th=int(len(ds.geometries)/self.nr_samples)
        if everyN_th==0:
            everyN_th=len(ds.geometries)
        for i,geom in enumerate(ds.geometries):
            if i%everyN_th ==0:
                atom_type=[]
                x=[]
                y=[]
                z=[]
                for element in geom:
                    atom_type.append(self.atom_types[int(element[0])-1])
                    x.append(str(element[1][0]))
                    y.append(str(element[1][1]))
                    z.append(str(element[1][2]))
                
                self.geometries.append([atom_type,x,y,z])
                
        make_files(self.geometries)
                
                
    def get_random_samples(self):
        
        self.geometries=create_geometries(self.nr_samples,self.nr_atom_types,2,atom_types)
        make_files(self.geometries)