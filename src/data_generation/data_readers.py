#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""RadaReaders module

This module harbours classes used to parse the results of e.g. MD runs
with quantum espresso, LAMMPS, etc. used to generate training data for
the neural networks.

Todo:
    * throw out unessessary functions or group them in super class for all

Authors:
    * Alexander Fuchs, IEP/TU Graz, https://github.com/alexf1991
    * Johannes Cartus, IEP/TU Graz, https://github.com/jcartus
"""
import re as _re
import os 
from os.path import isfile
from abc import ABCMeta, abstractmethod
import numpy as _np
import matplotlib.pyplot as plt
from progressbar import ProgressBar

def get_nr_atoms_per_type(types,geometry):
    nr_atoms=_np.zeros((len(types))).astype(_np.int32)
    for atom in geometry:
        this_type=atom[0]
        for i in range(len(types)):
            if types[i]==this_type:
                nr_atoms[i]+=1#
    
    return list(nr_atoms)

def is_numeric(string):
    try:
        f=float(string)
        out=True
    except:
        out=False
    return out

def read_atom_types(my_file):
        
    types=[]
    searchstr='atomic species   valence    mass     pseudopotential'
    start_idx=my_file.index('\n',my_file.index(searchstr))+1
    run=True
    types_len=0
    ct=0
    while(run):
        end_idx=my_file.index('\n',start_idx)
        line=my_file[start_idx:end_idx].split(' ')
        start_idx=end_idx+1

        for element in line:
            if element !='':
                types.append(element)
                types_len=len(types)
                break
        ct+=1
        if types_len!= ct:
            run=False

        return types

def read_ekin_temp_etot_force(my_file,E_conv_factor,Geom_conv_factor):
    #find total energies in file
    ex_idx=[i.start() for i in _re.finditer('!', my_file)]
    kin_idx=[i.start() for i in _re.finditer('kinetic energy', my_file)]
    const_idx=[i.start() for i in _re.finditer('(const)', my_file)]
    f1_idx=[j.start() for j in _re.finditer('Forces acting on atoms', my_file)]
    f2_idx=[j.start() for j in _re.finditer('Total SCF correction', my_file)]
    tot_start_idx=[]
    tot_end_idx=[]
    e_tot=[]
    e_kin=[]
    temperature=[]

    # as qe plots forces Ry/au, to create the standard case (i.e. having 
    # eV/angstrom) as force unit include this factor
    au=0.529177249 #Angstroem
    force_factor=E_conv_factor*(Geom_conv_factor/au)


    #Match start and end indices
    #Read total energy
    error_ct=0
    
    print("Reding total energy:")
    bar_etot = ProgressBar()
    for i,idx in bar_etot(enumerate(ex_idx)):
        
        part=my_file[idx:f1_idx[i]]
        tot_start_idx=part.index('=')
        tot_end_idx=part.index('Ry')
        try:
            e_tot.append(float(part[tot_start_idx+1:tot_end_idx])*E_conv_factor)
        except:
            if error_ct==0:
                print("Detected wrong value in file: total energy = "+str(part[tot_start_idx+1:tot_end_idx]))
            error_ct+=1

    #Read forces 
    forces=[]
    f_i_clean=[]
    f_clean=[]
    error_ct=0
    

    
    print("Reading forces:")
    bar_forces = ProgressBar()  
    for j,idx in bar_forces(enumerate(f1_idx)):
        #if j %int(len(f1_idx)/3)==0:
        #    print("..."+str(34+float(j)*100*0.33/len(f1_idx))+"%")
        part=my_file[idx:f2_idx[j]]
        part_eq_idx=[i.start() for i in _re.finditer('=', part)]
        part_bs_idx=[j.start() for j in _re.finditer('\n',part)]
        part_f_idx=[j.start() for j in _re.finditer('force',part)]
        last_eq=0
        last_bs=0
        
        for part_idx in part_f_idx:
            f_i_start,last_eq= search_idx(part_idx,part_eq_idx,last_eq)
            f_i_start+=1
            f_i_end,last_bs= search_idx(part_idx,part_bs_idx,last_bs)

            try:
                if f_i_end!=None:
                    f_i=part[f_i_start:f_i_end].split(' ')
                else:
                    f_i=[""]
                    
                if 'Total' in f_i or f_i_end==None:
                    if len(f_i_clean)>0:
                        forces.append(f_clean)
                        f_clean=[]
                else:
                    f_i_clean=[]
    #                L=f_i!=''
    #                f_i_clean=f_i[L]
                    for fi_temp in f_i:
                        if fi_temp!='':
                            f_i_clean.append(float(fi_temp)*force_factor)
                    f_clean.append(f_i_clean)
                    
            except:
                if error_ct==0:
                    print("Detected wrong value in file: forces = "+str(my_file[f_i_start:f_i_end]))
                error_ct+=1
                break
                
    #Read kinetic energy and temperature
    error_ct=0
    print("Reading kinetic energy:")
    bar_kin_energy = ProgressBar()
    for i,idx in bar_kin_energy(enumerate(kin_idx)):
        

        part=my_file[idx:const_idx[i]]
        part_eq_idx=[i.start() for i in _re.finditer('=', part)]
        part_ry_idx=part.index("Ry")
        part_k_idx=part.index("K")
        try:
            e_kin.append(float(part[part_eq_idx[0]+1:part_ry_idx])*E_conv_factor)
        except:
            e_kin.append(0)
            if error_ct==0:
                print("Detected wrong value in file: kinetic energy = "+str(part[part_eq_idx[0]+1:part_ry_idx]))
            error_ct+=1
        try:
            temperature.append(float(part[part_eq_idx[1]+1:part_k_idx]))
        except:
            temperature.append(0)
            if error_ct==0:
                print("Detected wrong value in file: temperature = "+str(part[part_eq_idx[1]+1:part_k_idx]))
            error_ct+=1
    
    return e_kin,temperature,e_tot,forces

def read_geometries(my_file,Geom_conv_factor,atom_types):
        
    #find scf accuracies in file
    pos_idx=[i.start() for i in  _re.finditer('ATOMIC_POSITIONS', my_file)]
    ket_idx=[i.start() for i in  _re.finditer("\)", my_file)]
    temp_idx=[i.start() for i in  _re.finditer('kinetic energy', my_file)]
    
    start_idx=[]
    end_idx=[]
    #Match start and end indices
    last_ket=0
    last_temp=0
    for i,idx in enumerate(pos_idx):
        temp,last_ket=search_idx(idx,ket_idx,last_ket)
        start_idx.append(temp)
        temp,last_ket=search_idx(idx,temp_idx,last_temp)
        end_idx.append(temp)
    
    all_geometries=[]
    pos=None
    pos3d=[]
    iterate=0
    #get values
    if len(start_idx)==len(end_idx):

        bar = ProgressBar()

        for i in bar(range(len(start_idx))):
            geom=[]
            values=my_file[start_idx[i]+1:end_idx[i]].split()

            for val in values:
                try:
                    num=float(val)*Geom_conv_factor
                    #pos=pos+(num,)
                    if len(pos3d)<3:
                        pos3d.append(num)
                        if len(pos3d)==3:
                            pos=pos+(_np.array(pos3d),)
                            geom.append(pos)
                except:
                    pos3d=[]
                    if len(atom_types)==0:
                        a_type=val
                    else:
                        a_type=atom_types[iterate]
                        if iterate<len(atom_types)-1:
                            iterate+=1
                        else:
                            iterate=0
                        
                    pos=(a_type,)
                    
            all_geometries.append(geom)
            
    return all_geometries

def read_geometry_scf(my_file,Geom_conv_factor):
    #get lattice constant for scaling
    a_idx=[i.start() for i in  _re.finditer('alat', my_file)]

    a_idx_start=my_file.index("=",a_idx[0])+1
    a_idx_end=my_file.index("a",a_idx_start)
    a=float(my_file[a_idx_start:a_idx_end])
    #get postitions
    geom_start_idx=my_file.index("\n",a_idx[-2])
    geom_end_idx=my_file.index("number of k points")
    geom=my_file[geom_start_idx:geom_end_idx]
    
    lines=geom.split('\n')
    sites=[]
    types=[]
    x=[]
    y=[]
    z=[]
    eq_i=10e10
    for line in lines:
        parts=line.replace("\t"," ").split(" ")
        ct=0
        ct2=0
        for i,part in enumerate(parts):
            if len(part)>0:
                ct2+=1
                
            if ct2==1 and part!='':
                sites.append(part)
            elif ct2==2 and part!='':
                types.append(part)
                
            if "=" in part:
                eq_i=i
            elif is_numeric(part) and i>eq_i:
                if ct==0:
                    x.append(float(part)*Geom_conv_factor)
                elif ct==1:
                    y.append(float(part)*Geom_conv_factor)
                elif ct==2:
                    z.append(float(part)*Geom_conv_factor)
                ct=ct+1
    geometry=[]
    for i in range(len(x)):
        xyz=[x[i]*a,y[i]*a,z[i]*a]
        atom=(types[i],_np.asarray(xyz))
        geometry.append(atom)

    return geometry

def search_idx(idx,idx_dataset,last_idx):
    for i,x in enumerate(idx_dataset[last_idx:]):
        if x>idx:
            return x,i
    
    return None,None          


class Deprecated(object):
    def get_ecut(my_file,E_conv_factor):
        
        try:
            start=my_file.index("ecut=")
            end=my_file.index("Ry",start)
            temp=my_file[start+5:end]
            return float(temp)*E_conv_factor
        except:
            temp=None
            
        return temp

    def read_cpu_time(my_file):
        
        #find cpu time in file
        start_idx=[i.start() for i in _re.finditer('total cpu time spent up to now is', my_file)]

        temp_idx=[j.start() for j in _re.finditer('secs', my_file)]
        
        end_idx=[]
        #Match start and end indices
        for idx in start_idx:
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+len('total cpu time spent up to now is'):end_idx[i]]))

        return values

    def calc_avg_cpu_time(times):
        
        avg=times[0]
        for i in range(1,len(times)):
            avg=(avg+(times[i]-times[i-1]))/2
            
        return avg

    def read_total_energy(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find total energies in file
        tot_idx=[i.start() for i in  _re.finditer('total energy', my_file)]
        not_idx=[i.start() for i in  _re.finditer('total energy is the sum of the following terms', my_file)]
        eq_idx=[i.start() for i in  _re.finditer('=', my_file)]
        temp_idx=[j.start() for j in  _re.finditer('Ry', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            if idx not in not_idx:
                start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
                end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        return values

    def read_harris_foulkes_estimate(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find harris foulkes estimates in file
        tot_idx=[i.start() for i in  _re.finditer('Harris-Foulkes estimate', my_file)]
        eq_idx=[i.start() for i in  _re.finditer('=', my_file)]
        temp_idx=[j.start() for j in  _re.finditer('Ry', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        return values
        
    def read_one_electron_contrib(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find harris foulkes estimates in file
        tot_idx=[i.start() for i in  _re.finditer('one-electron contribution', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
        
        if len(values)>1 or len(values)==0:
            return values
        else:
            return values[0]

    def read_hartree_contribution(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find harris foulkes estimates in file
        tot_idx=[i.start() for i in  _re.finditer('hartree contribution', my_file)]
        eq_idx=[i.start() for i in  _re.finditer('=', my_file)]
        temp_idx=[j.start() for j in  _re.finditer('Ry', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        if len(values)>1 or len(values)==0:
            return values
        else:
            return values[0]

    def read_xc_contribution(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find harris foulkes estimates in file
        tot_idx=[i.start() for i in  _re.finditer('xc contribution', my_file)]
        eq_idx=[i.start() for i in  _re.finditer('=', my_file)]
        temp_idx=[j.start() for j in  _re.finditer('Ry', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        if len(values)>1 or len(values)==0:
            return values
        else:
            return values[0]

    def read_ewald_contribution(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find harris foulkes estimates in file
        tot_idx=[i.start() for i in  _re.finditer('ewald contribution', my_file)]
        eq_idx=[i.start() for i in  _re.finditer('=', my_file)]
        temp_idx=[j.start() for j in  _re.finditer('Ry', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        if len(values)>1 or len(values)==0:
            return values
        else:
            return values[0]
    
    def read_one_center_paw_contrib(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find harris foulkes estimates in file
        tot_idx=[i.start() for i in  _re.finditer('one-center paw contrib', my_file)]
        eq_idx=[i.start() for i in  _re.finditer('=', my_file)]
        temp_idx=[j.start() for j in  _re.finditer('Ry', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        if len(values)>1 or len(values)==0:
            return values
        else:
            return values[0]

    def read_smearing_contrib(my_file,eq_idx,temp_idx,E_conv_factor):
        
        #find harris foulkes estimates in file
        tot_idx=[i.start() for i in  _re.finditer('smearing contrib', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        if len(values)>1 or len(values)==0:
            return values
        else:
            return values[0]

    def read_scf_accuracy(my_file,E_conv_factor):
        
        #find scf accuracies in file
        tot_idx=[i.start() for i in  _re.finditer('estimated scf accuracy', my_file)]
        eq_idx=[i.start() for i in  _re.finditer('<', my_file)]
        temp_idx=[j.start() for j in  _re.finditer('Ry', my_file)]
        
        start_idx=[]
        end_idx=[]
        #Match start and end indices
        for idx in tot_idx:
            start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
            end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
            
        #get values
        values=[]
        if len(start_idx)==len(end_idx):
            for i in range(0,len(start_idx)):
                values.append(float(my_file[start_idx[i]+1:end_idx[i]])*E_conv_factor)
                
        return values


class QE_MD_Reader(object):

    def __init__(self):
        
        self.atom_types=[]
        self.files=[]
        self.e_tot=[]
        self.e_pot=[]
        self.e_kin=[]
        self.forces=[]
        self.energies=[]
        self.temperature=[]
        self.geometries=[]
        self.Calibration=[] #list of tuples ,includes ("path to file",Nr of Atoms of this type)
        self.E_conv_factor=1 # 1 if calcualtions are in ev
        self.Geom_conv_factor=1 #1 if calculations are in Angstroem
        self.nr_atoms_per_type=[]
        
    def get_files(self,folder):
        if ".out" in folder:
            temp=open(folder,"r").read()
            if 'JOB DONE' in temp:
                self.files.append(temp)
        else:
            for dirpath, dirnames, filenames in os.walk(folder):
                for filename in [f for f in filenames if f.endswith(".out")]:
                    temp=open(os.path.join(dirpath, filename),"r").read()
                    if 'JOB DONE' in temp:
                        self.files.append(temp)
        return 1
    
    def read_all_files(self):
        
        if len(self.files)>0 and len(self.atom_types)==0:
            self.atom_types=read_atom_types(self.files[0])
        else:
            print("No files loaded!")

        for ct, this in enumerate(self.files):
            print("Reading energies and forces in file "+str(ct)+"...")
            e_kin,temperature,e_tot,forces=read_ekin_temp_etot_force(
                this,
                self.E_conv_factor,
                self.Geom_conv_factor
            )
            self.forces+=forces
            self.e_kin+=e_kin
            self.temperature+=temperature
            self.e_tot+=e_tot
            print("Reading geometries in file "+str(ct)+"...")

            # read starting geometry, and convert it from lattice unit to angstr.
            starting_geometry=read_geometry_scf(this,self.Geom_conv_factor) 
            self.geometries=[starting_geometry]

            self.geometries+=read_geometries(
                this, 
                self.Geom_conv_factor, 
                self.atom_types
            )

            # remove last geometry because no forces and energies are available
            self.geometries.pop(-1)

        if(len(self.e_kin)==len(self.e_tot)):
            self.e_pot=_np.subtract(self.e_tot,self.e_kin)
        else:
            self.e_pot=e_tot
        self.nr_atoms_per_type=get_nr_atoms_per_type(self.atom_types,self.geometries[0])
        
    def calibrate_energy(self):
        reader=QE_SCF_Reader()
        e_cal=0
        if len(self.Calibration)>0: #With calibration files it gives the total energy minus the single atom energies
            for cal in self.Calibration:
                reader.files=[]
                folder=cal[0]
                NrAtoms=cal[1]
                for dirpath, dirnames, filenames in os.walk(folder):
                    for filename in [f for f in filenames if f.endswith(".out")]:
                        temp=open(os.path.join(dirpath, filename),"r").read()
                        if 'JOB DONE' in temp:
                            reader.files=[temp]
                            reader.read_all_files()
                            e_cal+=reader.total_energies[0][-1]*NrAtoms*self.E_conv_factor
        else: #Sets minimum as zero point 
            e_cal=min(self.e_pot)
                        
        self.energies=_np.subtract(self.e_pot,e_cal)
        
class QE_SCF_Reader(object):

    def __init__(self):
        
        self.files=[]
        self.total_energies=[]
        self.e_tot=[]
        self.atom_types=[]
        self.harris_foulkes_energies=[]
        self.scf_accuracies=[]
        self.e_cutoffs=[]
        self.cpu_times=[]
        self.avg_cpu_times=[]
        self.one_e_contrib=[]
        self.hartree_contrib=[]
        self.xc_contrib=[]
        self.ewald_contrib=[]
        self.one_center_paw_contrib=[]
        self.smearing_contrib=[]
        self.e_tot_rel=[]
        self.Calibration=[]
        self.geometries=[]
        self.E_conv_factor=1
        self.Geom_conv_factor=1
        
    def calibrate_energy(self):
        reader=QE_SCF_Reader()
        e_cal=0
        for cal in self.Calibration:
            reader.files=[]
            folder=cal[0]
            NrAtoms=cal[1]
            for dirpath, dirnames, filenames in os.walk(folder):
                for filename in [f for f in filenames if f.endswith(".out")]:
                    temp=open(os.path.join(dirpath, filename),"r").read()
                    if 'JOB DONE' in temp:
                        reader.files=[temp]
                        reader.read_all_files()
                        e_cal+=reader.total_energies[0][-1]*NrAtoms
                        
        self.e_tot_rel=_np.subtract(self.e_tot,e_cal)
        
    def get_converged_energies(self):
        for es in self.total_energies:
            self.e_tot.append(es[-1])

        
    def get_files(self,folder):
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".out")]:
                temp=open(os.path.join(dirpath, filename),"r").read()
                if 'JOB DONE' in temp:
                    self.files.append(temp)
        return 1
    
    def read_all_files(self):
        
        temp_tot=[]
        temp_harris=[]
        temp_scf_a=[]
        temp_cpu_t=[]
        temp_ecut=[]
        temp_avg_t=[]
        temp_one_e=[]
        temp_hartree=[]
        temp_xc=[]
        temp_ewald=[]
        temp_one_center=[]
        temp_smearing=[]
        temp_geoms=[]
        if len(self.files)>0:
            self.atom_types=read_atom_types(self.files[0])
        else:
            print("No files loaded!")
        for this in self.files:
            eq_idx=[i.start() for i in  _re.finditer('=', this)]
            temp_idx=[j.start() for j in  _re.finditer('Ry',this)]
            temp_tot.append(read_total_energy(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_harris.append(read_harris_foulkes_estimate(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_scf_a.append(read_scf_accuracy(this,self.E_conv_factor))
            temp_ecut.append(get_ecut(this,self.E_conv_factor))
            temp_cpu_t.append(read_cpu_time(this))
            temp_avg_t.append(calc_avg_cpu_time(temp_cpu_t[-1]))
            temp_one_e.append(read_one_electron_contrib(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_hartree.append(read_hartree_contribution(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_xc.append(read_xc_contribution(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_ewald.append(read_ewald_contribution(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_one_center.append(read_one_center_paw_contrib(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_smearing.append(read_smearing_contrib(this,eq_idx,temp_idx,self.E_conv_factor))
            temp_geoms.append(read_geometry_scf(this,self.Geom_conv_factor))
            
        all_data=zip(temp_ecut,temp_tot,temp_harris,temp_scf_a,temp_cpu_t,\
                     temp_avg_t,temp_one_e,temp_hartree,temp_xc,temp_ewald,\
                     temp_one_center,temp_smearing,temp_geoms)
        all_data.sort(key=lambda t:t[0])
        self.e_cutoffs,self.total_energies,self.harris_foulkes_energies,\
        self.scf_accuracies,self.cpu_times,self.avg_cpu_times,\
        self.one_e_contrib,self.hartree_contrib,self.xc_contrib,\
        self.ewald_contrib,self.one_center_paw_contrib,\
        self.smearing_contrib,self.geometries= map(list,zip(*all_data))
        self.nr_atoms_per_type=get_nr_atoms_per_type(self.atom_types,self.geometries[0])    
        QE_SCF_Reader.get_converged_energies(self)
           
class LammpsReader(object):
    """This class fetches results from the dump/thermo/xyzfile from a 
    LAMMPS calculation.  

    Attributes:
        geometries: list of list of tuples, (atom species, atomic positions: xyz)
        forces: list of list of np array containig the forces fx,fy,fz
        energies: list of the energies (double)

        E_conv_factor = conversion factor from unit energies should be
            interpreted in to eV.
        Geom_conv_factor = conversion factor from the unit the goemetries
            should be read in and Angstroem.
    """

    def __init__(self):

        self.geometries = []
        self.energies = []
        self.forces = []
    
        #internal cnoversion factors
        self.E_conv_factor = 1
        self.Geom_conv_factor = 1 

        # for internal handling of atom_types
        self._species = [] 

        
    #--- getter/setter for atomic species ---
    @property
    def atom_types(self):
        """ the atomic species occurring in the measurement (can also be
            set before information is read)."""
        return list(set(self._species))

    @atom_types.setter
    def atom_types(self, value):
        #assume count 1 for all atom types until set differently
        self._species = value
    #---

    #--- getter/setter for counts of atomic species ---
    @property
    def nr_atoms_per_type(self):
        """number of atoms for each species (can also be
            set before information is read)"""
        return [self._species.count(x) for x in set(self._species)]

    @nr_atoms_per_type.setter
    def nr_atoms_per_type(self, value):
        if len(self._species) == 0:
            msg = "Atomic species not set! Must set them prior to stating count..."
            raise ValueError(msg)

        # create list of combinations
        self._species = [[x] * y for x, y in zip(self._species, value)]
        # flatten out to simple list
        self._species = \
            [species for count_and_species in self._species for species in count_and_species ]
    #---

    
    def read_lammps(self, dumpfile, thermofile, xyzfile=""):
        """Extracts data like atom types. geometries and forces from LAMPS
        result files (thermo file, custom dum and xyz-files).
        It will try to use the dump file first to find geometries and forces.
        If the dump if not found the geometries will be read from xyz-file.
        The energy is currently read from xyz file. 
        
        If unit conversion factors are set they will be applied automatically.
        Further on, atoms will be labeled automatically if atom types were set.

        Args: 
            dumpfile: path to a custom dump file in the following format:
                TODO: specify format of dump file here.
            thermofile: path to the .log file output by LAMPS.
            xyzfile (optional): path to .xyz-file output by LAMPS. It is
            only attempted to be used if dump file is not found.
        """

        # read geometries and forces and species from dump file or xyz file
        if isfile(dumpfile):
            self._read_from_dump(dumpfile)
        else:
            print("Dump file not found at {0}.\n".format(dumpfile))

            if xyzfile != "":

                print("Trying xyz file ...")

                if isfile(xyzfile):
                    self._read_geometries_from_xyz(xyzfile)
                else:
                    print("XYZ file not found at {0}.\n".format(xyzfile))

        # read energies and potentials
        if isfile(thermofile):
            self._read_energies_from_thermofile(thermofile)
        else:
            print("Invalid file path: {0} is not a file!".format(thermofile))
            
        

    def _read_from_dump(self, dumpfile):
        """reads species, geometries and forces from dump file.
        If species are not set yet (i.e. if self.species is empty) the atom names
        are recovered from the file too.
        
        Args:
            dumpfile: path to file
        """

        # flag whether species where given (if false then it will be read)
        species_unknown = len(self._species) == 0

        try:
            with open(dumpfile) as f_dump:
                
                file_contents = f_dump.read()

                #--- find start and end positions of lines of interest ---
                searchstring_start = "ITEM: ATOMS element x y z fx fy fz"
                searchstring_end = "ITEM: TIMESTEP"
            
                start_index = [i.start() for i in _re.finditer(
                    searchstring_start, 
                    file_contents)]
                end_index = [i.start() - 1 for i in _re.finditer(
                    searchstring_end, 
                    file_contents)]
                end_index.pop(0)

                # add final end token if necessary
                if len(start_index) == len(end_index) + 1:
                    end_index.append(len(file_contents))

                if len(start_index) != len(end_index):
                    msg = "Numer of start and end points does not mathch!" + \
                    "Possibly broken dump file {0}".format(dumpfile)
                    raise UserWarning(msg)
                #---

                bar = ProgressBar()
                print("Reading dump file ...")

                # loop over time steps
                for i in bar(range(min([len(start_index), len(end_index)]))):
        
                   # remove trailing EOL marker to avoid empty line at the end
                    section = \
                        file_contents[start_index[i]:end_index[i]].rstrip("\n")
                    section = section.split("\n")

                    # first line is just header
                    section.pop(0)

                    # buffers for logging geometries/forces for the current time step
                    geometries_current_step = []
                    forces_current_step = []

                    # loop over atom entries
                    for line_index, line in enumerate(section):

                        line_splits = line.split()

                        # log species if not known yet
                        if species_unknown:
                            self._species.append(line_splits[0])
                        
                        #--- parse + log positions/forces, do unit conversion---
                        geometries_current_step.append(
                            (self._species[line_index], 
                            _np.array(map(float, line_splits[1:4])) \
                            * self.Geom_conv_factor
                            )
                        )
                        forces_current_step.append(
                            _np.array(map(float, line_splits[4:7]) \
                            * (self.E_conv_factor / self.Geom_conv_factor)
                            )
                        )
                        #---

                    # put results of current time step in overall list
                    self.geometries.append(geometries_current_step)
                    self.forces.append(forces_current_step)

                    # toggle flag is information on species was acquired
                    if species_unknown:
                        species_unknown = False
                        
        except IOError as e:
            print("File could not be read! {0}".format(e.errno))
        except Exception as e:
            msg = "An unknown error occurred " + \
                "during parsing of dump file: {0}".format(e.message)
            print(msg)

    def _read_geometries_from_xyz(self, xyzfile):
        
        # check if species already set
        species_unknown = len(self._species) == 0
        number_of_atoms_total = None if species_unknown else len(self._species)

        try:
            with open(xyzfile, "r") as f_xyz:
                counter = -1

                print("Reading xyz file ...")

                for line in f_xyz:
                    # read number of atoms if not known yet
                    if number_of_atoms_total is None:
                        number_of_atoms_total = int(line)

                    if counter == -1: 
                        # New geometry, read number of atoms and skip the comment
                    
                        geo = []
                        counter = 0

                        next(f_xyz)
                        continue

                    else: 
                        # read geometries
                        sp = line.split()

                        if species_unknown:
                            self._species.append(sp[0])
                        
                        geo.append(
                            (self._species[counter],
                            _np.array(map(float, sp[1:4])) \
                            * self.Geom_conv_factor)
                        )

                        counter += 1

                        if counter == number_of_atoms_total:
                            # Current geometry finished -> save to self.geometries
                            self.geometries.append(geo)
                            counter = -1

                            # toggle flag to look for species
                            if species_unknown:
                                species_unknown = False
    
        except Exception as e:
            print("Error reading xyz file: {0}".format(e.message))
        
    def _read_energies_from_thermofile(self, thermofile):
        try:
            with open(thermofile, "r") as f_thermo:
                switch = False

                
                print("Reading thermo file ...")

                for line in f_thermo:
                    if line.startswith("Step"):
                        ind = line.split().index("PotEng")
                        switch = True
                    elif switch and line.startswith("Loop time"):
                        switch = False
                    elif switch:
                        self.energies.append(
                            float(line.split()[ind]) \
                            * self.E_conv_factor
                        )
        except Exception as e:
            print("Error reading thermodynamics file: {0}".format(e.message))
