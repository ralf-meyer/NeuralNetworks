#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:23:34 2017

@author: afuchs
"""
import re
import os 
import matplotlib.pyplot as plt
import numpy as np

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
    start_idx=[i.start() for i in re.finditer('total cpu time spent up to now is', my_file)]

    temp_idx=[j.start() for j in re.finditer('secs', my_file)]
    
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

def read_ekin_temp_etot(my_file,E_conv_factor):
    #find total energies in file
    kin_idx=[i.start() for i in re.finditer('kinetic energy', my_file)]
    eq_idx=[i.start() for i in re.finditer('=', my_file)]
    k_idx=[i.start() for i in re.finditer('K', my_file)]
    ry_idx=[j.start() for j in re.finditer('Ry', my_file)]
    kin_start_idx=[]
    kin_end_idx=[]
    temp_start_idx=[]
    temp_end_idx=[]
    tot_start_idx=[]
    tot_end_idx=[]
    #Match start and end indices
    for idx in kin_idx:
        kin_start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > idx)])
        kin_end_idx.append(ry_idx[next(x[0] for x in enumerate(ry_idx) if x[1] > idx)])
        temp_start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > kin_end_idx[-1])])
        temp_end_idx.append(k_idx[next(x[0] for x in enumerate(k_idx) if x[1] > idx)])
        tot_start_idx.append(eq_idx[next(x[0] for x in enumerate(eq_idx) if x[1] > temp_end_idx[-1])])
        tot_end_idx.append(ry_idx[next(x[0] for x in enumerate(ry_idx) if x[1] > temp_end_idx[-1])])
    #get values
    e_kin=[]

    if len(kin_start_idx)==len(kin_end_idx):
        for i in range(0,len(kin_start_idx)):
            try:
                e_kin.append(float(my_file[kin_start_idx[i]+1:kin_end_idx[i]])*E_conv_factor)
            except:
                print("Detected wrong value in file: "+str(my_file[kin_start_idx[i]+1:kin_end_idx[i]]))
            
    temp=[]
    if len(temp_start_idx)==len(temp_end_idx):
        for i in range(0,len(temp_start_idx)):
            try:
                temp.append(float(my_file[temp_start_idx[i]+1:temp_end_idx[i]]))
            except:
                print("Detected wrong value in file: "+str(my_file[temp_start_idx[i]+1:temp_end_idx[i]]))
    e_tot=[]
    if len(tot_start_idx)==len(tot_end_idx):
        for i in range(0,len(tot_start_idx)):
            try:
                e_tot.append(float(my_file[tot_start_idx[i]+1:tot_end_idx[i]])*E_conv_factor)
            except:
                print("Detected wrong value in file: "+str(my_file[tot_start_idx[i]+1:tot_end_idx[i]]))
               
    return e_kin,temp,e_tot

def read_total_energy(my_file,eq_idx,temp_idx,E_conv_factor):
    
    #find total energies in file
    tot_idx=[i.start() for i in re.finditer('total energy', my_file)]
    not_idx=[i.start() for i in re.finditer('total energy is the sum of the following terms', my_file)]
    eq_idx=[i.start() for i in re.finditer('=', my_file)]
    temp_idx=[j.start() for j in re.finditer('Ry', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('Harris-Foulkes estimate', my_file)]
    eq_idx=[i.start() for i in re.finditer('=', my_file)]
    temp_idx=[j.start() for j in re.finditer('Ry', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('one-electron contribution', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('hartree contribution', my_file)]
    eq_idx=[i.start() for i in re.finditer('=', my_file)]
    temp_idx=[j.start() for j in re.finditer('Ry', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('xc contribution', my_file)]
    eq_idx=[i.start() for i in re.finditer('=', my_file)]
    temp_idx=[j.start() for j in re.finditer('Ry', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('ewald contribution', my_file)]
    eq_idx=[i.start() for i in re.finditer('=', my_file)]
    temp_idx=[j.start() for j in re.finditer('Ry', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('one-center paw contrib', my_file)]
    eq_idx=[i.start() for i in re.finditer('=', my_file)]
    temp_idx=[j.start() for j in re.finditer('Ry', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('smearing contrib', my_file)]
    
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
    tot_idx=[i.start() for i in re.finditer('estimated scf accuracy', my_file)]
    eq_idx=[i.start() for i in re.finditer('<', my_file)]
    temp_idx=[j.start() for j in re.finditer('Ry', my_file)]
    
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

def is_numeric(string):
    try:
        f=float(string)
        out=True
    except:
        out=False
    return out

def read_geometry_scf(my_file,Geom_conv_factor):
    #get lattice constant for scaling
    a_idx=[i.start() for i in re.finditer('alat', my_file)]

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
        atom=(types[i],np.asarray(xyz))
        geometry.append(atom)

    return geometry
    
def read_geometries(my_file,Geom_conv_factor):
        
    #find scf accuracies in file
    pos_idx=[i.start() for i in re.finditer('ATOMIC_POSITIONS', my_file)]
    ket_idx=[i.start() for i in re.finditer("\)", my_file)]
    temp_idx=[i.start() for i in re.finditer('kinetic energy', my_file)]
    
    start_idx=[]
    end_idx=[]
    #Match start and end indices
    for idx in pos_idx:
        start_idx.append(ket_idx[next(x[0] for x in enumerate(ket_idx) if x[1] > idx)])
        end_idx.append(temp_idx[next(x[0] for x in enumerate(temp_idx) if x[1] > idx)])
    
    all_geometries=[]
    pos=None
    pos3d=[]
    #get values
    if len(start_idx)==len(end_idx):
        for i in range(0,len(start_idx)):
            geom=[]
            values=my_file[start_idx[i]+1:end_idx[i]].split()

            for val in values:
                try:
                    num=float(val)*Geom_conv_factor
                    #pos=pos+(num,)
                    if len(pos3d)<3:
                        pos3d.append(num)
                        if len(pos3d)==3:
                            pos=pos+(np.array(pos3d),)
                            geom.append(pos)
                except:
                    pos3d=[]
                    a_type=val
                    pos=(a_type,)
                    
            all_geometries.append(geom)
               
    return all_geometries

class QE_MD_Reader(object):

    def __init__(self):
        
        self.files=[]
        self.e_tot=[]
        self.e_pot=[]
        self.e_kin=[]
        self.e_pot_rel=[]
        self.temperature=[]
        self.geometries=[]
        self.Calibration=[] #list of tuples ,includes ("path to file",Nr of Atoms of this type)
        self.E_conv_factor=1
        self.Geom_conv_factor=1
        
    def get_files(self,folder):
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".out")]:
                temp=open(os.path.join(dirpath, filename),"r").read()
                if 'JOB DONE' in temp:
                    self.files.append(temp)
        return 1
    
    def read_all_files(self):
        
        for this in self.files:
            e_kin,temperature,e_tot=read_ekin_temp_etot(this,self.E_conv_factor)
            self.e_kin+=e_kin
            self.temperature+=temperature
            self.e_tot+=e_tot
            self.geometries+=read_geometries(this,self.Geom_conv_factor)
        if(len(self.e_kin)==len(self.e_tot)):
            self.e_pot=np.subtract(self.e_tot,self.e_kin)
        else:
            self.e_pot=e_tot
        
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
                        e_cal+=reader.total_energies[0][-1]*NrAtoms*self.E_conv_factor
                        
        self.e_pot_rel=np.subtract(self.e_pot,e_cal)

class QE_SCF_Reader(object):

    def __init__(self):
        
        self.files=[]
        self.total_energies=[]
        self.e_tot=[]
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
                        
        self.e_tot_rel=np.subtract(self.e_tot,e_cal)
        
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
        
        for this in self.files:
            eq_idx=[i.start() for i in re.finditer('=', this)]
            temp_idx=[j.start() for j in re.finditer('Ry',this)]
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
            
        QE_SCF_Reader.get_converged_energies(self)
        
        
        
            
        
        


        
    