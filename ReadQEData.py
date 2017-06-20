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

def get_ecut(my_file):
    
    try:
        start=my_file.index("ecut=")
        end=my_file.index("Ry",start)
        temp=my_file[start+5:end]
    except:
        temp=None
        
    return float(temp)

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

def read_ekin_temp_etot(my_file):
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
            e_kin.append(float(my_file[kin_start_idx[i]+1:kin_end_idx[i]]))
            
    temp=[]
    if len(temp_start_idx)==len(temp_end_idx):
        for i in range(0,len(temp_start_idx)):
            temp.append(float(my_file[temp_start_idx[i]+1:temp_end_idx[i]]))
    e_tot=[]
    if len(tot_start_idx)==len(tot_end_idx):
        for i in range(0,len(tot_start_idx)):
            e_tot.append(float(my_file[tot_start_idx[i]+1:tot_end_idx[i]]))
               
    return e_kin,temp,e_tot

def read_total_energy(my_file):
    
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
            values.append(float(my_file[start_idx[i]+1:end_idx[i]]))
               
    return values


def read_harris_foulkes_estimate(my_file):
    
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
            values.append(float(my_file[start_idx[i]+1:end_idx[i]]))
               
    return values
    
def read_scf_accuracy(my_file):
    
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
            values.append(float(my_file[start_idx[i]+1:end_idx[i]]))
               
    return values

def read_geometries(my_file):
        
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
    #get values
    if len(start_idx)==len(end_idx):
        for i in range(0,len(start_idx)):
            geom=[]
            values=my_file[start_idx[i]+1:end_idx[i]].split()
            for val in values:
                try:
                    num=float(val)
                    pos=pos+(num,)
                    if len(pos)==4:
                        geom.append(pos)
                except:
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
        
    def get_files(self,folder):
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".out")]:
                temp=open(os.path.join(dirpath, filename),"r").read()
                if 'JOB DONE' in temp:
                    self.files.append(temp)
        return 1
    
    def read_all_files(self):
        
        for this in self.files:
            e_kin,temperature,e_tot=read_ekin_temp_etot(this)
            self.e_kin+=e_kin
            self.temperature+=temperature
            self.e_tot+=e_tot
            self.geometries+=read_geometries(this)
            
        self.e_pot=np.subtract(self.e_tot,self.e_kin)
        print(self.geometries)
        
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
                        
        self.e_pot_rel=np.subtract(self.e_pot,e_cal)

class QE_SCF_Reader(object):

    def __init__(self):
        
        self.files=[]
        self.total_energies=[]
        self.harris_foulkes_energies=[]
        self.scf_accuracies=[]
        self.e_cutoffs=[]
        self.cpu_times=[]
        self.avg_cpu_times=[]
        
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
        
        for this in self.files:
            temp_tot.append(read_total_energy(this))
            temp_harris.append(read_harris_foulkes_estimate(this))
            temp_scf_a.append(read_scf_accuracy(this))
            temp_ecut.append(get_ecut(this))
            temp_cpu_t.append(read_cpu_time(this))
            temp_avg_t.append(calc_avg_cpu_time(temp_cpu_t[-1]))
            
        all_data=zip(temp_ecut,temp_tot,temp_harris,temp_scf_a,temp_cpu_t,temp_avg_t)
        all_data.sort(key=lambda t:t[0])
        self.e_cutoffs,self.total_energies,self.harris_foulkes_energies,self.scf_accuracies,self.cpu_times,self.avg_cpu_times = map(list,zip(*all_data))
            
        
        
        
        
            
        
        


        
    