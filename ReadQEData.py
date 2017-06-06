#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:23:34 2017

@author: afuchs
"""
import re
import os 
import matplotlib.pyplot as plt

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
                self.files.append(open(os.path.join(dirpath, filename),"r").read())
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
            
        
        
        
        
            
        
        


        
    