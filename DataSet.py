import numpy as _np

class DataSet(object):
    
    def __init__(self):
        self.geometries = []
        self.energies = []
    
    def read_xyz(self, filename):
        pass
    
    def read_lammps(self, xyzfile, thermofile):
        with open(xyzfile, "r") as f_xyz:
            counter = 0
            for line in f_xyz:
                if counter == 0: 
                    # New geometry, read number of atoms and skip the comment
                    # line
                    geo = []
                    counter = int(line)
                    next(f_xyz)
                    continue
                else:                
                    sp = line.split()
                    geo.append((sp[0],_np.array([float(sp[1]), 
                                float(sp[2]), float(sp[3])])))
                    counter -= 1
                    if counter == 0:
                        # Current geometry finished -> save to self.geometries
                        self.geometries.append(geo)
                        
        with open(thermofile, "r") as f_thermo:
            switch = False
            for line in f_thermo:
                if line.startswith("Step"):
                    ind = line.split().index("PotEng")
                    switch = True
                elif switch and line.startswith("Loop time"):
                    switch = False
                elif switch:
                    self.energies.append(line.split()[ind])