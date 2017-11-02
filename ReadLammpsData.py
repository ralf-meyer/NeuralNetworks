import re as _re
from os.path import isfile
import numpy as _np

class LammpsReader(object): 

    def __init__(self):

        self.geometries = []
        self.energies = []
        self.forces = []
    
        #self.nr_atoms_per_type=[]
        #self.atom_types=[]
        self._species = [] # for internal handling of atom_types

        #--- set internal cnoversion factors ---
        # 1 if calcualtions are in ev
        self._energy_conversion_factor = 1

        #1 if calculations are in Angstroem
        self._geometry_conversion_factor = 1 

        # depends of energy and geometry conv. factors
        self._force_conversion_factor = 1
        #---
        
    #--- getter/setter conversion factors ---
    @property
    def E_conv_factor(self):
        return self._energy_conversion_factor
    @E_conv_factor.setter
    def E_conv_factor(self, value):
        # set new value for energy
        self._energy_conversion_factor = value
        # rescale factor for force
        self._force_conversion_factor *= value

    @property
    def Geom_conv_factor(self):
        return self._geometry_conversion_factor
    @Geom_conv_factor.setter
    def Geom_conv_factor(self, value):
        # set new value for energy
        self._geometry_conversion_factor = value
        # rescale factor for force
        self._force_conversion_factor /= value
    #---

    #--- getter/setter for atomic species ---
    @property
    def atom_types(self):
        return list(set(self._species))

    @atom_types.setter
    def atom_types(self, value):
        #assume count 1 for all atom types until set differently
        self._species = value
    #---

    #--- getter/setter for counts of atomic species ---
    @property
    def number_of_atoms_per_type(self):
        return [self._species.count(x) for x in set(self._species)]

    @number_of_atoms_per_type.setter
    def number_of_atoms_per_type(self, value):
        if len(self._species) == 0:
            msg = "Atomic species not set! Must set them prior to stating count..."
            raise ValueError(msg)

        # create list of combinations
        self._species = [[x] * y for x, y in zip(self._species, value)]
        # flatten out to simple list
        self._species = \
            [species for count_and_species in self._species for species in count_and_species ]
    #---

    
    def read_lammps(self, dumpfile, xyzfile, thermofile):
        """Extracts data like atom types. geometries and forces from LAMPS
        result files (thermo file, custom dum and xyz-files).
        It will try to use the dump file first to find geometries and forces.
        If the dump if not found the geometries will be read from xyz-file.
        The energy is currently read from xyz file. 
        
        If unit conversion factors are set they will be applied automatically.
        Further on, atoms will be labeled automatically if atom types were set.

        Args: 
            xyzfile: path to .xyz-file output by LAMPS.
            thermofile: path to the .log file output by LAMPS.
            dumpfile: path to a custom dump file in the following format:
                TODO: specify format of dump file here.
        """

        # read geometries and forces and species from dump file
        if isfile(dumpfile):
            self._read_from_dump(dumpfile)
        else:
            msg = "Dump file not found at {0}.\n".format(dumpfile)
            msg += "Trying xyz file ..."
            print(msg)

            if isfile(xyzfile):
                # try to acquire species and geometries from xyzfile
                self._read_geometries_from_xyz(xyzfile)
            else:
                msg = "XYZ file not found at {0}.\n".format(xyzfile)
                raise ValueError("Neither dump nor xyz file given!")

        # read energies and potentials
        if isfile(thermofile):
            self._read_energies_from_thermofile(thermofile)
        else:
            msg = "Invalid file path: {0} is not a file!".format(dumpfile)
            raise ValueError(msg)
        

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

                # loop over time steps
                for i in range(min([len(start_index), len(end_index)])):
                    section = file_contents[start_index[i]:end_index[i]].split("\n")

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
                            * self._geometry_conversion_factor
                            )
                        )
                        forces_current_step.append(
                            _np.array(map(float, line_splits[4:7]) \
                            * self._force_conversion_factor
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
            msg = "An unknown error occurred" + \
                "during parsing of dump file: {0}".format(e.message)
            print(msg)

    def _read_geometries_from_xyz(self, xyzfile):
        
        # check if species already set
        species_unknown = len(self._species) == 0
        number_of_atoms_total = None if species_unknown else len(self._species)

        try:
            with open(xyzfile, "r") as f_xyz:
                counter = -1

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
                            * self._geometry_conversion_factor)
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
                for line in f_thermo:
                    if line.startswith("Step"):
                        ind = line.split().index("PotEng")
                        switch = True
                    elif switch and line.startswith("Loop time"):
                        switch = False
                    elif switch:
                        self.energies.append(
                            float(line.split()[ind]) \
                            * self._energy_conversion_factor
                        )
        except Exception as e:
            print("Error reading thermodynamics file: {0}".format(e.message))
