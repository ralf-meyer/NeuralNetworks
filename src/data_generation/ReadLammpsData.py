import re as _re
from os.path import isfile
import numpy as _np
from progressbar import ProgressBar


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
    def number_of_atoms_per_type(self):
        """number of atoms for each species (can also be
            set before information is read)"""
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
