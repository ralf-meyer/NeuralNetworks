import unittest
from os.path import normpath, dirname, join
import numpy as _np
from NeuralNetworks import ReadLammpsData
#from ReadLammpsData import LammpsReader
#import sys
#sys.path.append(normpath("../ReadLammpsData"))
#from ..ReadLammpsData import LammpsReader

class TestLammpsReader(unittest.TestCase):
    """This class containts tests for ReadLammpsData.py"""

    def setUp(self):
        # set paths to result files to read from
        test_files_dir = join(dirname(__file__), normpath("TestData/Lammps"))
        self._dump_path = join(test_files_dir, "Au_md.dump")
        self._xyz_path = join(test_files_dir, "Au_md.xyz")
        self._thermo_path = join(test_files_dir, "Au.log")

        #--- set reference values ---
        # species
        self._expected_atom_types = ["Au"]
        self._expected_number_of_atoms_per_type = [26]

        # geometries
        self._expected_geometry_first_step_third_atom = \
            ("Au", _np.array([-7.06365, 1.76129, -0.29539]))
        self._expected_geometry_third_step_eleventh_atom = \
            ("Au", _np.array([-3.26698, -2.05631, 1.26889]))
        self._expected_geometry_sixth_step_twenty_sixth_atom = \
            ("Au", _np.array([-2.07949, 0.0551922, 5.46916]))

        # forces
        self._expected_force_first_step_first_atom = \
            _np.array([10.1944, -0.782955, 2.71506])
        self._expected_force_fourth_step_twenty_first_atom = \
            _np.array([2.04384, -9.01468, 15.4655])
        self._expected_force_sixth_step_eighth_atom = \
            _np.array([0.783606, 0.30807, 1.6233])

        # energies
        self._expected_energies = \
            [-62.118812, -64.454685, -66.708751, -68.5237, -69.796816, -70.589006]

    def _check_species(self, actual_atom_types, actual_number_of_atoms_per_type):
        # test species detection
        self.assertItemsEqual(actual_atom_types, self._expected_atom_types)
        self.assertItemsEqual(
            actual_number_of_atoms_per_type,
            self._expected_number_of_atoms_per_type
        )

    def _check_geometry(self, actual_geometry):
        """Checks geometry entries against stored references"""
        self._assert_geometry_entries_equals(
            actual_geometry[0][2],
            self._expected_geometry_first_step_third_atom
        )
        self._assert_geometry_entries_equals(
            actual_geometry[2][10],
            self._expected_geometry_third_step_eleventh_atom
        )
        self._assert_geometry_entries_equals(
            actual_geometry[5][25],
            self._expected_geometry_sixth_step_twenty_sixth_atom
        )
    
    def _assert_geometry_entries_equals(self, expected, actual):
        """Compares objects of the specific tuple formart (string, np.array)"""

        # compare atomic species
        self.assertEqual(expected[0], actual[0])

        # compare geometry values
        self.assertItemsEqual(expected[1], actual[1])

    def _check_forces(self, actual_forces):
        self.assertItemsEqual(
            actual_forces[0][0],
            self._expected_force_first_step_first_atom
        )
        self.assertItemsEqual(
            actual_forces[3][20],
            self._expected_force_fourth_step_twenty_first_atom
        )
        self.assertItemsEqual(
            actual_forces[5][7],
            self._expected_force_sixth_step_eighth_atom
        )

    def test_read_dump(self):
        """Read from dump (in test data) and compare against hard coded string"""

        reader = ReadLammpsData.LammpsReader()

        # read from dump file
        reader.read_lammps(self._dump_path, self._xyz_path, self._thermo_path)

        # test species
        self._check_species(reader.atom_types, reader.number_of_atoms_per_type)

        # test geometries
        self._check_geometry(reader.geometries)

        # test forces
        self._check_forces(reader.forces)

    def test_read_thermo(self):
        """Read energies from thermo file and compare to hard coded reference"""
        
        reader = ReadLammpsData.LammpsReader()        
        reader._read_energies_from_thermofile(self._thermo_path)

        self.assertItemsEqual(self._expected_energies, reader.energies)

    def test_read_xyz(self):
        """test reading geometries and species from xyz file"""

        reader = ReadLammpsData.LammpsReader()
        reader._read_geometries_from_xyz(self._xyz_path)

        self._check_species(reader.atom_types, reader.number_of_atoms_per_type)
        self._check_geometry(reader.geometries)


        

if __name__ == '__main__':
    unittest.main()