import unittest
from os.path import normpath
import numpy as _np

#import NeuralNetworks.ReadLammpsData.py
import sys
sys.path.append(normpath("../ReadLammpsData"))
#from ..ReadLammpsData import LammpsReader

class TestLammpsReader(unittest.TestCase):
    """This class containts tests for ReadLammpsData.py"""

    def __init__(self):
        # set paths to result files to read from
        self._dump_path = normpath("./TestData/Lammps/Au_md.dump")

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

        super.__init__()

    def _test_species(self, actual_atom_types, actual_number_of_atoms_per_type):
        # test species detection
        self.assertItemsEqual(actual_atom_types, self._expected_atom_types)
        self.assertItemsEqual(
            actual_number_of_atoms_per_type,
            self._expected_number_of_atoms_per_type
        )

    def _test_geometry(self, actual_geometry):
        self.assertItemsEqual(
            actual_geometry[0][2],
            self._expected_geometry_first_step_third_atom
        )
        self.assertItemsEqual(
            actual_geometry[2][10],
            self._expected_geometry_third_step_eleventh_atom
        )
        self.assertItemsEqual(
            actual_geometry[5][25],
            self._expected_geometry_sixth_step_twenty_sixth_atom
        )

    def _test_forces(self, actual_forces):
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
        """Read from dump (in test data) and compare agains hard coded string"""

        reader = ReadLammpsData.LammpsReader()

        # read from dump file
        reader._read_from_dump(self._dump_path)

        # test species
        self._test_species(reader.atom_types, reader.number_of_atoms_per_type)

        # test geometries
        self._test_geometry(reader.geometries)

        # test forces
        self._test_forces(reader.forces)


if __name__ == '__main__':
    unittest.main()
