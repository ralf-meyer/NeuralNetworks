import unittest
from os.path import normpath, dirname, join
import numpy as _np
from NeuralNetworks.data_generation import data_readers

class PathProvider(object):
    """This class is just used to provide hard coded file paths"""
    _test_files_dir = join(dirname(__file__), normpath("TestData"))
    LAMMPS_test_files_dir = join(_test_files_dir, "Lammps")
    QE_test_files_dir = join(_test_files_dir, "QuantumEspresso")

class BaseTestsWrapper(object):
    """This class is only used to hide the base tests classes,
    to avoid them beeing executed by unittest module.
    """

    class DataReadersTestUtilities(unittest.TestCase):
        """This class provides some utility test functions tailored for
        the attributes of the data readers, which will then be called
        by the actual tests
        """

        def setUp(self):
            self._expected_atom_types = None
            self._expected_number_of_atoms_per_type = None

            # expected geometries
            # should be of the following form:
            # a list of tuples with:
            # - species_ij (species of atom j in step i)
            # - geometries_ij (position of atom j in step i)
            # Example:
            #    [ ((i,j), (species_ij, np.array(geometries_ij))) ]
            self._expected_geometries = None
            # list of indices (i,j) as a tuple
            self._expected_geometries_indices = None

            # expected forces
            # should be of same form as expected geometries, but w/o species_ij,
            # i.e. just a list containing force values (np.array(forces_ij),
            # with forces_ij the forces of atom j in step i).
            self._expected_forces = None
            self._expected_forces_indices = None

        def _check_species(self, reader):
            """Test is species are detected correctly"""
            self.assertItemsEqual(self._expected_atom_types, reader.atom_types)
            self.assertItemsEqual(
                self._expected_number_of_atoms_per_type,
                reader.nr_atoms_per_type
            )

        def _check_geometry(self, reader):
            """Test read geometry entries"""

            for i, (step, atom) in enumerate(self._expected_geometries_indices):
                self._assert_geometry_entries_equals(
                    reader.geometries[step][atom],
                    self._expected_geometries[i]
                )
        
        def _assert_geometry_entries_equals(self, expected, actual):
            """Compares objects of the specific tuple formart (string, np.array)"""

            # compare atomic species
            self.assertEqual(expected[0], actual[0])

            # compare geometry values
            self.assertItemsEqual(expected[1], actual[1])

        def _check_forces(self, reader):
            
            for i, (step, atom) in enumerate(self._expected_forces_indices):
                self.assertItemsEqual(
                    reader.forces[step][atom],
                    self._expected_forces[i]
                )

class TestLammpsReader(unittest.TestCase):
    """This class containts tests for ReadLammpsData.py"""

    path_provider = PathProvider

    def setUp(self):
        # set paths to result files to read from
        self._dump_path = \
            join(self.path_provider.LAMMPS_test_files_dir , "Au_md.dump")
        self._xyz_path = \
            join(self.path_provider.LAMMPS_test_files_dir , "Au_md.xyz")
        self._thermo_path = \
            join(self.path_provider.LAMMPS_test_files_dir , "Au.log")

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

        reader = data_readers.LammpsReader()

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
        
        reader = data_readers.LammpsReader()        
        reader._read_energies_from_thermofile(self._thermo_path)

        self.assertItemsEqual(self._expected_energies, reader.energies)

    def test_read_xyz(self):
        """test reading geometries and species from xyz file"""

        reader = data_readers.LammpsReader()
        reader._read_geometries_from_xyz(self._xyz_path)

        self._check_species(reader.atom_types, reader.number_of_atoms_per_type)
        self._check_geometry(reader.geometries)

    def test_bad_dump_with_xyz_as_backup(self):
        """test if reader can read if dump not found and xyz given as backup"""

        reader = data_readers.LammpsReader()

        try:
            reader.read_lammps(
                "bad/path/to/no_where.dump", 
                self._thermo_path, 
                self._xyz_path)            

        except ValueError as ex:
            self.fail(
                "Probably one xyz file not found... Details: {0}".format(
                    ex.message)
                )
        except Exception as ex:
            self.fail("Unknown error: {0}".format(ex.message))

        # if nothing goes wrong, cheack if any geometries were read
        self.assertGreater(len(reader.geometries), 0)

    def test_setting_species_before_reading(self):
        """Test if geometries are labelled correctly if species are set """

        # define species to be set
        new_types = ["H"]
        new_count_per_type = [26]

        reader = data_readers.LammpsReader()

        # set atom species
        reader.atom_types = new_types
        reader.number_of_atoms_per_type = new_count_per_type

        reader.read_lammps(self._dump_path, self._thermo_path)

        # check if set species persisted or where overwritten
        self.assertItemsEqual(new_types, reader.atom_types)
        self.assertItemsEqual(new_count_per_type, reader.number_of_atoms_per_type)

        # check if label in geometries are correct

        try:
            index_geometry = 0
            for i_atom, atom_type in enumerate(new_types):
                for count in range(new_count_per_type[i_atom]):
                    
                    self.assertEqual(
                        atom_type, 
                        reader.geometries[0][index_geometry][0]
                    )

                    index_geometry += 1

        except IndexError:
            self.fail("Atom types/counts per type, does not match read data!")

class TestQEMDReader(BaseTestsWrapper.DataReadersTestUtilities):
    """Tests the QEReader's read functions
    
    Attributes:
        path_provider: reference to static class that holds values for paths
            (e.g. to qe output files)
    """

    path_provider = PathProvider

    def setUp(self):
        self._out_file = join(self.path_provider.QE_test_files_dir, "Au_qe6.out")

        #--- reference values ---
        self._expected_atom_types = ["Au"]
        self._expected_number_of_atoms_per_type = [13]

        # expected geometries values:
        # first step/first atom, third step/tenth atom and
        # sixth step/thirteenth atom
        self._expected_geometries = [
            ("Au", _np.array([-0.0000000, 0.0000000, 0.0000000])),
            ("Au", _np.array([2.260768389, 0.011928105, -1.490806648])),
            ("Au", _np.array([-0.109880711, -1.599014933, -2.258045452]))
        ]
        self._expected_geometries_indices = [(0, 0), (2, 9), (5, 13)]

        # expected forces: 
        # first step/first atom, fourth step/eighth atom and sixth step/13th atom
        self._expected_forces = [
            _np.array([0.00000100, -0.00000508, 0.00000782]),
            _np.array([-0.02532810, -0.01860887, 0.00830192]),
            _np.array([0.00399592, -0.03540779, -0.04765918])
        ]
        self._expected_forces_indices = [(0, 0), (3, 7), (5, 12)]
        #---

        self._reader = self._create_prepared_reader()

    def _create_prepared_reader(self):
        """will prep the reader by showing it the 
        outfiles and having it read them."""

        path = self._out_file

        try:
            reader = data_readers.QE_MD_Reader()
            reader.get_files(path)
            reader.read_all_files()
            return reader
        except IOError as ex:
            print("IOError when preparing QEMDReader: {0}".format(ex.message))
            self.fail("QEMDReader: file not found at {0}".format(path))


    def test_species_discover(self):
        """tests if species are read correctly from file"""
        self._check_species(self._reader)

    def test_geometries(self):
        self._check_geometry(self._reader)

    def test_forces(self):
        self._check_forces(self._reader)

if __name__ == '__main__':
    unittest.main()