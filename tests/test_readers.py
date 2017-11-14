import unittest
from os.path import normpath, dirname, join
import numpy as _np
from NeuralNetworks.data_generation import data_readers
from warnings import warn

class PathProvider(object):
    """This class is just used to provide hard coded file paths"""
    _test_files_dir = join(dirname(__file__), normpath("TestData"))
    LAMMPS_test_files_dir = join(_test_files_dir, "Lammps")
    QE_test_files_dir = join(_test_files_dir, "QuantumEspresso")

class _BaseTestsWrapper(object):
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
            """Test if species are detected correctly"""
            self.assertItemsEqual(self._expected_atom_types, reader.atom_types)
            self.assertItemsEqual(
                self._expected_number_of_atoms_per_type,
                reader.nr_atoms_per_type
            )

        def _check_geometry(self, reader):
            """Test read geometry entries"""

            for i, (step, atom) in enumerate(self._expected_geometries_indices):
                self._assert_geometry_entries_equal(
                    self._expected_geometries[i],
                    reader.geometries[step][atom]
                )
        
        def _assert_geometry_entries_equal(self, expected, actual):
            """Compares objects of the specific tuple formart (string, np.array)"""

            # compare atomic species
            self.assertEqual(expected[0], actual[0])

            # compare geometry 
            self._assert_list_almost_equal(expected[1], actual[1])
        
        def _assert_list_almost_equal(self, expected, actual, delta=2):
            # compare length
            self.assertEqual(len(expected), len(actual))

            # compare elements
            for (expected_element, actual_element) in zip(expected, actual):
                self.assertAlmostEqual(expected_element, actual_element, delta)

        def _check_forces(self, reader):
            for i, (step, atom) in enumerate(self._expected_forces_indices):
                self._assert_list_almost_equal(
                    reader.forces[step][atom],
                    self._expected_forces[i]
                )

class TestLammpsReader(_BaseTestsWrapper.DataReadersTestUtilities):
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
        # fist step/third atom, third step/eleventh atom, and
        # sixth step/ twenty sixth atom
        self._expected_geometries = [
            ("Au", _np.array([-7.06365, 1.76129, -0.29539])),
            ("Au", _np.array([-3.26698, -2.05631, 1.26889])),
            ("Au", _np.array([-2.07949, 0.0551922, 5.46916]))
        ]
        self._expected_geometries_indices = [(0, 2), (2, 10), (5, 25)]

        # forces
        # first step/first atom, fourth step/twenty first atom, and
        # sixth step/eighth atom
        self._expected_forces = [
            _np.array([10.1944, -0.782955, 2.71506]),
            _np.array([2.04384, -9.01468, 15.4655]),
            _np.array([0.783606, 0.30807, 1.6233])
        ]
        self._expected_forces_indices = [(0, 0), (3, 20), (5, 7)]

        # energies
        self._expected_energies = \
            [-62.118812, -64.454685, -66.708751, -68.5237, -69.796816, -70.589006]

    def test_read_dump(self):
        """Read from dump (in test data) and compare against hard coded string"""

        reader = data_readers.LammpsReader()

        # read from dump file
        reader.read(self._dump_path, self._xyz_path, self._thermo_path)

        # test results
        self._check_species(reader)
        self._check_geometry(reader)
        self._check_forces(reader)

    def test_read_thermo(self):
        """Read energies from thermo file and compare to hard coded reference"""
        
        reader = data_readers.LammpsReader()        
        reader._read_energies_from_thermofile(self._thermo_path)

        self.assertItemsEqual(self._expected_energies, reader.energies)

    def test_read_xyz(self):
        """test reading geometries and species from xyz file"""

        reader = data_readers.LammpsReader()
        reader._read_geometries_from_xyz(self._xyz_path)

        self._check_species(reader)
        self._check_geometry(reader)

    def test_bad_dump_with_xyz_as_backup(self):
        """test if reader can read if dump not found and xyz given as backup"""

        reader = data_readers.LammpsReader()

        try:
            reader.read(
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
        reader.nr_atoms_per_type = new_count_per_type

        reader.read(self._dump_path, self._thermo_path)

        # check if set species persisted or where overwritten
        self.assertItemsEqual(new_types, reader.atom_types)
        self.assertItemsEqual(new_count_per_type, reader.nr_atoms_per_type)

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

    def test_read_folder(self):
        reader = data_readers.LammpsReader()
        try:
            reader.read_folder(self.path_provider.LAMMPS_test_files_dir)
        except Exception as ex:
            warn(ex.message)
            self.fail("Could not read files from folder.")
        
        self._check_species(reader)
        self._check_geometry(reader)
        self._check_forces(reader)

    def test_read_folder_with_filename(self):
        reader = data_readers.LammpsReader()
        try:
            reader.read_folder(
                join(self.path_provider.LAMMPS_test_files_dir, "Au_md.out")
            )
        except Exception as ex:
            warn(ex.message)
            self.fail("Could not read files from folder.")

        self._check_species(reader)
        self._check_geometry(reader)
        self._check_forces(reader)


class TestQEMDReader(_BaseTestsWrapper.DataReadersTestUtilities):
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
            ("Au", _np.array([2.259098563, 0.007969433, -1.506424622])),
            ("Au", _np.array([-0.091557598, -1.588272346, -2.257035259]))
        ]
        self._expected_geometries_indices = [(0, 0), (2, 9), (5, 12)]

        # expected forces: 
        # sixth step/first atom, fourth step/eighth atom and sixth step/13th atom
        self._expected_forces = [
            _np.array([1.88972599e-06, -9.59980802e-06, 1.47776572e-05]),
            _np.array([-0.04786317, -0.03516567, 0.01568835]),
            _np.array([0.00755119, -0.06691102, -0.09006279])
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