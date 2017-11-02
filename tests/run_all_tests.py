import unittest

tests = ['tests.dummy_test.py',]

if __name__ == '__main__':
    testsuite = unittest.TestLoader().loadTestsFromNames(tests)
    unittest.TextTestRunner(verbosity=1).run(testsuite)
    
