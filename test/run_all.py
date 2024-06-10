import glob, os, sys, unittest, struct, tempfile
from netCDF4 import __hdf5libversion__,__netcdf4libversion__,__version__, Dataset
# can also just run
# python -m unittest discover . 'tst*py'

# Find all test files.
test_files = glob.glob('test_*.py')
# run opendap test first (issue #856).
test_files.remove('test_dap.py')
test_files.insert(0,'test_dap.py')

# Build the test suite from the tests found in the test files.
testsuite = unittest.TestSuite()
for f in test_files:
    m = __import__(os.path.splitext(f)[0])
    testsuite.addTests(unittest.TestLoader().loadTestsFromModule(m))


if __name__ == '__main__':
    import numpy, cython
    sys.stdout.write('\n')
    sys.stdout.write('netcdf4-python version: %s\n' % __version__)
    sys.stdout.write('HDF5 lib version:       %s\n' % __hdf5libversion__)
    sys.stdout.write('netcdf lib version:     %s\n' % __netcdf4libversion__)
    sys.stdout.write('numpy version           %s\n' % numpy.__version__)
    sys.stdout.write('cython version          %s\n' % cython.__version__)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(testsuite)
    if not result.wasSuccessful():
        sys.exit(1)
