import glob, os, sys, unittest, netCDF4
from netCDF4 import getlibversion,__hdf5libversion__,__netcdf4libversion__,__version__

# can also just run
# python -m unittest discover . 'tst*py'

python3 = sys.version_info[0] > 2

# Find all test files.
test_files = glob.glob('tst_*.py')
if python3:
    test_files.remove('tst_unicode.py')
    sys.stdout.write('not running tst_unicode.py ...\n')
else:
    test_files.remove('tst_unicode3.py')
    sys.stdout.write('not running tst_unicode3.py ...\n')
if __netcdf4libversion__ < '4.2.1':
    test_files.remove('tst_diskless.py')
    sys.stdout.write('not running tst_diskless.py ...\n')

# Build the test suite from the tests found in the test files.
testsuite = unittest.TestSuite()
for f in test_files:
    m = __import__(os.path.splitext(f)[0])
    testsuite.addTests(unittest.TestLoader().loadTestsFromModule(m))

# Run the test suite.
def test(verbosity=1):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite)

if __name__ == '__main__':
    test(verbosity=1)
    sys.stdout.write('\n')
    sys.stdout.write('netcdf4-python version: %s\n' % __version__)
    sys.stdout.write('HDF5 lib version:       %s\n' % __hdf5libversion__)
    sys.stdout.write('netcdf lib version:     %s\n' % __netcdf4libversion__)
