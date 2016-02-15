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
if __netcdf4libversion__ < '4.1.2':
    test_files.remove('tst_filepath.py')
    sys.stdout.write('not running tst_filepath.py ...\n')
if __netcdf4libversion__ < '4.4.0' or sys.maxsize < 2**32:
    test_files.remove('tst_cdf5.py')
    sys.stdout.write('not running tst_cdf5.py ...\n')

# Don't run tests that require network connectivity
if os.getenv('NO_NET'):
    test_files.remove('tst_dap.py');
    sys.stdout.write('not running tst_dap.py ...\n')

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
    import numpy
    sys.stdout.write('\n')
    sys.stdout.write('netcdf4-python version: %s\n' % __version__)
    sys.stdout.write('HDF5 lib version:       %s\n' % __hdf5libversion__)
    sys.stdout.write('netcdf lib version:     %s\n' % __netcdf4libversion__)
    sys.stdout.write('numpy version           %s\n' % numpy.__version__)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(testsuite)
    if not result.wasSuccessful():
        sys.exit(1)
