import glob, os, sys, unittest, netCDF4
from netCDF4 import getlibversion

__all__ = ['test']
# Find all test files.
test_files = glob.glob('tst_*.py')
py_path = os.environ.get('PYTHONPATH')
if py_path is None:
    py_path = '.'
else:
    py_path = os.pathsep.join(['.',py_path])
os.environ['PYTHONPATH'] = py_path

# Build the test suite from the tests found in the test files.
testsuite = unittest.TestSuite()
version = getlibversion().split()[0]
for f in test_files:
    ff = os.path.join(sys.path[0],f)
    m = __import__(os.path.splitext(f)[0])
    if m.__name__ == 'tst_compoundvar':
        if version < "4.1":
            print \
            "skipping tst_compoundvar, requires netcdf 4.1 or higher, you"\
            +" have %s." % version
            continue
    testsuite.addTests(unittest.TestLoader().loadTestsFromModule(m))
    

# Run the test suite. 


def test(verbosity=1):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite)

if __name__ == '__main__':
    test()
