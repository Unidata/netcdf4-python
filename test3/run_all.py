import glob, os, sys, unittest

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
for f in test_files:
    ff = os.path.join(sys.path[0],f)
    m = __import__(os.path.splitext(f)[0])
    testsuite.addTests(unittest.TestLoader().loadTestsFromModule(m))

# Run the test suite. 
runner = unittest.TextTestRunner()

def test():
    runner.run(testsuite)

if __name__ == '__main__':
    test()
