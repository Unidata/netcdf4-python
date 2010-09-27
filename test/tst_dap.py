import unittest
import netCDF4

# test accessing data over http with opendap.

URL = 'http://test.opendap.org/dap/data/nc/testfile.nc'
firstvarname = 'aa'
firstvarmin = -2
firstvarmax = -0
firstvarshape = (4,)

class DapTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """testing access of data over http using opendap"""
        file = netCDF4.Dataset(URL)
        assert firstvarname in file.variables.keys()
        firstvar = file.variables[firstvarname]
        assert firstvar.shape == firstvarshape
        data = firstvar[:]
        assert data.min() == firstvarmin
        assert data.max() == firstvarmax
        file.close()

if __name__ == '__main__':
    unittest.main()
