import unittest
import netCDF4

# test accessing data over http with opendap.

URL = 'http://test.opendap.org/dap/data/nc/test.nc'
firstvarname = 'b44'
firstvarmin = -30
firstvarmax = -12

class DapTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """testing access of data over http using opendap"""
        file = netCDF4.Dataset(URL)
        assert file.variables.keys()[0] == firstvarname
        firstvar = file.variables[firstvarname]
        assert firstvar.shape == (4,4)
        data = firstvar[:]
        assert data.min() == firstvarmin
        assert data.max() == firstvarmax
        file.close()

if __name__ == '__main__':
    unittest.main()
