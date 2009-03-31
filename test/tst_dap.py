import unittest
import netCDF4

# test accessing data over http with opendap.

# due to a bug in netcdf 4.0.1, this URL will fail.
#URL = 'http://test.opendap.org/dap/data/nc/test.nc'
#firstvarname = 'b44'
#firstvarmin = -30
#firstvarmax = -12
#firstvarshape = (4,4)
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
        assert file.variables.keys()[0] == firstvarname
        firstvar = file.variables[firstvarname]
        assert firstvar.shape == firstvarshape
        data = firstvar[:]
        assert data.min() == firstvarmin
        assert data.max() == firstvarmax
        file.close()

if __name__ == '__main__':
    unittest.main()
