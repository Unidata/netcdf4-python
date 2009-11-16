import unittest
import netCDF4

# test accessing data over http with opendap.

# due to a bug in netcdf 4.0.1, this URL will fail.
#URL = 'http://test.opendap.org/dap/data/nc/test.nc'
#secondvarname = 'b44'
#secondvarmin = -30
#secondvarmax = -12
#secondvarshape = (4,4)
URL = 'http://test.opendap.org/dap/data/nc/testfile.nc'
secondvarname = 'aa'
secondvarmin = -2
secondvarmax = -0
secondvarshape = (4,)

class DapTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """testing access of data over http using opendap"""
        file = netCDF4.Dataset(URL)
        assert file.variables.keys()[1] == secondvarname
        secondvar = file.variables[secondvarname]
        assert secondvar.shape == secondvarshape
        data = secondvar[:]
        assert data.min() == secondvarmin
        assert data.max() == secondvarmax
        file.close()

if __name__ == '__main__':
    unittest.main()
