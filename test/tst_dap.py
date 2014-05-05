import unittest
import netCDF4

# test accessing data over http with opendap.

URL = 'http://test.opendap.org/opendap/hyrax/data/nc/bears.nc'
firstvarname = 'i'
firstvarmin = 10
firstvarmax = 20
firstvarshape = (2,)

class DapTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """testing access of data over http using opendap"""
        ncfile = netCDF4.Dataset(URL)
        assert firstvarname in ncfile.variables.keys()
        firstvar = ncfile.variables[firstvarname]
        assert firstvar.shape == firstvarshape
        data = firstvar[:]
        assert data.min() == firstvarmin
        assert data.max() == firstvarmax
        ncfile.close()

if __name__ == '__main__':
    unittest.main()
