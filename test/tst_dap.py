import unittest
import netCDF4
from numpy.testing import assert_array_almost_equal

# test accessing data over http with opendap.

URL = "http://remotetest.unidata.ucar.edu/thredds/dodsC/testdods/testData.nc"
URL_https = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/modis/L3/aqua/11um/v2019.0/4km/daily/2017/365/AQUA_MODIS.20171231.L3m.DAY.NSST.sst.4km.nc'
varname = 'Z_sfc'
varmin = 0
varmax = 3292
varshape = (1,95,135)

class DapTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """testing access of data over http using opendap"""
        ncfile = netCDF4.Dataset(URL)
        assert varname in ncfile.variables.keys()
        var = ncfile.variables[varname]
        assert var.shape == varshape
        data = var[:]
        assert_array_almost_equal(data.min(),varmin)
        assert_array_almost_equal(data.max(),varmax)
        ncfile.close()
        # test https support (linked curl lib must built with openssl support)
        ncfile = netCDF4.Dataset(URL_https)
        assert(ncfile['sst'].long_name=='Sea Surface Temperature')    
        ncfile.close()

if __name__ == '__main__':
    unittest.main()
