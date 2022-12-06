import unittest
import netCDF4
import numpy as np
from datetime import datetime, timedelta
from numpy.testing import assert_array_almost_equal

# test accessing data over http with opendap.

yesterday = datetime.utcnow() - timedelta(days=1)
URL = "http://nomads.ncep.noaa.gov/dods/gfs_1p00/gfs%s/gfs_1p00_00z" % yesterday.strftime('%Y%m%d')
URL_https = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/modis/L3/aqua/11um/v2019.0/4km/daily/2017/365/AQUA_MODIS.20171231.L3m.DAY.NSST.sst.4km.nc'
varname = 'hgtsfc'
data_min = -40; data_max = 5900
varshape = (181, 360)

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
        data = var[0,...]
        assert data.shape == varshape
        assert(np.abs(data.min()-data_min) < 10)
        assert(np.abs(data.max()-data_max) < 100)
        ncfile.close()
        # test https support (linked curl lib must built with openssl support)
        ncfile = netCDF4.Dataset(URL_https)
        assert(ncfile['sst'].long_name=='Sea Surface Temperature')    
        ncfile.close()

if __name__ == '__main__':
    unittest.main()
