import unittest
import netCDF4
import numpy as np
from datetime import datetime, timedelta
from numpy.testing import assert_array_almost_equal
import os

# test accessing data over http with opendap.

yesterday = datetime.utcnow() - timedelta(days=1)
URL = f'http://nomads.ncep.noaa.gov/dods/gfs_1p00/gfs{yesterday:%Y%m%d}/gfs_1p00_00z'
URL_https = 'https://www.neracoos.org/erddap/griddap/WW3_EastCoast_latest'
varname = 'hgtsfc'
data_min = -40; data_max = 5900
varshape = (181, 360)


@unittest.skipIf(os.getenv("NO_NET"), "network tests disabled")
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
        assert np.abs(data.min()-data_min) < 10 
        assert np.abs(data.max()-data_max) < 100 
        ncfile.close()
        # test https support (linked curl lib must built with openssl support)
        ncfile = netCDF4.Dataset(URL_https)
        assert ncfile['hs'].long_name=='Significant Wave Height' 
        ncfile.close()

if __name__ == '__main__':
    unittest.main()
