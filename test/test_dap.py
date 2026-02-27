import unittest
import netCDF4
import numpy as np
from datetime import datetime, timedelta
from numpy.testing import assert_array_almost_equal
import os

# test accessing data over http with opendap.

yesterday = datetime.now() - timedelta(days=1)
URL = f'https://tds.scigw.unidata.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_onedegree_noaaport/GFS_Global_onedeg_noaaport_{yesterday:%Y%m%d}_1800.grib2'
varname = 'Geopotential_height_surface'
data_min = -40; data_max = 5900
varshape = (181, 360)


@unittest.skipIf(os.getenv("NO_NET"), "network tests disabled")
class DapTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        """testing access of data over https using opendap"""
        ncfile = netCDF4.Dataset(URL)
        assert varname in ncfile.variables.keys()
        var = ncfile.variables[varname]
        data = var[0,...]
        assert data.shape == varshape
        assert np.abs(data.min()-data_min) < 10 
        assert np.abs(data.max()-data_max) < 100 
        ncfile.close()

if __name__ == '__main__':
    unittest.main()
