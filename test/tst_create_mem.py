import unittest
import netCDF4
import numpy as np
from numpy.testing import assert_array_equal

class TestCreateMem(unittest.TestCase):
    def test_mem_create(self):
        def check_inmemory(format):
            nc = netCDF4.Dataset('test.nc','w',memory=1,format=format)
            d = nc.createDimension('x',None)
            v = nc.createVariable('v',np.int32,'x')
            data = np.arange(5)
            v[0:5] = data
            b = nc.close()
            nc = netCDF4.Dataset('test.nc','r',memory=b)
            assert_array_equal(nc['v'][:],data)
            nc.close()
        check_inmemory('NETCDF3_CLASSIC')
        check_inmemory('NETCDF4_CLASSIC')

if __name__ == '__main__':
    unittest.main()
