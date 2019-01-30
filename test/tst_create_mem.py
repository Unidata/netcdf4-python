import unittest
import netCDF4
import numpy as np
from numpy.testing import assert_array_equal

class TestCreateMem(unittest.TestCase):
    def test_mem_create(self):
        def check_inmemory(format):
            # memory is 'advisory size' - not needed for NETCDF4/HDF5
            # but is used for NETCDF3.
            nc = netCDF4.Dataset('test.nc','w',memory=1028,format=format)
            d = nc.createDimension('x',None)
            v = nc.createVariable('v',np.int32,'x')
            data = np.arange(5)
            v[0:5] = data
            # retrieve memory buffer
            b = nc.close()
            # open a new file using this memory buffer
            nc2 = netCDF4.Dataset('test2.nc','r',memory=b)
            assert_array_equal(nc2['v'][:],data)
            nc2.close()
        check_inmemory('NETCDF3_CLASSIC')
        check_inmemory('NETCDF4_CLASSIC')

if __name__ == '__main__':
    unittest.main()
