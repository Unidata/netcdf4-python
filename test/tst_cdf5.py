from netCDF4 import Dataset
import numpy as np
import sys, os, unittest, tempfile
from numpy.testing import assert_array_equal

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
dimsize = np.iinfo(np.int32).max*2 # only allowed in CDF5
ndim = 100
arrdata = np.random.randint(np.iinfo(np.uint8).min,np.iinfo(np.uint8).max,size=ndim)

class test_cdf5(unittest.TestCase):

    def setUp(self):
        self.netcdf_file = FILE_NAME
        nc = Dataset(self.netcdf_file,'w',format='NETCDF3_64BIT_DATA')
        # create a 64-bit dimension
        d = nc.createDimension('dim',dimsize) # 64-bit dimension
        # create an 8-bit unsigned integer variable
        v = nc.createVariable('var',np.uint8,'dim')
        v[:ndim] = arrdata
        # create a 64-bit integer attribute (issue #878)
        nc.setncattr('int64_attr', np.int64(-9223372036854775806))
        nc.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.netcdf_file)

    def runTest(self):
        """testing NETCDF3_64BIT_DATA format (CDF-5)"""
        f  = Dataset(self.netcdf_file, 'r')
        assert f.dimensions['dim'].size == dimsize
        assert_array_equal(arrdata, f.variables['var'][:ndim])
        assert (type(f.int64_attr) == np.int64)
        f.close()

if __name__ == '__main__':
    unittest.main()
