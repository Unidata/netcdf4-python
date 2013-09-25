import netCDF4
import numpy as np
import unittest, os, tempfile
from numpy.testing import assert_array_equal, assert_array_almost_equal

data = np.arange(12,dtype='f4').reshape(3,4)
little_little = np.array(data, dtype='<f4')
little_big = np.array(data, dtype='>f4')
big_little = np.array(data, dtype='<f4')
big_big = np.array(data, dtype='>f4')
FILE_NAME = tempfile.mktemp(".nc")
FILE_NAME2 = tempfile.mktemp(".nc")

class EndianTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        def create_file(file,format):
            dataset = netCDF4.Dataset(file,'w',format=format)
            dataset.createDimension('time', None)
            dataset.createDimension('space', 4)
            dims = ('time', 'space')
            ll = dataset.createVariable('little-little', '<f4', dims)
            lb = dataset.createVariable('little-big', '<f4', dims)
            bl = dataset.createVariable('big-little', '>f4', dims)
            bb = dataset.createVariable('big-big', '>f4', dims)
            ll[:] = little_little
            lb[:] = little_big
            bl[:] = big_little
            bb[:] = big_big
            dataset.close()
        create_file(FILE_NAME,'NETCDF3_CLASSIC'); self.file=FILE_NAME
        create_file(FILE_NAME2,'NETCDF4_CLASSIC'); self.file2=FILE_NAME2

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
        os.remove(self.file2)

    def runTest(self):
        """testing endian conversion capability"""
        def check_data(file):
            f = netCDF4.Dataset(file)
            ll = f.variables['little-little']
            lb = f.variables['little-big']
            bb = f.variables['big-big']
            bl = f.variables['big-little']
            # check data.
            assert_array_almost_equal(ll[:], data)
            assert_array_almost_equal(lb[:], data)
            assert_array_almost_equal(bl[:], data)
            assert_array_almost_equal(bb[:], data)
            f.close()
        check_data(self.file)
        check_data(self.file2)

if __name__ == '__main__':
    unittest.main()
