import os
import unittest
import netCDF4

class test_filepath(unittest.TestCase):

    def setUp(self):
        self.netcdf_file = os.path.join(os.getcwd(), "netcdf_dummy_file.nc")
        self.nc = netCDF4.Dataset(self.netcdf_file)

    def test_filepath(self):
        assert self.nc.filepath() == str(self.netcdf_file)

if __name__ == '__main__':
    unittest.main()
