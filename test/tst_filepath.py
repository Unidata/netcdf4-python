# -*- coding: utf-8 -*-

import os
import unittest
import tempfile
import netCDF4

class test_filepath(unittest.TestCase):

    def setUp(self):
        self.netcdf_file = os.path.join(os.getcwd(), "netcdf_dummy_file.nc")
        self.nc = netCDF4.Dataset(self.netcdf_file)

    def test_filepath(self):
        assert self.nc.filepath() == str(self.netcdf_file)

    def test_filepath_with_non_ascii_characters(self):
        # create nc-file in a filepath with Non-Ascii-Characters
        tempdir = tempfile.mkdtemp(prefix='ÄÖÜß_')
        nc_non_ascii_file = os.path.join(tempdir, "Besançonalléestraße.nc")
        nc_non_ascii = netCDF4.Dataset(nc_non_ascii_file, 'w')
        
        # test that no UnicodeDecodeError occur in the filepath() method
        assert nc_non_ascii.filepath() == str(nc_non_ascii_file)
        
        # cleanup
        nc_non_ascii.close()
        os.remove(nc_non_ascii_file)
        os.rmdir(tempdir)
        
        
if __name__ == '__main__':
    unittest.main()
