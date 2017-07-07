# -*- coding: utf-8 -*-
import os, sys
import unittest
import tempfile
import netCDF4

python3 = sys.version_info[0] > 2

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
        nc_non_ascii = netCDF4.Dataset(nc_non_ascii_file, 'w',encoding='utf-8')
        # test that no UnicodeDecodeError occurs in the filepath() method
        if python3:
            assert nc_non_ascii.filepath(encoding='utf-8') == str(nc_non_ascii_file)
        else:
            assert nc_non_ascii.filepath(encoding='utf-8') ==\
            unicode(str(nc_non_ascii_file),encoding='utf-8')
        # cleanup
        nc_non_ascii.close()
        os.remove(nc_non_ascii_file)
        os.rmdir(tempdir)
        
        
if __name__ == '__main__':
    unittest.main()
