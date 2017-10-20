import os, sys, shutil
import tempfile
import unittest
import netCDF4


class test_filepath(unittest.TestCase):

    def setUp(self):
        self.netcdf_file = os.path.join(os.getcwd(), "netcdf_dummy_file.nc")
        self.nc = netCDF4.Dataset(self.netcdf_file)

    def test_filepath(self):
        assert self.nc.filepath() == str(self.netcdf_file)

    def test_filepath_with_non_ascii_characters(self):
        # create nc-file in a filepath using a cp1252 string
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir,b'Pl\xc3\xb6n.nc'.decode('cp1252'))
        nc = netCDF4.Dataset(filepath,'w',encoding='cp1252')
        filepatho = nc.filepath(encoding='cp1252')
        assert filepath == filepatho
        assert filepath.encode('cp1252') == filepatho.encode('cp1252')
        nc.close()
        shutil.rmtree(tmpdir)

    def test_no_such_file_raises(self):
        fname = 'not_a_nc_file.nc'
        with self.assertRaisesRegexp(IOError, fname):
            netCDF4.Dataset(fname, 'r')


if __name__ == '__main__':
    unittest.main()
