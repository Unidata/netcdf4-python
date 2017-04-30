import os
import unittest
import netCDF4

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class TestOpenMem(unittest.TestCase):
    def test_mem_open(self):
        # Needs: https://github.com/Unidata/netcdf-c/pull/400
        if netCDF4.__netcdf4libversion__ >= '4.4.1.2':
            fpath = os.path.join(CURRENT_DIR, "netcdf_dummy_file.nc")

            with open(fpath, 'rb') as f:
                nc_bytes = f.read()

            with netCDF4.Dataset('foo_bar', memory=nc_bytes) as nc:
                print(nc.filepath())
                assert nc.filepath() == 'foo_bar'
                assert nc.project_summary == 'Dummy netCDF file'

if __name__ == '__main__':
    unittest.main()
