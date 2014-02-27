import unittest, netCDF4, tempfile, os

file_name = tempfile.mktemp(".nc")

class RefCountTestCase(unittest.TestCase):

    def setUp(self):
        nc = netCDF4.Dataset(file_name, mode='w', format='NETCDF4')
        d = nc.createDimension('fred', 2000)
        v = nc.createVariable('frank','f',('fred',))
        self.file = file_name
        self.nc = nc

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
        pass

    def runTest(self):
        """testing garbage collection (issue 218)"""
        del self.nc
        nc = netCDF4.Dataset(self.file, mode='w', format='NETCDF4')

if __name__ == '__main__':
    unittest.main()
