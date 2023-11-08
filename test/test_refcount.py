import unittest, netCDF4, tempfile, os

file_name = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

class RefCountTestCase(unittest.TestCase):

    def setUp(self):
        nc = netCDF4.Dataset(file_name, mode='w', keepweakref=True, format='NETCDF4')
        d = nc.createDimension('fred', 2000)
        v = nc.createVariable('frank','f',('fred',))
        self.file = file_name
        self.nc = nc

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing garbage collection (issue 218)"""
        # this should trigger garbage collection (__dealloc__ method)
        del self.nc
        # if __dealloc__ not called to close file, then this
        # will fail with "Permission denied" error (since you can't
        # open a file 'w' that is already open for writing).
        nc = netCDF4.Dataset(self.file, mode='w', format='NETCDF4')

if __name__ == '__main__':
    unittest.main()
