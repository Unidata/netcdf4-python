from netCDF4 import Dataset
import tempfile, unittest, os
import numpy as np

file_name = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
xdim=None; ydim=121; zdim=169
datashape = (ydim,zdim)
data = np.ones(datashape,dtype=np.float)

class ShapeTestCase(unittest.TestCase):

    def setUp(self):
        self.file = file_name
        f = Dataset(file_name,'w')
        f.createDimension('x',xdim)
        f.createDimension('y',ydim)
        f.createDimension('z',zdim)
        v = f.createVariable('data',np.float,('x','y','z'))
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """test for issue 90 (array shape should not be modified by
        assigment to netCDF variable)"""
        f  = Dataset(self.file, 'a')
        v = f.variables['data']
        v[0] = data
        # make sure shape of data array
        # is not changed by assigning it
        # to a netcdf var with one more dimension (issue 90)
        assert(data.shape == datashape)
        f.close()

if __name__ == '__main__':
    unittest.main()
