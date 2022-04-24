from numpy.random.mtrand import uniform
from netCDF4 import Dataset
from numpy.testing import assert_almost_equal
import os, tempfile, unittest

ndim = 100000
filename1 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
filename2 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
array = uniform(size=(ndim,))

def write_netcdf(filename,dtype='f8',complevel=6):
    nc = Dataset(filename,'w')
    nc.createDimension('n', ndim)
    foo = nc.createVariable('data',\
            dtype,('n'),compression='zstd',complevel=complevel)
    foo[:] = array
    nc.close()

class CompressionTestCase(unittest.TestCase):

    def setUp(self):
        self.filename1 = filename1
        self.filename2 = filename2
        write_netcdf(self.filename1,complevel=0) # no compression
        write_netcdf(self.filename2,complevel=4) # with compression

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.filename1)
        os.remove(self.filename2)

    def runTest(self):
        uncompressed_size = os.stat(self.filename1).st_size
        # check uncompressed data
        f = Dataset(self.filename1)
        size = os.stat(self.filename1).st_size
        assert_almost_equal(array,f.variables['data'][:])
        assert f.variables['data'].filters() ==\
        {'zlib':False,'zstd':False,'bzip2':False,'shuffle':False,'complevel':0,'fletcher32':False}
        assert_almost_equal(size,uncompressed_size)
        f.close()
        # check compressed data.
        f = Dataset(self.filename2)
        size = os.stat(self.filename2).st_size
        assert_almost_equal(array,f.variables['data'][:])
        #print(f.variables['data'].filters())
        assert f.variables['data'].filters() ==\
        {'zlib':False,'zstd':True,'bzip2':False,'shuffle':False,'complevel':4,'fletcher32':False}
        assert(size < 0.96*uncompressed_size)
        f.close()

if __name__ == '__main__':
    unittest.main()
