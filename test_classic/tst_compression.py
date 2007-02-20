from numpy.random.mtrand import uniform
from netCDF4_classic import Dataset, _quantize
from numpy.testing import assert_almost_equal
import os, tempfile, unittest

ndim = 100000
nfiles = 5
files = [tempfile.mktemp(".nc") for nfile in range(nfiles)]
array = uniform(size=(ndim,))
lsd = 3

def write_netcdf(filename,zlib,least_significant_digit,data,dtype='f8',shuffle=False,chunking='seq',complevel=6):
    file = Dataset(filename,'w')
    file.createDimension('n', ndim)
    foo = file.createVariable('data', dtype,('n'),zlib=zlib,least_significant_digit=least_significant_digit,shuffle=shuffle,chunking=chunking,complevel=complevel)
    foo[:] = data
    file.close()
    file = Dataset(filename)
    data = file.variables['data'][:]

class CompressionTestCase(unittest.TestCase):

    def setUp(self):
        self.files = files
        # no compression
        write_netcdf(self.files[0],False,None,array)
        # compressed, lossless, no shuffle.
        write_netcdf(self.files[1],True,None,array)
        # compressed, lossless, with shuffle.
        write_netcdf(self.files[2],True,None,array,shuffle=True)
        # compressed, lossy, no shuffle.
        write_netcdf(self.files[3],True,lsd,array)
        # compressed, lossy, with shuffle.
        write_netcdf(self.files[4],True,lsd,array,shuffle=True)

    def tearDown(self):
        # Remove the temporary files
        for file in self.files:
            os.remove(file)

    def runTest(self):
        """testing zlib and shuffle compression filters"""
        uncompressed_size = os.stat(self.files[0]).st_size
        # check compressed data.
        f = Dataset(self.files[1])
        size = os.stat(self.files[1]).st_size
        assert_almost_equal(array,f.variables['data'][:])
        assert f.variables['data'].compression() == {'zlib':True,'shuffle':False,'complevel':6}
        assert(size < 0.95*uncompressed_size)
        f.close()
        # check compression with shuffle
        f = Dataset(self.files[2])
        size = os.stat(self.files[2]).st_size
        assert_almost_equal(array,f.variables['data'][:])
        assert f.variables['data'].compression() == {'zlib':True,'shuffle':True,'complevel':6}
        assert(size < 0.85*uncompressed_size)
        # check lossy compression without shuffle
        f = Dataset(self.files[3])
        size = os.stat(self.files[3]).st_size
        checkarray = _quantize(array,lsd)
        assert_almost_equal(checkarray,f.variables['data'][:])
        assert(size < 0.27*uncompressed_size)
        # check lossy compression with shuffle
        f = Dataset(self.files[4])
        size = os.stat(self.files[4]).st_size
        assert_almost_equal(checkarray,f.variables['data'][:])
        assert(size < 0.20*uncompressed_size)

if __name__ == '__main__':
    unittest.main()
