from numpy.random.mtrand import uniform
from netCDF4 import Dataset, _quantize
from numpy.testing import assert_almost_equal
import os, tempfile, unittest

ndim = 100000
nfiles = 6
files = [tempfile.mktemp(".nc") for nfile in range(nfiles)]
array = uniform(size=(ndim,))
lsd = 3

def write_netcdf(filename,zlib,least_significant_digit,data,dtype='f8',shuffle=False,contiguous=False,complevel=6,fletcher32=False):
    file = Dataset(filename,'w')
    file.createDimension('n', ndim)
    foo = file.createVariable('data', dtype,('n'),zlib=zlib,least_significant_digit=least_significant_digit,shuffle=shuffle,contiguous=contiguous,complevel=complevel,fletcher32=fletcher32)
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
        # compressed, lossy, with shuffle and fletcher32 checksum.
        write_netcdf(self.files[5],True,lsd,array,shuffle=True,fletcher32=True)

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
        assert f.variables['data'].filters() == {'zlib':True,'shuffle':False,'complevel':6,'fletcher32':False}
        assert(size < 0.95*uncompressed_size)
        f.close()
        # check compression with shuffle
        f = Dataset(self.files[2])
        size = os.stat(self.files[2]).st_size
        assert_almost_equal(array,f.variables['data'][:])
        assert f.variables['data'].filters() == {'zlib':True,'shuffle':True,'complevel':6,'fletcher32':False}
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
        size_save = size
        # check lossy compression with shuffle and fletcher32 checksum.
        f = Dataset(self.files[5])
        size = os.stat(self.files[5]).st_size
        assert_almost_equal(checkarray,f.variables['data'][:])
        assert f.variables['data'].filters() == {'zlib':True,'shuffle':True,'complevel':6,'fletcher32':True}
        assert(size < 0.20*uncompressed_size)
        # should be slightly larger than without fletcher32
        assert(size > size_save)

if __name__ == '__main__':
    unittest.main()
