from numpy.random.mtrand import uniform
from netCDF4 import Dataset
from netCDF4.utils import _quantize
from numpy.testing import assert_almost_equal
import numpy as np
import os, tempfile, unittest

ndim = 100000
nfiles = 6
files = [tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name for nfile in range(nfiles)]
array = uniform(size=(ndim,))
nsd = 3
complevel = 6

def write_netcdf(filename,zlib,significant_digits,data,dtype='f8',shuffle=False,\
                 complevel=6):
    file = Dataset(filename,'w')
    file.createDimension('n', ndim)
    foo = file.createVariable('data',\
            dtype,('n'),zlib=zlib,significant_digits=significant_digits,\
            shuffle=shuffle,complevel=complevel)
    foo[:] = data
    file.close()
    file = Dataset(filename)
    data = file.variables['data'][:]
    file.close()

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
        write_netcdf(self.files[3],True,nsd,array)
        # compressed, lossy, with shuffle.
        write_netcdf(self.files[4],True,nsd,array,shuffle=True)
        # compressed, lossy, with shuffle, and alternate quantization.
        write_netcdf(self.files[5],True,-nsd,array,shuffle=True)

    def tearDown(self):
        # Remove the temporary files
        for file in self.files:
            os.remove(file)

    def runTest(self):
        """testing zlib and shuffle compression filters"""
        uncompressed_size = os.stat(self.files[0]).st_size
        #print('uncompressed size = ',uncompressed_size)
        # check compressed data.
        f = Dataset(self.files[1])
        size = os.stat(self.files[1]).st_size
        #print('compressed lossless no shuffle = ',size)
        assert_almost_equal(array,f.variables['data'][:])
        assert f.variables['data'].filters() == {'zlib':True,'shuffle':False,'complevel':complevel,'fletcher32':False}
        assert(size < 0.95*uncompressed_size)
        f.close()
        # check compression with shuffle
        f = Dataset(self.files[2])
        size = os.stat(self.files[2]).st_size
        #print('compressed lossless with shuffle ',size)
        assert_almost_equal(array,f.variables['data'][:])
        assert f.variables['data'].filters() == {'zlib':True,'shuffle':True,'complevel':complevel,'fletcher32':False}
        assert(size < 0.85*uncompressed_size)
        f.close()
        # check lossy compression without shuffle
        f = Dataset(self.files[3])
        size = os.stat(self.files[3]).st_size
        errmax = (np.abs(array-f.variables['data'][:])).max()
        #print('compressed lossy no shuffle = ',size,' max err = ',errmax)
        assert(f.variables['data'].significant_digits() == nsd)
        assert(errmax < 1.e-3)
        assert(size < 0.35*uncompressed_size)
        f.close()
        # check lossy compression with shuffle
        f = Dataset(self.files[4])
        size = os.stat(self.files[4]).st_size
        errmax = (np.abs(array-f.variables['data'][:])).max()
        #print('compressed lossy with shuffle and standard quantization = ',size,' max err = ',errmax)
        assert(f.variables['data'].significant_digits() == nsd)
        assert(errmax < 1.e-3)
        assert(size < 0.24*uncompressed_size)
        f.close()
        # check lossy compression with shuffle and alternate quantization
        f = Dataset(self.files[5])
        size = os.stat(self.files[5]).st_size
        errmax = (np.abs(array-f.variables['data'][:])).max()
        #print('compressed lossy with shuffle and alternate quantization = ',size,' max err = ',errmax)
        assert(f.variables['data'].significant_digits() == -nsd)
        assert(errmax < 1.e-3)
        assert(size < 0.24*uncompressed_size)
        f.close()

if __name__ == '__main__':
    unittest.main()
