from numpy.random.mtrand import uniform
from netCDF4 import Dataset, __has_quantization_support__
from numpy.testing import assert_almost_equal
import numpy as np
import os, tempfile, unittest

ndim = 100000
nfiles = 7
files = [tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name for nfile in range(nfiles)]
data_array = uniform(size=(ndim,))
nsd = 3
nsb = 10 # for BitRound, use significant bits (~3.32 sig digits)
complevel = 6

def write_netcdf(filename,zlib,significant_digits,data,dtype='f8',shuffle=False,\
                 complevel=6,quantize_mode="BitGroom"):
    file = Dataset(filename,'w')
    file.createDimension('n', ndim)
    foo = file.createVariable('data',\
            dtype,('n'),zlib=zlib,significant_digits=significant_digits,\
            shuffle=shuffle,complevel=complevel,quantize_mode=quantize_mode)
    foo[:] = data
    file.close()
    file = Dataset(filename)
    data = file.variables['data'][:]
    file.close()


@unittest.skipIf(not __has_quantization_support__, "missing quantisation support")
class CompressionTestCase(unittest.TestCase):
    def setUp(self):
        self.files = files
        # no compression
        write_netcdf(self.files[0],False,None,data_array)
        # compressed, lossless, no shuffle.
        write_netcdf(self.files[1],True,None,data_array)
        # compressed, lossless, with shuffle.
        write_netcdf(self.files[2],True,None,data_array,shuffle=True)
        # compressed, lossy, no shuffle.
        write_netcdf(self.files[3],True,nsd,data_array)
        # compressed, lossy, with shuffle.
        write_netcdf(self.files[4],True,nsd,data_array,shuffle=True)
        # compressed, lossy, with shuffle, and alternate quantization.
        write_netcdf(self.files[5],True,nsd,data_array,quantize_mode='GranularBitRound',shuffle=True)
        # compressed, lossy, with shuffle, and alternate quantization.
        write_netcdf(self.files[6],True,nsb,data_array,quantize_mode='BitRound',shuffle=True)

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
        assert_almost_equal(data_array,f.variables['data'][:])
        assert f.variables['data'].filters() ==\
        {'zlib':True,'szip':False,'zstd':False,'bzip2':False,'blosc':False,'shuffle':False,'complevel':complevel,'fletcher32':False}
        assert size < 0.95*uncompressed_size 
        f.close()
        # check compression with shuffle
        f = Dataset(self.files[2])
        size = os.stat(self.files[2]).st_size
        #print('compressed lossless with shuffle ',size)
        assert_almost_equal(data_array,f.variables['data'][:])
        assert f.variables['data'].filters() ==\
        {'zlib':True,'szip':False,'zstd':False,'bzip2':False,'blosc':False,'shuffle':True,'complevel':complevel,'fletcher32':False}
        assert size < 0.85*uncompressed_size 
        f.close()
        # check lossy compression without shuffle
        f = Dataset(self.files[3])
        size = os.stat(self.files[3]).st_size
        errmax = (np.abs(data_array-f.variables['data'][:])).max()
        #print('compressed lossy no shuffle = ',size,' max err = ',errmax)
        assert f.variables['data'].quantization() == (nsd,'BitGroom') 
        assert errmax < 1.e-3 
        assert size < 0.35*uncompressed_size 
        f.close()
        # check lossy compression with shuffle
        f = Dataset(self.files[4])
        size = os.stat(self.files[4]).st_size
        errmax = (np.abs(data_array-f.variables['data'][:])).max()
        print('compressed lossy with shuffle and standard quantization = ',size,' max err = ',errmax)
        assert f.variables['data'].quantization() == (nsd,'BitGroom') 
        assert errmax < 1.e-3 
        assert size < 0.24*uncompressed_size 
        f.close()
        # check lossy compression with shuffle and alternate quantization
        f = Dataset(self.files[5])
        size = os.stat(self.files[5]).st_size
        errmax = (np.abs(data_array-f.variables['data'][:])).max()
        print('compressed lossy with shuffle and alternate quantization = ',size,' max err = ',errmax)
        assert f.variables['data'].quantization() == (nsd,'GranularBitRound') 
        assert errmax < 1.e-3 
        assert size < 0.24*uncompressed_size 
        f.close()
        # check lossy compression with shuffle and alternate quantization
        f = Dataset(self.files[6])
        size = os.stat(self.files[6]).st_size
        errmax = (np.abs(data_array-f.variables['data'][:])).max()
        print('compressed lossy with shuffle and alternate quantization = ',size,' max err = ',errmax)
        assert f.variables['data'].quantization() == (nsb,'BitRound') 
        assert errmax < 1.e-3 
        assert size < 0.24*uncompressed_size 
        f.close()

if __name__ == '__main__':
    unittest.main()
