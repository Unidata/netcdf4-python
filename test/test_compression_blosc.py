from numpy.random.mtrand import uniform
from netCDF4 import Dataset
from numpy.testing import assert_almost_equal
import os, tempfile, unittest, sys
from filter_availability import no_plugins, has_blosc_filter


ndim = 100000
iblosc_shuffle=2
iblosc_complevel=4
filename = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
datarr = uniform(size=(ndim,))

def write_netcdf(filename,dtype='f8',blosc_shuffle=1,complevel=6):
    nc = Dataset(filename,'w')
    nc.createDimension('n', ndim)
    foo = nc.createVariable('data',\
            dtype,('n'),compression=None)
    foo_lz = nc.createVariable('data_lz',\
            dtype,('n'),compression='blosc_lz',blosc_shuffle=blosc_shuffle,complevel=complevel)
    foo_lz4 = nc.createVariable('data_lz4',\
            dtype,('n'),compression='blosc_lz4',blosc_shuffle=blosc_shuffle,complevel=complevel)
    foo_lz4hc = nc.createVariable('data_lz4hc',\
            dtype,('n'),compression='blosc_lz4hc',blosc_shuffle=blosc_shuffle,complevel=complevel)
    foo_zlib = nc.createVariable('data_zlib',\
            dtype,('n'),compression='blosc_zlib',blosc_shuffle=blosc_shuffle,complevel=complevel)
    foo_zstd = nc.createVariable('data_zstd',\
            dtype,('n'),compression='blosc_zstd',blosc_shuffle=blosc_shuffle,complevel=complevel)
    foo_lz[:] = datarr
    foo_lz4[:] = datarr
    foo_lz4hc[:] = datarr
    foo_zlib[:] = datarr
    foo_zstd[:] = datarr
    nc.close()


@unittest.skipIf(no_plugins or not has_blosc_filter, "blosc filter not available")
class CompressionTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = filename
        write_netcdf(self.filename,complevel=iblosc_complevel,blosc_shuffle=iblosc_shuffle)

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.filename)

    def runTest(self):
        f = Dataset(self.filename)
        assert_almost_equal(datarr,f.variables['data'][:])
        assert f.variables['data'].filters() ==\
        {'zlib':False,'szip':False,'zstd':False,'bzip2':False,'blosc':False,'shuffle':False,'complevel':0,'fletcher32':False}
        assert_almost_equal(datarr,f.variables['data_lz'][:])
        dtest = {'zlib': False, 'szip':False, 'zstd': False, 'bzip2': False, 'blosc':
                {'compressor': 'blosc_lz', 'shuffle': iblosc_shuffle},
                'shuffle': False, 'complevel': iblosc_complevel, 'fletcher32': False}
        assert f.variables['data_lz'].filters() == dtest
        assert_almost_equal(datarr,f.variables['data_lz4'][:])
        dtest = {'zlib': False, 'szip':False, 'zstd': False, 'bzip2': False, 'blosc':
                {'compressor': 'blosc_lz4', 'shuffle': iblosc_shuffle},
                'shuffle': False, 'complevel': iblosc_complevel, 'fletcher32': False}
        assert f.variables['data_lz4'].filters() == dtest
        assert_almost_equal(datarr,f.variables['data_lz4hc'][:])
        dtest = {'zlib': False, 'szip':False, 'zstd': False, 'bzip2': False, 'blosc':
                {'compressor': 'blosc_lz4hc', 'shuffle': iblosc_shuffle},
                'shuffle': False, 'complevel': iblosc_complevel, 'fletcher32': False}
        assert f.variables['data_lz4hc'].filters() == dtest
        assert_almost_equal(datarr,f.variables['data_zlib'][:])
        dtest = {'zlib': False, 'szip':False, 'zstd': False, 'bzip2': False, 'blosc':
                {'compressor': 'blosc_zlib', 'shuffle': iblosc_shuffle},
                'shuffle': False, 'complevel': iblosc_complevel, 'fletcher32': False}
        assert f.variables['data_zlib'].filters() == dtest
        assert_almost_equal(datarr,f.variables['data_zstd'][:])
        dtest = {'zlib': False, 'szip':False, 'zstd': False, 'bzip2': False, 'blosc':
                {'compressor': 'blosc_zstd', 'shuffle': iblosc_shuffle},
                'shuffle': False, 'complevel': iblosc_complevel, 'fletcher32': False}
        assert f.variables['data_zstd'].filters() == dtest
        f.close()


if __name__ == '__main__':
    unittest.main()
