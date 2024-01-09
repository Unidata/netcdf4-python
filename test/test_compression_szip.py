from numpy.random.mtrand import uniform
from netCDF4 import Dataset
from numpy.testing import assert_almost_equal
import os, tempfile, unittest, sys
from filter_availability import has_szip_filter

ndim = 100000
filename = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
datarr = uniform(size=(ndim,))

def write_netcdf(filename,dtype='f8'):
    nc = Dataset(filename,'w')
    nc.createDimension('n', ndim)
    foo = nc.createVariable('data',\
            dtype,('n'),compression=None)
    foo_szip = nc.createVariable('data_szip',\
            dtype,('n'),compression='szip',szip_coding='ec',szip_pixels_per_block=32)
    foo[:] = datarr
    foo_szip[:] = datarr
    nc.close()


@unittest.skipIf(not has_szip_filter, "szip filter not available")
class CompressionTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = filename
        write_netcdf(self.filename)

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.filename)

    def runTest(self):
        f = Dataset(self.filename)
        assert_almost_equal(datarr,f.variables['data'][:])
        assert f.variables['data'].filters() ==\
        {'zlib':False,'szip':False,'zstd':False,'bzip2':False,'blosc':False,'shuffle':False,'complevel':0,'fletcher32':False}
        assert_almost_equal(datarr,f.variables['data_szip'][:])
        dtest = {'zlib': False, 'szip': {'coding': 'ec', 'pixels_per_block': 32}, 'zstd': False, 'bzip2': False, 'blosc': False, 'shuffle': False, 'complevel': 0, 'fletcher32': False}
        assert f.variables['data_szip'].filters() == dtest
        f.close()


if __name__ == '__main__':
    unittest.main()
