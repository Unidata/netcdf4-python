import unittest, os, tempfile
import numpy as np
from numpy.random.mtrand import uniform
from numpy.testing import assert_array_equal, assert_array_almost_equal
import netCDF4

# rudimentary test of diskless file capability.

# create an n1dim by n2dim by n3dim random array
n1dim = 10
n2dim = 73
n3dim = 144
ranarr = 100.*uniform(size=(n1dim,n2dim,n3dim))
ranarr2 = 100.*uniform(size=(n1dim,n2dim,n3dim))
FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=True).name
FILE_NAME2 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

class DisklessTestCase(unittest.TestCase):

    def setUp(self):
        # in memory file, does not exist on disk (closing it
        # makes data disappear from memory)
        self.file = FILE_NAME
        f = netCDF4.Dataset(self.file,'w',diskless=True, persist=False)
        self.f = f
        # foo has a single unlimited dimension
        f.createDimension('n1', n1dim)
        f.createDimension('n2', n2dim)
        f.createDimension('n3', n3dim)
        foo = f.createVariable('data1', ranarr.dtype.str[1:], ('n1','n2','n3'))
        # write some data to it.
        foo[0:n1dim-1] = ranarr[:-1,:,:]
        foo[n1dim-1] = ranarr[-1,:,:]
        # bar has 2 unlimited dimensions
        f.createDimension('n4', None)
        # write some data to it.
        bar = f.createVariable('data2', ranarr.dtype.str[1:], ('n1','n2','n4'))
        bar[0:n1dim,:, 0:n3dim] = ranarr2

        # in memory file, that is persisted to disk when close method called.
        self.file2 = FILE_NAME2
        f2 = netCDF4.Dataset(self.file2,'w',diskless=True, persist=True)
        f2.createDimension('n1', n1dim)
        f2.createDimension('n2', n2dim)
        f2.createDimension('n3', n3dim)
        foo = f2.createVariable('data1', ranarr.dtype.str[1:], ('n1','n2','n3'))
        # write some data to it.
        foo[0:n1dim-1] = ranarr[:-1,:,:]
        foo[n1dim-1] = ranarr[-1,:,:]
        f2.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file2)
        self.f.close()

    def runTest(self):
        """testing diskless file capability"""
        foo = self.f.variables['data1']
        bar = self.f.variables['data2']
        # check shape.
        self.assertTrue(foo.shape == (n1dim,n2dim,n3dim))
        self.assertTrue(bar.shape == (n1dim,n2dim,n3dim))
        # check data.
        assert_array_almost_equal(foo[:], ranarr)
        assert_array_almost_equal(bar[:], ranarr2)
        # file does not actually exist on disk
        assert(os.path.isfile(self.file)==False)
        # open persisted file.
        # first, check that file does actually exist on disk
        assert(os.path.isfile(self.file2)==True)
        f = netCDF4.Dataset(self.file2)
        foo = f.variables['data1']
        # check shape.
        self.assertTrue(foo.shape == (n1dim,n2dim,n3dim))
        # check data.
        assert_array_almost_equal(foo[:], ranarr)
        f.close()

if __name__ == '__main__':
    unittest.main()
