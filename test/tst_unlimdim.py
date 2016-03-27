import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform
from numpy.testing import assert_array_equal, assert_array_almost_equal
import netCDF4

# test creating variables with unlimited dimensions,
# writing to and retrieving data from such variables.

# create an n1dim by n2dim by n3dim random array
n1dim = 4
n2dim = 10
n3dim = 8
ranarr = 100.*uniform(size=(n1dim,n2dim,n3dim))
FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

class UnlimdimTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = netCDF4.Dataset(self.file, 'w')
        # foo has a single unlimited dimension
        f.createDimension('n1', n1dim)
        f.createDimension('n2', None)
        f.createDimension('n3', n3dim)
        foo = f.createVariable('data1', ranarr.dtype.str[1:], ('n1','n2','n3'))
        # write some data to it.
        #foo[:,0:n2dim,:] = ranarr
        foo[:] = ranarr
        foo[:,n2dim:,:] = 2.*ranarr
        # bar has 2 unlimited dimensions
        f.createDimension('n4', None)
        f.createDimension('n5', n2dim)
        f.createDimension('n6', None)
        # write some data to it.
        bar = f.createVariable('data2', ranarr.dtype.str[1:], ('n4','n5','n6'))
#       bar[0:n1dim,:, 0:n3dim] = ranarr
        bar[0:n1dim,:, 0:n3dim] = 2.0
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing unlimited dimensions"""
        f  = netCDF4.Dataset(self.file, 'r')
        foo = f.variables['data1']
        # check shape.
        self.assertTrue(foo.shape == (n1dim,2*n2dim,n3dim))
        # check data.
        assert_array_almost_equal(foo[:,0:n2dim,:], ranarr)
        assert_array_almost_equal(foo[:,n2dim:2*n2dim,:], 2.*ranarr)
        bar = f.variables['data2']
        # check shape.
        self.assertTrue(bar.shape == (n1dim,n2dim,n3dim))
        # check data.
        #assert_array_almost_equal(bar[:,:,:], ranarr)
        assert_array_almost_equal(bar[:,:,:], 2.*NP.ones((n1dim,n2dim,n3dim),ranarr.dtype))
        f.close()

if __name__ == '__main__':
    unittest.main()
