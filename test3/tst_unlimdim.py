import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform 
from numpy.testing import assert_array_equal, assert_array_almost_equal
import netCDF3

# test creating variables with unlimited dimensions,
# writing to and retrieving data from such variables.

# create an n1dim by n2dim by n3dim random array
n1dim = 4
n2dim = 10
n3dim = 8
ranarr = 100.*uniform(size=(n1dim,n2dim,n3dim))
FILE_NAME = tempfile.mktemp(".nc")

class UnlimdimTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = netCDF3.Dataset(self.file, 'w')
        # foo has a single unlimited dimension
        f.createDimension('n1', None)
        f.createDimension('n2', n2dim)
        f.createDimension('n3', n3dim)
        foo = f.createVariable('data1', ranarr.dtype.str[1:], ('n1','n2','n3'))
        # write some data to it.
        #foo[0:n1dim,:,:] = ranarr
        foo[:] = ranarr
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing unlimited dimensions"""
        f  = netCDF3.Dataset(self.file, 'r')
        foo = f.variables['data1']
        # make sure n1 dimension is unlimited.
        self.assert_(f.dimensions['n1'].isunlimited())
        # check shape.
        self.assert_(foo.shape == (n1dim,n2dim,n3dim))
        # check data.
        assert_array_almost_equal(foo[:,:,:], ranarr)
        f.close()

if __name__ == '__main__':
    unittest.main()
