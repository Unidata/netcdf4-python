from netCDF3 import Dataset
from numpy.random import seed, randint
from numpy.testing import assert_array_equal
import tempfile, unittest, os, random
import numpy as NP

file_name = tempfile.mktemp(".nc")
xdim=9; ydim=10; zdim=11
#seed(9) # fix seed
data = randint(0,10,size=(xdim,ydim,zdim)).astype('u1')
datarev = data[:,::-1,:]

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = file_name
        f = Dataset(file_name,'w')
        f.createDimension('x',xdim)
        f.createDimension('xu',None)
        f.createDimension('y',ydim)
        f.createDimension('z',zdim)
        v = f.createVariable('data','i1',('x','y','z'))
        vu = f.createVariable('datau','i1',('xu','y','z'))
        # variable with no unlimited dim.
        # write slice in reverse order
        v[:,::-1,:] = data
        # variable with an unlimited dimension.
        # write slice in reverse order
        vu[:,::-1,:] = data
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing variable slicing"""
        f  = Dataset(self.file, 'r')
        v = f.variables['data']
        vu = f.variables['datau']
        assert_array_equal(v[:], datarev)
        # test reading of slices.
        # negative value means count back from end.
        assert_array_equal(v[:-1,:-2,:-3],datarev[:-1,:-2,:-3])
        # every other element (positive step)
        assert_array_equal(v[2:-1:2,2:-2:2,2:-3:2],datarev[2:-1:2,2:-2:2,2:-3:2])
        # every other element (negative step)
        assert_array_equal(v[-1:2:-2,-2:2:-2,-3:2:-2],datarev[-1:2:-2,-2:2:-2,-3:2:-2])
        # read elements in reverse order
        assert_array_equal(v[:,::-1,:],data)
        assert_array_equal(v[::-1,:,::-1],datarev[::-1,:,::-1])
        assert_array_equal(v[xdim-1::-3,:,zdim-1::-3],datarev[xdim-1::-3,:,zdim-1::-3])           
        # ellipsis slice.
        assert_array_equal(v[...,2:],datarev[...,2:])
        # variable with an unlimited dimension.
        assert_array_equal(vu[:], data[:,::-1,:])
        # read data in reverse order
        assert_array_equal(vu[:,::-1,:],data)
        # index using an integer array scalar
        i = NP.ones(1,'i4')[0]
        assert_array_equal(v[i],datarev[1])

if __name__ == '__main__':
    unittest.main()
