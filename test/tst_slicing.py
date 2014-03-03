from netCDF4 import Dataset
from numpy.random import seed, randint
from numpy.testing import assert_array_equal, assert_equal
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
        f.createDimension('zu',None)
        v = f.createVariable('data','u1',('x','y','z'))
        vu = f.createVariable('datau','u1',('xu','y','zu'))
        v1 = f.createVariable('data1d', 'u1', ('x',))
        # variable with no unlimited dim.
        # write slice in reverse order
        v[:,::-1,:] = data
        # variable with an unlimited dimension.
        # write slice in reverse order
        #vu[0:xdim,::-1,0:zdim] = data
        vu[:,::-1,:] = data

        v1[:] = data[:, 0, 0]
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def test_3d(self):
        """testing variable slicing"""
        f  = Dataset(self.file, 'r')
        v = f.variables['data']
        vu = f.variables['datau']

        # test return of array scalar.
        assert_equal(v[0,0,0].shape,())
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

        f.close()

    def test_1d(self):
        f  = Dataset(self.file, 'r')
        v1 = f.variables['data1d']
        d = data[:,0,0]
        assert_equal(v1[:], d)
        assert_equal(v1[4:], d[4:])
        # test return of array scalar.
        assert_equal(v1[0].shape, ())
        i1 = NP.array([2,3,4])
        assert_equal(v1[i1], d[i1])
        i2 = NP.array([2,3,5])
        assert_equal(v1[i2], d[i2])
        assert_equal(v1[d<5], d[d<5])
        assert_equal(v1[5], d[5])
        f.close()

    def test_0d(self):
        f = Dataset(self.file, 'w')
        v = f.createVariable('data', float)
        v[...] = 10
        assert_array_equal(v[...], 10)
        assert_equal(v.shape, v[...].shape)
        f.close()
 

if __name__ == '__main__':
    unittest.main()
