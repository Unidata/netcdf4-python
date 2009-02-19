from netCDF4 import Dataset
from numpy.random import seed, randint
from numpy.testing import assert_array_equal, assert_equal
import tempfile, unittest, os, random
import numpy as np

file_name = tempfile.mktemp(".nc")
xdim=9; ydim=10; zdim=11
i = np.array([2,5,7],'i4')
ib = np.zeros(ydim,dtype=np.bool)
ib[2] = True; ib[5] = True; ib[7] = True
ib2 = np.zeros(xdim, dtype=np.bool)
ib2[1] = True; ib2[4] = True; ib2[6] = True
# this one should be converted to a slice.
ib3 = np.zeros(xdim, dtype=np.bool)
ib3[0] = True; ib2[4] = True; ib2[8] = True
#seed(9) # fix seed
data = randint(0,10,size=(xdim,ydim,zdim)).astype('i2')
data1 = data[:,0,0].copy()

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = file_name
        f = Dataset(file_name,'w')
        f.createDimension('x',None)
        f.createDimension('y',ydim)
        f.createDimension('z',zdim)
        f.createDimension('time', None)
        v = f.createVariable('data','i2',('x','y','z'))

        vu = f.createVariable('datau', 'i2', ('x', 'y', 'time'))
        
        v[:] = data

        v1 = f.createVariable('data1','i2','x')
        self.data1 = data1
        self.data = data
        # test __setitem___
        v[0:xdim] = self.data
        # integer array slice.
        v[:,i,:] = -100
        self.data[:,i,:] = -100
        # boolen array slice.
        v[ib2] = -200
        self.data[ib2] = -200
        v[ib3,:,:] = -300
        self.data[ib3,:,:] = -300
        # same as above, for 1d array
        v1[0:xdim] = self.data1
        v1[i] = -100
        self.data1[i] = -100
        v1[ib2] = -200
        self.data1[ib2] = -200
        v1[ib3] = -300
        self.data1[ib3] = -300

        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def test_get(self):
        """testing 'fancy indexing'"""
        f  = Dataset(self.file, 'r')
        v = f.variables['data']
        # slice with an array of integers.
        assert_array_equal(v[0:-1:2,i,:],self.data[0:-1:2,i,:])
        # slice with an array of booleans.
        assert_array_equal(v[0:-1:2,ib,:],self.data[0:-1:2,ib,:])
        # Two slices
        assert_array_equal(v[1:2,1:3,:], self.data[1:2,1:3,:])
        # Three sequences
        assert_array_equal(v[i,i,i], self.data[i,i,i])
        assert_equal(v[i,i,i].shape, (3,))
        
        # Two booleans and one slice.  Different from NumPy
        # ibx,ibz should be converted to slice, iby not.
        ibx = np.array([True, False, True, False, True, False, True, False, True])
        iby = np.array([True, False, True, False, False, False, True, False, True, False])
        ibz = np.array([True, False, True, False, True, False, True, False,\
            True, False, True])
        datatmp = self.data[::2,:,:]
        datatmp = datatmp[:,iby,:]
        assert_array_equal(v[ibx, iby, :], datatmp)
        
        # Three booleans
        datatmp = self.data[::2,:,:]
        datatmp = datatmp[:,iby,::2]
        assert_array_equal(v[ibx,iby,ibz], datatmp)
        
        # Ellipse
        assert_array_equal(v[...,::2],self.data[..., ::2])
        assert_array_equal(v[...,::-2],self.data[..., ::-2])
        assert_array_equal(v[[1,2],...],self.data[[1,2],...])
        
        assert_array_equal(v[0], self.data[0])
        
        f.close()

    def test_set(self):
        f  = Dataset(self.file, 'a')
        data = np.arange(180).reshape((9,10,2))
        vu = f.variables['datau']
        
        vu[:,:,0] = data[:,:,0]
        print vu.shape   # This is OK
        #assert_array_equal(vu[:,:,:], data[:,:,:1])
        
        
        vu[:,:,1:] = data[:]
        print data[:].shape, vu.shape  # This is not OK
                
        #print vu[:,:,0]
        #assert_array_equal(vu[:, :, 1:], data)
        
        #vu[:,:,0] = 0.0
        #assert_array_equal(vu[:, :, 0], 0.)
        f.close()
        

if __name__ == '__main__':
    unittest.main()
