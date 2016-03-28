from netCDF4 import Dataset
from numpy.random import seed, randint
from numpy.testing import assert_array_equal, assert_equal,\
assert_array_almost_equal
import tempfile, unittest, os, random
import numpy as np

file_name = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
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
        i = np.ones(1,'i4')[0]
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
        i1 = np.array([2,3,4])
        assert_equal(v1[i1], d[i1])
        i2 = np.array([2,3,5])
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
        assert(type(v[...]) == np.ndarray)
        f.close()

    def test_issue259(self):
        dset = Dataset(self.file, 'w', format='NETCDF4_CLASSIC')
        dset.createDimension('dim', None)
        a = dset.createVariable('a', 'i', ('dim',))
        b = dset.createVariable('b', 'i', ('dim',))
        c = dset.createVariable('c', 'i', ('dim',))
        c[:] = 1 # c initially is empty, new entry created
        assert_array_equal(c[...], np.array([1]))
        b[:] = np.array([1,1])
        a[:] = 1 # a should be same as b
        assert_array_equal(a[...], b[...])
        dset.close()

    def test_issue371(self):
        dataset = Dataset(self.file, 'w')
        dataset.createDimension('dim', 5)
        var = dataset.createVariable('bar', 'i8', ('dim', ))
        data = [1, 2, 3, 4, 5]
        var[..., :] = data
        assert_array_equal(var[..., :], np.array(data))
        dataset.close()

    def test_issue306(self):
        f = Dataset(self.file,'w')
        nlats = 7; lat = f.createDimension('lat',nlats)
        nlons = 12; lon = f.createDimension('lon',nlons)
        nlevs = 1; lev = f.createDimension('lev',nlevs)
        time = f.createDimension('time',None)
        var = f.createVariable('var',np.float,('time','lev','lat','lon'))
        a = np.random.uniform(size=(10,nlevs,nlats,nlons))
        var[0:10] = a
        f.close()
        f = Dataset(self.file)
        aa = f.variables['var'][4,-1,:,:]
        assert_array_almost_equal(a[4,-1,:,:],aa)
        v = f.variables['var']
        try:
            aa = v[4,-2,:,:] # -2 when dimension is length 1
        except IndexError:
            pass
        else:
            raise IndexError('This test should have failed.')
        try:
            aa = v[4,...,...,:] # more than one Ellipsis
        except IndexError:
            pass
        else:
            raise IndexError('This test should have failed.')
        try:
            aa = v[:,[True,True],:,:] # boolean array too long.
        except IndexError:
            pass
        else:
            raise IndexError('This test should have failed.')
        try:
            aa = v[:,[0,1],:,:] # integer index too large
        except IndexError:
            pass
        else:
            raise IndexError('This test should have failed.')
        f.close()

    def test_issue300(self):
        f = Dataset(self.file,'w')
        nlats = 11; lat = f.createDimension('lat',nlats)
        nlons = 20; lon = f.createDimension('lon',nlons)
        time = f.createDimension('time',None)
        var = f.createVariable('var',np.float,('time','lat','lon'))
        a = np.random.uniform(size=(3,nlats,nlons))
        var[[True,True,False,False,False,True]] = a
        var[0,2.0,"-1"] = 0 # issue 312
        a[0,2,-1]=0
        f.close()
        f = Dataset(self.file)
        var = f.variables['var']
        aa = var[[0,1,5]]
        bb = var[[True,True,False,False,False,True]]
        lats = np.arange(nlats); lons = np.arange(nlons)
        cc = var[-1,lats > 2,lons < 6]
        assert_array_almost_equal(a,aa)
        assert_array_almost_equal(bb,aa)
        assert_array_almost_equal(cc,a[-1,3:,:6])
        f.close()

    def test_retain_single_dims(self):
        f = Dataset(self.file, 'r')
        v = f.variables['data']
        keys = ((0, 1, 2, 3, 4, 5, 6, 7, 8), (5,), (4,))
        shape = (9, 1, 1)
        data = v[keys]
        assert_equal(data.shape, shape)
        keys = ((0, 1, 2, 3, 4, 5, 6, 7, 8), 5, 4,)
        shape = (9,)
        data = v[keys]
        assert_equal(data.shape, shape)
        f.close()

if __name__ == '__main__':
    unittest.main()
