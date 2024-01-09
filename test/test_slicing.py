from netCDF4 import Dataset
from numpy.random import seed, randint
from numpy.testing import assert_array_equal, assert_equal,\
assert_array_almost_equal
import tempfile, unittest, os, random, sys
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
        f.createDimension('xu2',None)
        f.createDimension('y',ydim)
        f.createDimension('z',zdim)
        f.createDimension('zu',None)
        v = f.createVariable('data','u1',('x','y','z'))
        vu = f.createVariable('datau','u1',('xu','y','zu'))
        v1 = f.createVariable('data1d', 'u1', ('x',))
        v2 = f.createVariable('data1dx', 'u1', ('xu2',))
        # variable with no unlimited dim.
        # write slice in reverse order
        v[:,::-1,:] = data
        # variable with an unlimited dimension.
        # write slice in reverse order
        #vu[0:xdim,::-1,0:zdim] = data
        vu[:,::-1,:] = data

        v1[:] = data[:, 0, 0]
        if sys.maxsize > 2**32:
            v2[2**31] = 1 # issue 1112 (overflow on windows)
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
        v2 = f.variables['data1dx']
        d = data[:,0,0]
        assert_equal(v1[:], d)
        if sys.maxsize > 2**32:
            assert_equal(v2[2**31], 1)
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
        # issue #785: always return masked array
        #assert(type(v[...]) == np.ndarray)
        assert(type(v[...]) == np.ma.core.MaskedArray)
        f.set_auto_mask(False)
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
        var = f.createVariable('var',np.float64,('time','lev','lat','lon'))
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
        var = f.createVariable('var',np.float64,('time','lat','lon'))
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

    def test_issue743(self):
        nc = Dataset(self.file,'w',format='NETCDF3_CLASSIC')
        td = nc.createDimension('t',None)
        xd = nc.createDimension('x',33)
        yd = nc.createDimension('y',4)
        v = nc.createVariable('v',np.float64,('t','x','y'))
        nc.close()
        nc = Dataset(self.file)
        data = np.empty(nc['v'].shape, nc['v'].dtype)
        data2 = nc['v'][...]
        assert_array_equal(data,data2)
        nc.close()

    def test_issue906(self):
        f = Dataset(self.file,'w')
        f.createDimension('d1',3)
        f.createDimension('d2',None)
        f.createDimension('d3',5)
        f.createVariable('v2',np.float64,('d1','d2','d3'))
        f['v2'][:] = np.zeros((3,4,5))
        f['v2'][0,:,0] = np.arange(4)
        f['v2'][0,:,:] = np.ones((4,5))
        f.close()

    def test_issue919(self):
        with Dataset(self.file,'w') as f:
            f.createDimension('time',2)
            f.createDimension('lat',10)
            f.createDimension('lon',9)
            f.createVariable('v1',np.int64,('time', 'lon','lat',))
            arr = np.arange(9*10).reshape((9, 10))
            f['v1'][:] = arr
            assert_array_equal(f['v1'][:],np.broadcast_to(arr,f['v1'].shape))
            arr = np.arange(10)
            f['v1'][:] = arr
            assert_array_equal(f['v1'][:],np.broadcast_to(arr,f['v1'].shape))

    def test_issue922(self):
        with Dataset(self.file,'w') as f:
            f.createDimension('d1',3)
            f.createDimension('d2',None)
            f.createVariable('v1',np.int64,('d2','d1',))
            f['v1'][0] = np.arange(3,dtype=np.int64)
            f['v1'][1:3] = np.arange(3,dtype=np.int64)
            assert_array_equal(f['v1'][:], np.broadcast_to(np.arange(3),(3,3)))
            f.createVariable('v2',np.int64,('d1','d2',))
            f['v2'][:,0] = np.arange(3,dtype=np.int64)
            f['v2'][:,1:3] = np.arange(6,dtype=np.int64).reshape(3,2)
            assert_array_equal(f['v2'][:,1:3],np.arange(6,dtype=np.int64).reshape(3,2))
            assert_array_equal(f['v2'][:,0],np.arange(3,dtype=np.int64))

    def test_issue1083(self):
        with Dataset(self.file, "w") as nc:
            nc.createDimension("test", 5)
            v = nc.createVariable("var", "f8", ("test", "test", "test"))
            v[:] = 1 # works
            v[:] = np.ones(()) # works
            v[:] = np.ones((1,)) # works
            v[:] = np.ones((5,)) # works
            v[:] = np.ones((5,5,5)) # works
            v[:] = np.ones((5,1,1)) # fails (before PR #1084)
            v[:] = np.ones((5,1,5)) # fails (before PR #1084)
            v[:] = np.ones((5,5,1)) # fails (before PR #1084)

if __name__ == '__main__':
    unittest.main()
