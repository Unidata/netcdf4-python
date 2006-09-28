from netCDF4 import Dataset
from numpy.random import seed, randint
from numpy.testing import assert_array_equal
import tempfile, unittest, os, random
import numpy as NP

file_name = tempfile.mktemp(".nc")
xdim=9; ydim=10; zdim=11
#seed(9) # fix seed
data = randint(0,10,size=(xdim,ydim,zdim)).astype('u1')
datarev = data[:,::-1,:]
datao = NP.empty(xdim*ydim*zdim,'O')
datas = NP.empty(xdim*ydim*zdim,'O')
chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
for n in range(xdim*ydim*zdim):
    m = randint(1,20)
    # arrays of random length between 1 and 20
    datao[n] = (NP.arange(m+1)+1).astype('u1')
    # random strings of length 1 to 20
    datas[n] = ''.join([random.choice(chars) for i in range(m)])
datao = NP.reshape(datao,(xdim,ydim,zdim))
datas = NP.reshape(datas,(xdim,ydim,zdim))
datarevo = datao[:,::-1,:]
datarevs = datas[:,::-1,:]

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = file_name
        f = Dataset(file_name,'w')
        f.createDimension('x',xdim)
        f.createDimension('xu',None)
        f.createDimension('y',ydim)
        f.createDimension('yu',None)
        f.createDimension('z',zdim)
        f.createDimension('zu',None)
        v = f.createVariable('data','u1',('x','y','z'))
        vu = f.createVariable('datau','u1',('xu','y','zu'))
        vus = f.createVariable('datasu','S',('xu','y','zu'))
        vs = f.createVariable('datas','S',('x','y','z'))
        vlen = f.createUserType('u1','vlen','vlen1')
        vv = f.createVariable('datav', vlen, ('x','y','z'))
        vuv = f.createVariable('datavu', vlen, ('xu','y','zu'))
        # variable with no unlimited dim.
        # write slice in reverse order
        v[:,::-1,:] = data
        vv[:,::-1,:] = datao
        vs[:,::-1,:] = datas
        # variable with an unlimited dimension.
        # write slice in reverse order
        vu[0:xdim,::-1,0:zdim] = data
        vuv[0:xdim,::-1,0:zdim] = datao
        vus[0:xdim,::-1,0:zdim] = datas
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing variable slicing"""
        f  = Dataset(self.file, 'r')
        v = f.variables['data']
        vs = f.variables['datas']
        vv = f.variables['datav']
        vu = f.variables['datau']
        vus = f.variables['datasu']
        vuv = f.variables['datavu']
        assert_array_equal(v[:], datarev)
        for d1,d2 in zip(vv[:].flat,datarevo.flat):
            assert_array_equal(d1,d2)
        assert_array_equal(vs[:], datarevs)
        # test reading of slices.
        # negative value means count back from end.
        assert_array_equal(v[:-1,:-2,:-3],datarev[:-1,:-2,:-3])
        assert_array_equal(vs[:-1,:-2,:-3],datarevs[:-1,:-2,:-3])
        #assert_array_equal(vv[:-1,:-2,:-3],datarevo[:-1,:-2,:-3])
        for d1,d2 in zip(vv[:-1,:-2,:-3].flat,datarevo[:-1,:-2,:-3].flat):
            assert_array_equal(d1,d2)
        # every other element (positive step)
        assert_array_equal(v[2:-1:2,2:-2:2,2:-3:2],datarev[2:-1:2,2:-2:2,2:-3:2])
        # only strides of 1 work for string and vlen variables!
        assert_array_equal(vs[2:-1:1,2:-2:1,2:-3:1],datarevs[2:-1:1,2:-2:1,2:-3:1])
        for d1,d2 in zip(vv[2:-1:1,2:-2:1,2:-3:1].flat,datarevo[2:-1:1,2:-2:1,2:-3:1].flat):
            assert_array_equal(d1,d2)
        # every other element (negative step)
        assert_array_equal(v[-1:2:-2,-2:2:-2,-3:2:-2],datarev[-1:2:-2,-2:2:-2,-3:2:-2])
        # only strides of 1 work for string and vlen variables!
        assert_array_equal(vs[-1:2:-1,-2:2:-1,-3:2:-1],datarevs[-1:2:-1,-2:2:-1,-3:2:-1])
        for d1,d2 in zip(vv[-1:2:-1,-2:2:-1,-3:2:-1].flat,datarevo[-1:2:-1,-2:2:-1,-3:2:-1].flat):
            assert_array_equal(d1,d2)
        # read elements in reverse order
        assert_array_equal(v[:,::-1,:],data)
        assert_array_equal(vs[:,::-1,:],datas)
        for d1,d2 in zip(vv[:,::-1,:].flat,datao.flat):
            assert_array_equal(d1,d2)
        assert_array_equal(v[::-1,:,::-1],datarev[::-1,:,::-1])
        assert_array_equal(vs[::-1,:,::-1],datarevs[::-1,:,::-1])
        for d1,d2 in zip(vv[::-1,:,::-1].flat,datarevo[::-1,:,::-1].flat):
            assert_array_equal(d1,d2)
        assert_array_equal(v[xdim-1::-3,:,zdim-1::-3],datarev[xdim-1::-3,:,zdim-1::-3])           # only strides of 1 work for string and vlen variables!
        assert_array_equal(vs[xdim-1::-1,:,zdim-1::-1],datarevs[xdim-1::-1,:,zdim-1::-1])
        for d1,d2 in zip(vv[xdim-1::-1,:,zdim-1::-1].flat,datarevo[xdim-1::-1,:,zdim-1::-1].flat):
            assert_array_equal(d1,d2)
        # ellipsis slice.
        assert_array_equal(v[...,2:],datarev[...,2:])
        assert_array_equal(vs[...,2:],datarevs[...,2:])
        for d1,d2 in zip(vv[...,2:].flat,datarevo[...,2:].flat):
            assert_array_equal(d1,d2)
        # variable with an unlimited dimension.
        assert_array_equal(vu[:], data[:,::-1,:])
        assert_array_equal(vus[:], datas[:,::-1,:])
        for d1,d2 in zip(vuv[:].flat,datao[:,::-1,:].flat):
            assert_array_equal(d1,d2)
        # read data in reverse order
        assert_array_equal(vu[:,::-1,:],data)
        assert_array_equal(vus[:,::-1,:],datas)
        for d1,d2 in zip(vv[:,::-1,:].flat,datao.flat):
            assert_array_equal(d1,d2)

if __name__ == '__main__':
    unittest.main()
