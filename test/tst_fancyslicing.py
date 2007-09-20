from netCDF4 import Dataset
from numpy.random import seed, randint
from numpy.testing import assert_array_equal
import tempfile, unittest, os, random
import numpy as NP

file_name = tempfile.mktemp(".nc")
xdim=9; ydim=10; zdim=11
i = NP.array([2,5,7],'i4')
ib = NP.array(ydim*[False])
ib[2] = True; ib[5] = True; ib[7] = True
#seed(9) # fix seed
data = randint(0,10,size=(xdim,ydim,zdim)).astype('u1')

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = file_name
        f = Dataset(file_name,'w')
        f.createDimension('x',xdim)
        f.createDimension('y',ydim)
        f.createDimension('z',zdim)
        v = f.createVariable('data','i2',('x','y','z'))
        v[:] = data
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing 'fancy indexing'"""
        f  = Dataset(self.file, 'r')
        v = f.variables['data']
        # slice with an array of integers.
        assert_array_equal(v[0:-1:2,i,:],data[0:-1:2,i,:])
        # slice with an array of booleans.
        assert_array_equal(v[0:-1:2,ib,:],data[0:-1:2,ib,:])

if __name__ == '__main__':
    unittest.main()
