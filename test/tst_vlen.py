import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform 
from numpy.testing import assert_array_equal, assert_array_almost_equal
import netCDF4

# test vlen user-defined data type.
FILE_NAME = tempfile.mktemp(".nc")
VAR_NAME="dummy_var1"
VAR_NAME2="dummy_var2"
VLEN_NAME='vlen1'
DIM1_NAME="x"
DIM1_LEN=3
DIM2_NAME="y"
DIM2_LEN=3

data = NP.empty(DIM1_LEN*DIM2_LEN,'O')
for n in range(DIM1_LEN*DIM2_LEN):
    data[n] = NP.arange(n+1)+1
data = NP.reshape(data,(DIM1_LEN,DIM2_LEN))

class VlenTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netCDF4.Dataset(self.file,'w')
        f.createDimension(DIM1_NAME, DIM1_LEN)
        f.createDimension(DIM2_NAME, DIM2_LEN)
        vlen = f.createUserType('i4','vlen','vlen1')
        v = f.createVariable(VAR_NAME, vlen, (DIM1_NAME,DIM2_NAME))
        v[:] = data
        v[-1,-1] = [-99,-98,-97]
        #v[-1,-1] = NP.array([-99,-98,-97])
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing variables"""
        f  = netCDF4.Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
        self.assert_(v.usertype == 'vlen')
        self.assert_(v.dtype.base_datatype == 'i4')
        self.assert_(v.usertype_name == 'vlen1')
        self.assert_(v.shape == (DIM1_LEN, DIM2_LEN))
        datout = v[:]
        self.assert_(datout.dtype.char == 'O')
        self.assert_(datout.shape == (DIM1_LEN, DIM2_LEN))
        self.assert_(datout[0,0].dtype.str[1:] == v.dtype.base_datatype)
        data[-1,-1] = NP.array([-99,-98,-97])
        for d1,d2 in zip(data.flat,datout.flat):
            assert_array_equal(d1,d2)
        f.close()

if __name__ == '__main__':
    unittest.main()
