import random
import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.testing import assert_array_equal
import netCDF4

# test NC_STRING ('S') primitive data type.

chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
FILE_NAME = tempfile.mktemp(".nc")
GROUP_NAME = "subgroup"
VAR_NAME="dummy_var"
DIM1_NAME="x"
DIM1_LEN=3
DIM2_NAME="y"
DIM2_LEN=3
data = NP.empty(DIM1_LEN*DIM2_LEN,'O')
for n in range(DIM1_LEN*DIM2_LEN):
    stringlen = random.randint(3,12)
    data[n] = ''.join([random.choice(chars) for i in range(stringlen)])
data[0] = {'spam':1,'eggs':2} # will be pickled into a string.
data = NP.reshape(data,(DIM1_LEN,DIM2_LEN))

class StringVariableTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netCDF4.Dataset(self.file,'w')
        f.createDimension(DIM1_NAME, DIM1_LEN)
        f.createDimension(DIM2_NAME, DIM2_LEN)
        v = f.createVariable(VAR_NAME, 'S', (DIM1_NAME,DIM2_NAME))
        v[:] = data
        v[-1,-1] = 'hello'
        g = f.createGroup(GROUP_NAME)
        g.createDimension(DIM1_NAME, DIM1_LEN)
        g.createDimension(DIM2_NAME, DIM2_LEN)
        vg = g.createVariable(VAR_NAME, 'S', (DIM1_NAME,DIM2_NAME))
        vg[:] = data
        vg[-1,-1] = 'hello'
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing string ('S') data type"""
        f  = netCDF4.Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
        self.assert_(v.shape == data.shape)
        self.assert_(v.dtype == 'S')
        data[-1,-1] = 'hello'
        datout = v[:]
        self.assert_(datout.dtype.char == 'O')
        assert_array_equal(datout, data)
        g = f.groups[GROUP_NAME]
        vg = g.variables[VAR_NAME] 
        self.assert_(vg.shape == data.shape)
        self.assert_(vg.dtype == 'S')
        datout = vg[:]
        self.assert_(datout.dtype.char == 'O')
        assert_array_equal(datout, data)
        f.close()

if __name__ == '__main__':
    unittest.main()
