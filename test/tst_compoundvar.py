import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset, CompoundType
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

# test compound data types.

FILE_NAME = tempfile.mktemp(".nc")
DIM_NAME = 'phony_dim'
GROUP_NAME = 'phony_group'
VAR_NAME = 'phony_compound_var'
TYPE_NAME1 = 'cmp1'
TYPE_NAME2 = 'cmp2'
TYPE_NAME3 = 'cmp3'
TYPE_NAME4 = 'cmp4'
TYPE_NAME5 = 'cmp5'
dtype1=np.dtype([('i', 'i2'), ('j', 'i8')])
dtype2=np.dtype([('xx', 'f4'), ('yy', 'f8', (3,2))])
dtype3=np.dtype([('xxx', dtype1), ('yyy', dtype2, (4,))])
dtype4=np.dtype([('x', 'f4'), ('y', 'f8', (2,3)), ('z', dtype3, (2,2))])
dtype5=np.dtype([('x1', 'f4'), ('y1', 'f8', (2,3)), ('z1', dtype3, (2,2))])
data = np.zeros(10,dtype4)
data['x']=1
data['z']['xxx']['i'][:]=2
datag = np.zeros(10,dtype5)
datag['z1']['xxx']['i'][:]=3

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = Dataset(self.file, 'w')
        d = f.createDimension(DIM_NAME,None)
        g = f.createGroup(GROUP_NAME)
        # multiply nested compound types
        cmptype1 = f.createCompoundType(dtype1, TYPE_NAME1)
        cmptype2 = f.createCompoundType(dtype2, TYPE_NAME2)
        cmptype3 = f.createCompoundType(dtype3, TYPE_NAME3)
        cmptype4 = f.createCompoundType(dtype4, TYPE_NAME4)
        cmptype5 = g.createCompoundType(dtype5, TYPE_NAME5)
        v = f.createVariable(VAR_NAME,cmptype4, DIM_NAME)
        vv = g.createVariable(VAR_NAME,cmptype5, DIM_NAME)
        v[:] = data
        vv[:] = datag
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing compound variables"""
        f = Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
        g = f.groups[GROUP_NAME]
        vv = g.variables[VAR_NAME]
        dataout = v[:]
        dataoutg = vv[:]
        assert(v.dtype == dtype4)
        assert(vv.dtype == dtype5)
        assert_array_almost_equal(dataout['x'],data['x'])
        assert_array_equal(dataout['z']['xxx']['i'],data['z']['xxx']['i'])
        assert_array_equal(dataout['z']['xxx']['i'],data['z']['xxx']['i'])
        assert_array_equal(dataoutg['z1']['xxx']['i'],datag['z1']['xxx']['i'])
        f.close()

if __name__ == '__main__':
    unittest.main()
