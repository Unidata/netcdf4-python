import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset, CompoundType
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

# test compound data types.

FILE_NAME = tempfile.mktemp(".nc")
#FILE_NAME = 'test.nc'
DIM_NAME = 'phony_dim'
GROUP_NAME = 'phony_group'
VAR_NAME = 'phony_compound_var'
TYPE_NAME1 = 'cmp1'
TYPE_NAME2 = 'cmp2'
TYPE_NAME3 = 'cmp3'
TYPE_NAME4 = 'cmp4'
TYPE_NAME5 = 'cmp5'
DIM_SIZE=1 # FIXME: segfaults on linux if this set > 1
# FIXME:  must use align=True or test fails. 
dtype1=np.dtype([('i', 'i2'), ('j', 'i8')],align=True)
dtype2=np.dtype([('x', 'f4',), ('y', 'f8',(3,2))],align=True)
dtype3=np.dtype([('xx', dtype1), ('yy', dtype2)],align=True)
dtype4=np.dtype([('xxx',dtype3),('yyy','f8', (4,))],align=True)
dtype5=np.dtype([('x1', dtype1), ('y1', dtype2)],align=True)
data = np.zeros(DIM_SIZE,dtype4)
data['xxx']['xx']['i']=1
data['xxx']['xx']['j']=2
data['xxx']['yy']['x']=3
data['xxx']['yy']['y']=4
data['yyy'] = 5
datag = np.zeros(DIM_SIZE,dtype5)
datag['x1']['i']=10
datag['x1']['j']=20
datag['y1']['x']=30
datag['y1']['y']=40

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = Dataset(self.file, 'w')
        d = f.createDimension(DIM_NAME,DIM_SIZE)
        g = f.createGroup(GROUP_NAME)
        # multiply nested compound types
        cmptype1 = f.createCompoundType(dtype1, TYPE_NAME1)
        cmptype2 = f.createCompoundType(dtype2, TYPE_NAME2)
        cmptype3 = f.createCompoundType(dtype3, TYPE_NAME3)
        cmptype4 = f.createCompoundType(dtype4, TYPE_NAME4)
        cmptype5 = f.createCompoundType(dtype5, TYPE_NAME5)
        v = f.createVariable(VAR_NAME,cmptype4, DIM_NAME)
        vv = g.createVariable(VAR_NAME,cmptype5, DIM_NAME)
        v[:] = data
        vv[:] = datag
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
        #pass

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
        assert_array_equal(dataout['xxx']['xx']['i'],data['xxx']['xx']['i'])
        assert_array_equal(dataout['xxx']['xx']['j'],data['xxx']['xx']['j'])
        assert_array_almost_equal(dataout['xxx']['yy']['x'],data['xxx']['yy']['x'])
        assert_array_almost_equal(dataout['xxx']['yy']['y'],data['xxx']['yy']['y'])
        assert_array_almost_equal(dataout['yyy'],data['yyy'])
        assert_array_equal(dataoutg['x1']['i'],datag['x1']['i'])
        assert_array_equal(dataoutg['x1']['j'],datag['x1']['j'])
        assert_array_almost_equal(dataoutg['y1']['x'],datag['y1']['x'])
        assert_array_almost_equal(dataoutg['y1']['y'],datag['y1']['y'])
        f.close()

if __name__ == '__main__':
    unittest.main()
