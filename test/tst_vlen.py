import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_array_equal
from datetime import datetime

FILE_NAME = tempfile.mktemp(".nc")
VL_NAME = 'vlen_type'
VL_BASETYPE = np.int32
DIM1_NAME = 'lon'
DIM2_NAME = 'lat'
nlons = 4; nlats = 3
VAR1_NAME = 'ragged'
VAR2_NAME = 'strings'
data = np.empty(nlats*nlons,object)
datas = np.empty(nlats*nlons,object)
nn = 0
for n in range(nlats*nlons):
    nn = nn + 1
    data[n] = np.arange(nn)
    datas[n] = ''.join([chr(i) for i in range(97,97+nn+1)])
datas[0] = datetime.now() # should be converted to/from pickle string.
data = np.reshape(data,(nlats,nlons))
datas = np.reshape(datas,(nlats,nlons))

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = Dataset(self.file,'w')
        vlen_t = f.createVLType(VL_BASETYPE, VL_NAME)
        f.createDimension(DIM1_NAME,nlons)
        f.createDimension(DIM2_NAME,nlats)
        ragged = f.createVariable(VAR1_NAME, vlen_t, (DIM2_NAME,DIM1_NAME))
        strings = f.createVariable(VAR2_NAME, str, (DIM2_NAME,DIM1_NAME))
        ragged[:] = data
        strings[:] = datas
        ragged[nlats-1,nlons-1] = data[-1,-1]
        strings[nlats-1,nlons-1] = datas[-1,-1]
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing vlen variables"""
        f = Dataset(self.file, 'r')
        v = f.variables[VAR1_NAME]
        vs = f.variables[VAR2_NAME]
        assert f.vltypes.keys() == [VL_NAME]
        assert f.vltypes[VL_NAME].dtype == VL_BASETYPE
        data2 = v[:]
        data2s = vs[:]
        for i in range(nlons):
            for j in range(nlats):
                assert_array_equal(data2[j,i], data[j,i])
                assert datas[j,i] == data2s[j,i]
        f.close()

if __name__ == '__main__':
    unittest.main()
