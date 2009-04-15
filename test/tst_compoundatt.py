import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset, CompoundType, chartostring
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

# test compound attributes.

FILE_NAME = tempfile.mktemp(".nc")
DIM_NAME = 'time'
VAR_NAME = 'wind'
VAR_NAME2 = 'forecast_wind'
GROUP_NAME = 'forecasts'
dtype=np.dtype([('eastward', 'f4'), ('northward', 'f4')])
TYPE_NAME = 'wind_vector_type'
TYPE_NAMEC = 'wind_vectorunits_type'
dtypec=np.dtype([('eastward', 'c',(8,)), ('northward', 'c',(8,))])
missvals = np.zeros(1,dtype)
missvals['eastward']=9999.
missvals['northward']=-9999.
chararr = np.array(list('%-08s'%'m/s'))
windunits = np.zeros(1,dtypec)
windunits['eastward'] = chararr
windunits['northward'] = chararr

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = Dataset(self.file, 'w')
        d = f.createDimension(DIM_NAME,None)
        g = f.createGroup(GROUP_NAME)
        wind_vector_type = f.createCompoundType(dtype, TYPE_NAME)
        wind_vectorunits_type = f.createCompoundType(dtypec, TYPE_NAMEC)
        v = f.createVariable(VAR_NAME,wind_vector_type, DIM_NAME)
        vv = g.createVariable(VAR_NAME2,wind_vector_type,DIM_NAME)
        v.missing_values = missvals
        v.units = windunits
        vv.missing_values = missvals
        vv.units = windunits
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing compound attributes"""
        f = Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
        g = f.groups[GROUP_NAME]
        vv = g.variables[VAR_NAME2]
        assert_array_almost_equal(v.missing_values.item(), missvals.item())
        assert_array_almost_equal(vv.missing_values.item(), missvals.item())
        assert_array_equal(v.units, windunits)
        assert_array_equal(vv.units, windunits)
        assert(chartostring(v.units['eastward']).item().rstrip() == 'm/s')
        f.close()

if __name__ == '__main__':
    unittest.main()
