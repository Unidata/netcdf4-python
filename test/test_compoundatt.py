import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset, CompoundType
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

# test compound attributes.

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
DIM_NAME = 'time'
VAR_NAME = 'wind'
VAR_NAME2 = 'forecast_wind'
GROUP_NAME = 'forecasts'
dtype=np.dtype([('speed', 'f4'), ('direction', 'f4')])
TYPE_NAME = 'wind_vector_type'
TYPE_NAMEC = 'wind_vectorunits_type'
dtypec=np.dtype([('speed', 'S8'), ('direction', 'S8')])
missvals = np.empty(1,dtype)
missvals['direction']=1.e20
missvals['speed']=-999.
windunits = np.empty(1,dtypec)
windunits['speed'] = 'm/s'
windunits['direction'] = 'degrees'

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
        assert_array_almost_equal(v.missing_values['speed'], missvals['speed'])
        assert_array_almost_equal(v.missing_values['direction'],\
                missvals['direction'])
        assert_array_almost_equal(vv.missing_values['speed'], missvals['speed'])
        assert_array_almost_equal(vv.missing_values['direction'],\
                missvals['direction'])
        assert_array_equal(v.units['speed'], windunits['speed'].squeeze())
        assert_array_equal(v.units['direction'],\
                windunits['direction'].squeeze())
        assert_array_equal(vv.units['speed'], windunits['speed'].squeeze())
        assert_array_equal(vv.units['direction'],\
                windunits['direction'].squeeze())
        assert(v.units['speed'] == b'm/s')
        assert(v.units['direction'] == b'degrees')
        assert(vv.units['speed'] == b'm/s')
        assert(vv.units['direction'] == b'degrees')
        f.close()

if __name__ == '__main__':
    unittest.main()
