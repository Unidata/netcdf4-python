import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform
from numpy.testing import assert_array_equal, assert_array_almost_equal
import netCDF4

# test variable creation.

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
VAR_DOUBLE_NAME="dummy_var"
VAR_SHORT_NAME='dummy_var_short'
VARNAMES = sorted([VAR_DOUBLE_NAME,VAR_SHORT_NAME])
GROUP_NAME = "dummy_group"
DIM1_NAME="x"
DIM1_LEN=2
DIM2_NAME="y"
DIM2_LEN=3
DIM3_NAME="z"
DIM3_LEN=25

randomdata = uniform(size=(DIM1_LEN,DIM2_LEN,DIM3_LEN))

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f  = netCDF4.Dataset(self.file, 'w')
        f.createDimension(DIM1_NAME, DIM1_LEN)
        f.createDimension(DIM2_NAME, DIM2_LEN)
        f.createDimension(DIM3_NAME, DIM3_LEN)
        v1 = f.createVariable(VAR_DOUBLE_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v2 = f.createVariable(VAR_SHORT_NAME, 'i2',(DIM2_NAME,DIM3_NAME))
        v1.long_name = 'dummy data root'
        g = f.createGroup(GROUP_NAME)
        g.createDimension(DIM1_NAME, DIM1_LEN)
        g.createDimension(DIM2_NAME, DIM2_LEN)
        g.createDimension(DIM3_NAME, DIM3_LEN)
        v1g = g.createVariable(VAR_DOUBLE_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v2g = g.createVariable(VAR_SHORT_NAME, 'i2',(DIM2_NAME,DIM3_NAME))
        v1g.long_name = 'dummy data subgroup'
        v1[:] = randomdata
        v1g[:] = randomdata
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing primitive variables"""
        f  = netCDF4.Dataset(self.file, 'r')
        # check variables in root group.
        varnames = sorted(f.variables.keys())
        v1 = f.variables[VAR_DOUBLE_NAME]
        v2 = f.variables[VAR_SHORT_NAME]
        assert varnames == VARNAMES
        assert v1.dtype.str[1:] == 'f8'
        assert v2.dtype.str[1:] == 'i2'
        assert v1.long_name == 'dummy data root'
        assert v1.dimensions == (DIM1_NAME,DIM2_NAME,DIM3_NAME)
        assert v2.dimensions == (DIM2_NAME,DIM3_NAME)
        assert v1.shape == (DIM1_LEN,DIM2_LEN,DIM3_LEN)
        assert v2.shape == (DIM2_LEN,DIM3_LEN)
        assert v1.size == DIM1_LEN * DIM2_LEN * DIM3_LEN
        assert len(v1) == DIM1_LEN

        #assert NP.allclose(v1[:],randomdata)
        assert_array_almost_equal(v1[:],randomdata)
        # check variables in sub group.
        g = f.groups[GROUP_NAME]
        varnames = sorted(g.variables.keys())
        v1 = g.variables[VAR_DOUBLE_NAME]
        # test iterating over variable (should stop when
        # it gets to the end and raises IndexError, issue 121)
        for v in v1:
            pass
        v2 = g.variables[VAR_SHORT_NAME]
        assert varnames == VARNAMES
        assert v1.dtype.str[1:] == 'f8'
        assert v2.dtype.str[1:] == 'i2'
        assert v1.long_name == 'dummy data subgroup'
        assert v1.dimensions == (DIM1_NAME,DIM2_NAME,DIM3_NAME)
        assert v2.dimensions == (DIM2_NAME,DIM3_NAME)
        assert v1.shape == (DIM1_LEN,DIM2_LEN,DIM3_LEN)
        assert v2.shape == (DIM2_LEN,DIM3_LEN)
        #assert NP.allclose(v1[:],randomdata)
        assert_array_almost_equal(v1[:],randomdata)
        f.close()

if __name__ == '__main__':
    unittest.main()
