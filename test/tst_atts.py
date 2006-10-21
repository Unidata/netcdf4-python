import math
import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform 
import netCDF4

# test attribute creation.
FILE_NAME = tempfile.mktemp(".nc")
VAR_NAME="dummy_var"
GROUP_NAME = "dummy_group"
DIM1_NAME="x"
DIM1_LEN=2
DIM2_NAME="y"
DIM2_LEN=3
DIM3_NAME="z"
DIM3_LEN=25
STRATT = 'string attribute'
EMPTYSTRATT = ''
INTATT = 1
FLOATATT = math.pi
SEQATT = NP.arange(10)
STRINGSEQATT = ['mary ','had ','a ','little ','lamb']
OBJATT = {'spam':1,'eggs':2}

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netCDF4.Dataset(self.file,'w')
        f.stratt = STRATT
        f.emptystratt = EMPTYSTRATT
        f.intatt = INTATT
        f.floatatt = FLOATATT
        f.seqatt = SEQATT
        # sequences of strings converted to a single string.
        f.stringseqatt = STRINGSEQATT
        # python objects that cannot be cast to numpy arrays are stored
        # as pickled strings (and unpickled when accessed).
        f.objatt = OBJATT
        g = f.createGroup(GROUP_NAME)
        f.createDimension(DIM1_NAME, DIM1_LEN)
        f.createDimension(DIM2_NAME, DIM2_LEN)
        f.createDimension(DIM3_NAME, DIM3_LEN)
        g.createDimension(DIM1_NAME, DIM1_LEN)
        g.createDimension(DIM2_NAME, DIM2_LEN)
        g.createDimension(DIM3_NAME, DIM3_LEN)
        g.stratt = STRATT
        g.emptystratt = EMPTYSTRATT
        g.intatt = INTATT
        g.floatatt = FLOATATT
        g.seqatt = SEQATT
        g.stringseqatt = STRINGSEQATT
        g.objatt = OBJATT
        v = f.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v.stratt = STRATT
        v.emptystratt = EMPTYSTRATT
        v.intatt = INTATT
        v.floatatt = FLOATATT
        v.seqatt = SEQATT
        v.stringseqatt = STRINGSEQATT
        v.objatt = OBJATT
        v1 = g.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v1.stratt = STRATT
        v1.emptystratt = EMPTYSTRATT
        v1.intatt = INTATT
        v1.floatatt = FLOATATT
        v1.seqatt = SEQATT
        v1.stringseqatt = STRINGSEQATT
        v1.objatt = OBJATT
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing attributes"""
        f  = netCDF4.Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
        g = f.groups[GROUP_NAME]
        v1 = g.variables[VAR_NAME]
        # check attributes in root group.
        assert f.intatt == INTATT
        assert f.floatatt == FLOATATT
        assert f.stratt == STRATT
        assert f.emptystratt == EMPTYSTRATT
        assert f.seqatt.tolist() == SEQATT.tolist()
        assert f.stringseqatt == ''.join(STRINGSEQATT)
        assert f.objatt == OBJATT
        assert v.intatt == INTATT
        assert v.floatatt == FLOATATT
        assert v.stratt == STRATT
        assert v.seqatt.tolist() == SEQATT.tolist()
        assert v.stringseqatt == ''.join(STRINGSEQATT)
        assert v.objatt == OBJATT
        # check attributes in subgroup.
        assert g.intatt == INTATT
        assert g.floatatt == FLOATATT
        assert g.stratt == STRATT
        assert g.emptystratt == EMPTYSTRATT
        assert g.seqatt.tolist() == SEQATT.tolist()
        assert g.stringseqatt == ''.join(STRINGSEQATT)
        assert g.objatt == OBJATT
        assert v1.intatt == INTATT
        assert v1.floatatt == FLOATATT
        assert v1.stratt == STRATT
        assert v1.emptystratt == EMPTYSTRATT
        assert v1.seqatt.tolist() == SEQATT.tolist()
        assert v1.stringseqatt == ''.join(STRINGSEQATT)
        assert v1.objatt == OBJATT
        f.close()

if __name__ == '__main__':
    unittest.main()
