import math
import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform 
import netCDF4_classic as netCDF4

# test attribute creation.

FILE_NAME = tempfile.mktemp(".nc")
VAR_NAME="dummy_var"
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
        f.createDimension(DIM1_NAME, DIM1_LEN)
        f.createDimension(DIM2_NAME, DIM2_LEN)
        f.createDimension(DIM3_NAME, DIM3_LEN)
        v = f.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v.stratt = STRATT
        v.emptystratt = EMPTYSTRATT
        v.intatt = INTATT
        v.floatatt = FLOATATT
        v.seqatt = SEQATT
        v.stringseqatt = STRINGSEQATT
        v.objatt = OBJATT
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing attributes"""
        f  = netCDF4.Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
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
        assert v.emptystratt == EMPTYSTRATT
        assert v.seqatt.tolist() == SEQATT.tolist()
        assert v.stringseqatt == ''.join(STRINGSEQATT)
        assert v.objatt == OBJATT
        f.close()

if __name__ == '__main__':
    unittest.main()
