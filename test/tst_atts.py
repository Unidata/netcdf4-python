import math
import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform
import netCDF4

try:
    from collections import OrderedDict
except ImportError: # or else use drop-in substitute
    from ordereddict import OrderedDict

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
ATTDICT = {'stratt':STRATT,'floatatt':FLOATATT,'seqatt':SEQATT,
           'stringseqatt':''.join(STRINGSEQATT),
           'emptystratt':EMPTYSTRATT,'intatt':INTATT}

class VariablesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netCDF4.Dataset(self.file,'w')
        # try to set a dataset attribute with one of the reserved names.
        f.setncattr('file_format','netcdf4_format')
        # test attribute renameing
        f.stratt_tmp = STRATT
        f.renameAttribute('stratt_tmp','stratt')
        f.emptystratt = EMPTYSTRATT
        f.intatt = INTATT
        f.floatatt = FLOATATT
        f.seqatt = SEQATT
        # sequences of strings converted to a single string.
        f.stringseqatt = STRINGSEQATT
        g = f.createGroup(GROUP_NAME)
        f.createDimension(DIM1_NAME, DIM1_LEN)
        f.createDimension(DIM2_NAME, DIM2_LEN)
        f.createDimension(DIM3_NAME, DIM3_LEN)
        g.createDimension(DIM1_NAME, DIM1_LEN)
        g.createDimension(DIM2_NAME, DIM2_LEN)
        g.createDimension(DIM3_NAME, DIM3_LEN)
        g.stratt_tmp = STRATT
        g.renameAttribute('stratt_tmp','stratt')
        g.emptystratt = EMPTYSTRATT
        g.intatt = INTATT
        g.floatatt = FLOATATT
        g.seqatt = SEQATT
        g.stringseqatt = STRINGSEQATT
        v = f.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        # try to set a variable attribute with one of the reserved names.
        v.setncattr('ndim','three')
        v.setncatts({'foo': 1})
        v.setncatts(OrderedDict(bar=2))
        v.stratt_tmp = STRATT
        v.renameAttribute('stratt_tmp','stratt')
        v.emptystratt = EMPTYSTRATT
        v.intatt = INTATT
        v.floatatt = FLOATATT
        v.seqatt = SEQATT
        v.stringseqatt = STRINGSEQATT
        v1 = g.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v1.stratt = STRATT
        v1.emptystratt = EMPTYSTRATT
        v1.intatt = INTATT
        v1.floatatt = FLOATATT
        v1.seqatt = SEQATT
        v1.stringseqatt = STRINGSEQATT
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
        # global attributes.
        # check __dict__ method for accessing all netCDF attributes.
        for key,val in ATTDICT.items():
            if type(val) == NP.ndarray:
                assert f.__dict__[key].tolist() == val.tolist()
            else:
                assert f.__dict__[key] == val
        # check accessing individual attributes.
        assert f.intatt == INTATT
        assert f.floatatt == FLOATATT
        assert f.stratt == STRATT
        assert f.emptystratt == EMPTYSTRATT
        assert f.seqatt.tolist() == SEQATT.tolist()
        assert f.stringseqatt == ''.join(STRINGSEQATT)
        assert f.getncattr('file_format') == 'netcdf4_format'
        # variable attributes.
        # check __dict__ method for accessing all netCDF attributes.
        for key,val in ATTDICT.items():
            if type(val) == NP.ndarray:
                assert v.__dict__[key].tolist() == val.tolist()
            else:
                assert v.__dict__[key] == val
        # check accessing individual attributes.
        assert v.intatt == INTATT
        assert v.floatatt == FLOATATT
        assert v.stratt == STRATT
        assert v.seqatt.tolist() == SEQATT.tolist()
        assert v.stringseqatt == ''.join(STRINGSEQATT)
        assert v.getncattr('ndim') == 'three'
        assert v.getncattr('foo') == 1
        assert v.getncattr('bar') == 2
        # check attributes in subgroup.
        # global attributes.
        for key,val in ATTDICT.items():
            if type(val) == NP.ndarray:
                assert g.__dict__[key].tolist() == val.tolist()
            else:
                assert g.__dict__[key] == val
        assert g.intatt == INTATT
        assert g.floatatt == FLOATATT
        assert g.stratt == STRATT
        assert g.emptystratt == EMPTYSTRATT
        assert g.seqatt.tolist() == SEQATT.tolist()
        assert g.stringseqatt == ''.join(STRINGSEQATT)
        for key,val in ATTDICT.items():
            if type(val) == NP.ndarray:
                assert v1.__dict__[key].tolist() == val.tolist()
            else:
                assert v1.__dict__[key] == val
        assert v1.intatt == INTATT
        assert v1.floatatt == FLOATATT
        assert v1.stratt == STRATT
        assert v1.emptystratt == EMPTYSTRATT
        assert v1.seqatt.tolist() == SEQATT.tolist()
        assert v1.stringseqatt == ''.join(STRINGSEQATT)
        assert getattr(v1,'nonexistantatt',None) == None
        f.close()

if __name__ == '__main__':
    unittest.main()
