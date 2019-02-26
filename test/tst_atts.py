import math
import subprocess
import sys
import unittest
import os
import tempfile
import warnings

import numpy as NP
from numpy.random.mtrand import uniform
import netCDF4

try:
    from collections import OrderedDict
except ImportError: # or else use drop-in substitute
    from ordereddict import OrderedDict

# test attribute creation.
FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
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
STRINGSEQATT = ['mary ','','had ','a ','little ','lamb',]
#ATTDICT = {'stratt':STRATT,'floatatt':FLOATATT,'seqatt':SEQATT,
#           'stringseqatt':''.join(STRINGSEQATT), # changed in issue #770
#           'emptystratt':EMPTYSTRATT,'intatt':INTATT}
ATTDICT = {'stratt':STRATT,'floatatt':FLOATATT,'seqatt':SEQATT,
           'stringseqatt':STRINGSEQATT,
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
        f.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
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
        if netCDF4.__version__ > "1.4.2":
            with self.assertRaises(ValueError):
                g.arrayatt = [[1, 2], [3, 4]] # issue #841
        g.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
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
        v.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
        v1 = g.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v1.stratt = STRATT
        v1.emptystratt = EMPTYSTRATT
        v1.intatt = INTATT
        v1.floatatt = FLOATATT
        v1.seqatt = SEQATT
        v1.stringseqatt = STRINGSEQATT
        v1.setncattr_string('stringseqatt_array',STRINGSEQATT) # array of NC_STRING
        # issue #485 (triggers segfault in C lib
        # with version 1.2.1 without pull request #486)
        f.foo = NP.array('bar','S')
        f.foo = NP.array('bar','U')
        # issue #529 write string attribute as NC_CHAR unless
        # it can't be decoded to ascii.  Add setncattr_string
        # method to force NC_STRING.
        f.charatt = u'foo' # will be written as NC_CHAR
        f.setncattr_string('stringatt','bar') # NC_STRING
        f.cafe = u'caf\xe9' # NC_STRING
        f.batt = u'caf\xe9'.encode('utf-8') #NC_CHAR
        v.setncattr_string('stringatt','bar') # NC_STRING
        # issue #882 - provide an option to always string attribute
        # as NC_STRINGs. Testing various approaches to setting text attributes...
        f.set_ncstring_attrs(True)
        f.stringatt_ncstr = u'foo' # will now be written as NC_STRING
        f.setncattr_string('stringatt_ncstr','bar') # NC_STRING anyway
        f.caf_ncstr = u'caf\xe9' # NC_STRING anyway
        f.bat_ncstr = u'caf\xe9'.encode('utf-8') # now NC_STRING
        g.stratt_ncstr = STRATT # now NC_STRING
        #g.renameAttribute('stratt_tmp','stratt_ncstr')
        v.setncattr_string('stringatt_ncstr','bar') # NC_STRING anyway
        v.stratt_ncstr = STRATT
        v1.emptystratt_ncstr = EMPTYSTRATT
        f.close()

    def tearDown(self):
        # Remove the temporary files
        #pass
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
        #assert f.stringseqatt == ''.join(STRINGSEQATT) # issue 770
        assert f.stringseqatt == STRINGSEQATT
        assert f.stringseqatt_array == STRINGSEQATT
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
        #assert v.stringseqatt == ''.join(STRINGSEQATT) # issue 770
        assert v.stringseqatt == STRINGSEQATT
        assert v.stringseqatt_array == STRINGSEQATT
        assert v.getncattr('ndim') == 'three'
        assert v.getncattr('foo') == 1
        assert v.getncattr('bar') == 2
        # check type of attributes using ncdump (issue #529)
        try:  # ncdump may not be on the system PATH
            nc_proc = subprocess.Popen(
                ['ncdump', '-h', FILE_NAME], stdout=subprocess.PIPE)
        except OSError:
            warnings.warn('"ncdump" not on system path; cannot test '
                          'read of some attributes')
            pass
        else:  # We do have ncdump output
            dep = nc_proc.communicate()[0]
            try: # python 2
                ncdump_output = dep.split('\n')
            except TypeError: # python 3
                ncdump_output = str(dep,encoding='utf-8').split('\n')
            for line in ncdump_output:
                line = line.strip('\t\n\r')
                line = line.strip()# Must be done another time for group variables
                if "stringatt" in line: assert line.startswith('string')
                if "charatt" in line: assert line.startswith(':')
                if "cafe" in line: assert line.startswith('string')
                if "batt" in line: assert line.startswith(':')
                if "_ncstr" in line: assert line.startswith('string')
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
        #assert g.stringseqatt == ''.join(STRINGSEQATT) # issue 770
        assert g.stringseqatt == STRINGSEQATT
        assert g.stringseqatt_array == STRINGSEQATT
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
        #assert v1.stringseqatt == ''.join(STRINGSEQATT) # issue 770
        assert v1.stringseqatt == STRINGSEQATT
        assert v1.stringseqatt_array == STRINGSEQATT
        assert getattr(v1,'nonexistantatt',None) == None
        f.close()

if __name__ == '__main__':
    unittest.main()
