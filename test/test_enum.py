import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_array_equal

FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
ENUM_NAME = 'cloud_t'
ENUM_BASETYPE = np.int8
VAR_NAME = 'primary_cloud'
ENUM_DICT = {'Altocumulus': 7, 'Missing': 127, 'Stratus': 2, 'Clear': 0,
'Nimbostratus': 6, 'Cumulus': 4, 'Altostratus': 5, 'Cumulonimbus': 1,
'Stratocumulus': 3}
datain = np.array([ENUM_DICT['Clear'],ENUM_DICT['Stratus'],ENUM_DICT['Cumulus'],\
                   ENUM_DICT['Missing'],ENUM_DICT['Cumulonimbus']],dtype=ENUM_BASETYPE)
datain_masked = np.ma.masked_values(datain,ENUM_DICT['Missing'])


class EnumTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = Dataset(self.file,'w')
        cloud_type = f.createEnumType(ENUM_BASETYPE,ENUM_NAME,ENUM_DICT)
        # make sure KeyError raised if non-integer basetype used.
        try:
            cloud_typ2 = f.createEnumType(np.float32,ENUM_NAME,ENUM_DICT)
        except KeyError:
            pass
        f.createDimension('time',None)
        cloud_var =\
        f.createVariable(VAR_NAME,cloud_type,'time',\
                         fill_value=ENUM_DICT['Missing'])
        cloud_var[:] = datain_masked
        # make sure ValueError raised if illegal value assigned to Enum var.
        try:
            cloud_var[cloud_var.shape[0]] = 99
        except ValueError:
            pass
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing enum data type"""
        f = Dataset(self.file, 'r')
        v = f.variables[VAR_NAME]
        assert v.datatype.enum_dict == ENUM_DICT
        assert list(f.enumtypes.keys()) == [ENUM_NAME]
        assert f.enumtypes[ENUM_NAME].name == ENUM_NAME # issue 775
        assert f.enumtypes[ENUM_NAME].dtype == ENUM_BASETYPE
        assert v._FillValue == ENUM_DICT['Missing']
        v.set_auto_mask(False)
        data = v[:]
        assert_array_equal(data, datain)
        v.set_auto_mask(True) # check to see if auto masking works
        data = v[:]
        assert_array_equal(data, datain_masked)
        assert_array_equal(data.mask, datain_masked.mask)
        f.close()

class EnumDictTestCase(unittest.TestCase):
    # issue 1128
    def setUp(self):
        DT = np.int16; BITS = 8
        self.STORED_VAL = DT(2**BITS)
        self.VAL_MAP = {f'bits_{n}': DT(2**n) for n in range(1,BITS+1)}
        self.VAL_MAP['invalid'] = 0
        self.file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
        with netCDF4.Dataset(file, 'w') as nc:
            # The enum is created with dtype=int16, so it will allow BITS values up to 15
            et = nc.createEnumType(DT, 'etype', self.VAL_MAP)
            ev = nc.createVariable('evar', et)
            # Succeeds because the created EnumType does keep the correct dict
            ev[...] = self.STORED_VAL
        def tearDown(self):
            os.remove(self.file)
        def runTest(self):
            with netCDF4.Dataset(file, 'r') as nc:
                read_var = nc['evar']
                assert read_var[...] == self.STORED_VAL 
                assert read_et.enum_dict == self.VAL_MAP 

if __name__ == '__main__':
    unittest.main()
