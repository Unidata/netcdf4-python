import sys
import unittest
import os
import tempfile
from netCDF4 import Dataset
import numpy as np
from numpy.testing import assert_array_equal

FILE_NAME = tempfile.mktemp(".nc")
ENUM_NAME = 'cloud_t'
ENUM_BASETYPE = np.int8
VAR_NAME = 'primary_cloud'
ENUM_DICT = {u'Altocumulus': 7, u'Missing': 127, u'Stratus': 2, u'Clear': 0,
u'Nimbostratus': 6, u'Cumulus': 4, u'Altostratus': 5, u'Cumulonimbus': 1,
u'Stratocumulus': 3} 
datain = np.array([ENUM_DICT['Clear'],ENUM_DICT['Stratus'],ENUM_DICT['Cumulus'],\
                   ENUM_DICT['Missing'],ENUM_DICT['Cumulonimbus']],dtype=ENUM_BASETYPE)


class EnumTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = Dataset(self.file,'w')
        cloud_type = f.createEnumType(ENUM_BASETYPE,ENUM_NAME,ENUM_DICT)
        f.createDimension('time',None)
        cloud_var =\
        f.createVariable(VAR_NAME,cloud_type,'time',\
                         fill_value=ENUM_DICT['Missing'])
        cloud_var[:] = datain
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
        assert f.enumtypes[ENUM_NAME].dtype == ENUM_BASETYPE
        assert v._FillValue == ENUM_DICT['Missing']
        data = v[:]
        assert_array_equal(data, datain)
        f.close()

if __name__ == '__main__':
    unittest.main()
