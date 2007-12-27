import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.random.mtrand import uniform 
import netCDF3

# test primitive data types.

# create an n1dim by n2dim random ranarr.
FILE_NAME = tempfile.mktemp(".nc")
n1dim = 5
n2dim = 10
ranarr = 100.*uniform(size=(n1dim,n2dim))
datatypes = ['f8','f4','i1','i2','i4','S1']
FillValue = 1.0

class PrimitiveTypesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        file = netCDF3.Dataset(self.file,'w')
        file.createDimension('n1', None)
        file.createDimension('n2', n2dim)
        for type in datatypes:
            foo = file.createVariable('data_'+type, type, ('n1','n2',),fill_value=FillValue)
            #foo._FillValue = FillValue
            # test writing of _FillValue attribute for diff types
            # (should be cast to type of variable silently)
            foo[1:n1dim] = ranarr[1:n1dim]
        file.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing primitive data type """ 
        file = netCDF3.Dataset(self.file)
        for type in datatypes:
            data = file.variables['data_'+type]
            datarr = data[1:n1dim]
            # fill missing data with _FillValue
            # ('S1' array will have some missing values)
            if hasattr(datarr, 'mask'):
                datarr = datarr.filled()
            datfilled = data[0]
            # check to see that data type is correct
            self.assert_(data.dtype.str[1:] == type)
            # check data in variable.
            if data.dtype.str[1:] != 'S1':
                #assert NP.allclose(datarr, ranarr[1:n1dim].astype(data.dtype))
                assert_array_almost_equal(datarr,ranarr[1:n1dim].astype(data.dtype))
            else:
                assert datarr.tostring() == ranarr[1:n1dim].astype(data.dtype).tostring()
            # check that variable elements not yet written are filled
            # with the specified _FillValue.
            assert_array_equal(datfilled,data._FillValue)
        file.close()

if __name__ == '__main__':
    unittest.main()
