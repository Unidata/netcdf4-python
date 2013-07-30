import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.random.mtrand import uniform 
import netCDF4

# test primitive data types.

# create an n1dim by n2dim random ranarr.
FILE_NAME = tempfile.mktemp(".nc")
n1dim = 5
n2dim = 10
ranarr = 100.*uniform(size=(n1dim,n2dim))
zlib=False;complevel=0;shuffle=0;least_significant_digit=None
datatypes = ['f8','f4','i1','i2','i4','i8','u1','u2','u4','u8','S1']
FillValue = 1.0

class PrimitiveTypesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        file = netCDF4.Dataset(self.file,'w')
        file.createDimension('n1', None)
        file.createDimension('n2', n2dim)
        for typ in datatypes:
            foo = file.createVariable('data_'+typ, typ, ('n1','n2',),zlib=zlib,complevel=complevel,shuffle=shuffle,least_significant_digit=least_significant_digit,fill_value=FillValue)
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
        file = netCDF4.Dataset(self.file)
        for typ in datatypes:
            data = file.variables['data_'+typ]
            data.set_auto_maskandscale(False)
            datarr = data[1:n1dim]
            # fill missing data with _FillValue
            # ('S1' array will have some missing values)
            if hasattr(datarr, 'mask'):
                datarr = datarr.filled()
            datfilled = data[0]
            # check to see that data type is correct
            if typ == 'S1':
                self.assertTrue(data.dtype.str[1:] in ['S1','U1'])
            else:
                self.assertTrue(data.dtype.str[1:] == typ)
            # check data in variable.
            if data.dtype.str[1:] != 'S1':
                #assert NP.allclose(datarr, ranarr[1:n1dim].astype(data.dtype))
                assert_array_almost_equal(datarr,ranarr[1:n1dim].astype(data.dtype))
            else:
                assert datarr.tostring() == ranarr[1:n1dim].astype(data.dtype).tostring()
            # check that variable elements not yet written are filled
            # with the specified _FillValue.
            assert_array_equal(datfilled,NP.asarray(data._FillValue,datfilled.dtype))
        file.close()

if __name__ == '__main__':
    unittest.main()
