from netCDF4 import Dataset, stringtochar, chartostring
import random, numpy
import unittest
import os
from numpy.testing import assert_array_equal, assert_array_almost_equal

# test utilities for converting arrays of fixed-length strings
# to arrays of characters (with an extra dimension), and vice-versa.

FILE_NAME = 'tst_stringarr.nc'
FILE_FORMAT = 'NETCDF4_CLASSIC'
chars = '1234567890aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
n2 = 10; nchar = 12; nrecs = 4
data = numpy.empty((nrecs,n2),'S'+repr(nchar))
for nrec in range(nrecs):
    for n in range(n2):
        data[nrec,n] = ''.join([random.choice(chars) for i in range(nchar)])

class StringArrayTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        nc = Dataset(FILE_NAME,'w',format=FILE_FORMAT)
        nc.createDimension('n1',None)
        nc.createDimension('n2',n2)
        nc.createDimension('nchar',nchar)
        v = nc.createVariable('strings','S1',('n1','n2','nchar'))
        for nrec in range(nrecs):
            datac = stringtochar(data)
            v[nrec] = datac[nrec]
        nc.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing functions for converting arrays of chars to fixed-len strings"""
        nc = Dataset(FILE_NAME)
        assert nc.dimensions['n1'].isunlimited() == True
        v = nc.variables['strings']
        assert v.dtype.str[1:] in ['S1','U1']
        assert v.shape == (nrecs,n2,nchar)
        for nrec in range(nrecs):
            data2 = chartostring(v[nrec])
            assert_array_equal(data2,data[nrec])
        nc.close()

if __name__ == '__main__':
    unittest.main()
