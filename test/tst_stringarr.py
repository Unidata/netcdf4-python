from netCDF4 import Dataset, stringtochar, chartostring
import random, numpy, string
import unittest
import os
from numpy.testing import assert_array_equal, assert_array_almost_equal

def generateString(length, alphabet=string.ascii_letters + string.digits + string.punctuation):
    return(''.join([random.choice(alphabet) for i in range(length)]))

# test conversion of arrays of fixed-length strings
# to arrays of characters (with an extra dimension), and vice-versa.

FILE_NAME = 'tst_stringarr.nc'
FILE_FORMAT = 'NETCDF4_CLASSIC'
n2 = 10; nchar = 12; nrecs = 4
data = numpy.empty((nrecs,n2),'S'+repr(nchar))
for nrec in range(nrecs):
    for n in range(n2):
        data[nrec,n] = generateString(nchar)

class StringArrayTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        nc = Dataset(FILE_NAME,'w',format=FILE_FORMAT)
        nc.createDimension('n1',None)
        nc.createDimension('n2',n2)
        nc.createDimension('nchar',nchar)
        v = nc.createVariable('strings','S1',('n1','n2','nchar'))
        v2 = nc.createVariable('strings2','S1',('n1','n2','nchar'))
        # if _Encoding set, string array should automatically be converted
        # to a char array and vice-versan
        v2._Encoding = 'ascii'
        for nrec in range(nrecs):
            datac = stringtochar(data,encoding='ascii')
            v[nrec] = datac[nrec]
        v2[:-1] = data[:-1]
        v2[-1] = data[-1]
        nc.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing functions for converting arrays of chars to fixed-len strings"""
        nc = Dataset(FILE_NAME)
        assert nc.dimensions['n1'].isunlimited() == True
        v = nc.variables['strings']
        v2 = nc.variables['strings2']
        assert v.dtype.str[1:] in ['S1','U1']
        assert v.shape == (nrecs,n2,nchar)
        datau = data.astype('U')
        for nrec in range(nrecs):
            data2 = chartostring(v[nrec],encoding='ascii')
            assert_array_equal(data2,datau[nrec])
        data2 = v2[:]
        data2[0] = v2[0]
        data2[0,1] = v2[0,1]
        assert_array_equal(data2,datau)
        nc.close()

if __name__ == '__main__':
    unittest.main()
