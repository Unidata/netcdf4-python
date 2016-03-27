import sys
import unittest
import os
import tempfile
import numpy as np
from numpy import ma, seterr
from numpy.testing import assert_array_equal, assert_array_almost_equal
from netCDF4 import Dataset, default_fillvals

seterr(over='ignore') # don't print warning for overflow errors

# test automatic conversion of masked arrays, and
# packing/unpacking of short ints.

FILE_NAME1 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME2 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME3 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
datacheck1 =\
ma.array([0,5000.0,4000.0,0],dtype=np.float,mask=[True,False,False,True])
datacheck2 =\
ma.array([3000.0,5000.0,4000.0,0],dtype=np.float,mask=[False,False,False,True])
datacheck3 =\
ma.array([3000.0,5000.0,0,2000.0],dtype=np.float,mask=[False,False,True,False])
mask = [False,True,False,False]
datacheck4 = ma.array([1.5625,0,3.75,4.125],mask=mask,dtype=np.float32)
fillval = default_fillvals[datacheck4.dtype.str[1:]]
datacheck5 = np.array([1.5625,fillval,3.75,4.125],dtype=np.float32)

class PrimitiveTypesTestCase(unittest.TestCase):

    def setUp(self):

        self.files = [FILE_NAME1]
        f = Dataset(FILE_NAME1,'w')
        x = f.createDimension('x',None)
        v = f.createVariable('v',np.int16,'x')
        v.scale_factor = np.array(1,np.float32)
        v.add_offset = np.array(32066,np.float32)
        v.missing_value = np.array(-9999,v.dtype)
        #v[0] not set, will be equal to _FillValue
        v[1]=5000
        v[2]=4000
        v[3]=v.missing_value
        f.close()

        self.files.append(FILE_NAME2)
        f = Dataset(FILE_NAME1,'r')
        # create a new file, copy data, but change missing value and
        # scale factor offset.
        f2 = Dataset(FILE_NAME2,'w')
        a = f2.createDimension('a',None)
        b = f2.createVariable('b',np.int16,'a')
        b.scale_factor = np.array(10.,np.float32)
        b.add_offset = np.array(0,np.float32)
        b.missing_value = np.array(9999,v.dtype)
        b[:] = f.variables['v'][:]
        f.close()
        f2.close()

        self.files.append(FILE_NAME3)
        f = Dataset(FILE_NAME3,'w')
        x = f.createDimension('x',None)
        # create variable with lossy compression
        v = f.createVariable('v',np.float32,'x',zlib=True,least_significant_digit=1)
        # assign masked array to that variable with one missing value.
        data =\
        ma.array([1.5678,99.99,3.75145,4.127654],mask=np.array([False,True,False,False],np.bool))
        data.mask[1]=True
        v[:] = data
        f.close()

    def tearDown(self):
        # Remove the temporary files
        for f in self.files:
            os.remove(f)

    def runTest(self):
        """testing auto-conversion of masked arrays and packed integers"""

        f = Dataset(self.files[0])
        data = f.variables['v'][:]
        assert_array_almost_equal(data,datacheck1)
        f.close()

        f = Dataset(self.files[1])
        data = f.variables['b'][:]
        assert_array_almost_equal(data,datacheck1)
        f.close()

        f = Dataset(self.files[0],'a')
        # change first element from _FillValue to actual data.
        v = f.variables['v']
        v[0]=3000
        f.close()
        f = Dataset(self.files[0],'r')
        # read data back in, check.
        data = f.variables['v'][:]
        assert_array_almost_equal(data,datacheck2)
        f.close()

        f = Dataset(self.files[0],'a')
        # change 3rd element to missing, 4 element to valid data.
        v = f.variables['v']
        data = v[:]
        v[2]=-9999
        v[3]=2000
        f.close()
        f = Dataset(self.files[0],'r')
        # read data back in, check.
        data = f.variables['v'][:]
        assert_array_almost_equal(data,datacheck3)
        f.close()

        # check that masked arrays are handled correctly when lossy compression
        # is used.
        f = Dataset(self.files[2],'r')
        data = f.variables['v'][:]
        assert_array_almost_equal(data,datacheck4)
        assert_array_almost_equal(data.filled(),datacheck5)
        f.close()

if __name__ == '__main__':
    unittest.main()
