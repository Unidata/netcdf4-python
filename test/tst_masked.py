import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy import ma
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.random.mtrand import uniform 
import netCDF4

# test automatic conversion of masked arrays, and
# packing/unpacking of short ints.

# create an n1dim by n2dim random ranarr.
FILE_NAME = tempfile.mktemp(".nc")
ndim = 10
ranarr = 100.*uniform(size=(ndim))
packeddata = 10.*uniform(size=(ndim))
missing_value = -9999.
ranarr[::2] = missing_value
maskedarr = ma.masked_values(ranarr,-9999.)
scale_factor = (packeddata.max()-packeddata.min())/(2.*32766.)
add_offset = 0.5*(packeddata.max()+packeddata.min())
packeddata2 = ((packeddata-add_offset)/scale_factor).astype('i2')

class PrimitiveTypesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        file = netCDF4.Dataset(self.file,'w')
        file.createDimension('n', ndim)
        foo = file.createVariable('maskeddata', 'f8', ('n',))
        foo.missing_value = missing_value
        foo.set_auto_maskandscale(True)
        bar = file.createVariable('packeddata', 'i2', ('n',))
        bar.set_auto_maskandscale(True)
        bar.scale_factor = scale_factor
        bar.add_offset = add_offset
        foo[:] = maskedarr
        bar[:] = packeddata
        file.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """testing auto-conversion of masked arrays and packed integers""" 
        file = netCDF4.Dataset(self.file)
        datamasked = file.variables['maskeddata']
        datapacked = file.variables['packeddata']
        # check missing_value, scale_factor and add_offset attributes.
        assert datamasked.missing_value == missing_value
        assert datapacked.scale_factor == scale_factor
        assert datapacked.add_offset == add_offset
        # no auto-conversion.
        datamasked.set_auto_maskandscale(False)
        datapacked.set_auto_maskandscale(False)
        assert_array_equal(datapacked[:],packeddata2)
        assert_array_almost_equal(datamasked[:],ranarr)
        # auto-conversion
        datamasked.set_auto_maskandscale(True)
        datapacked.set_auto_maskandscale(True)
        assert_array_almost_equal(datamasked[:].filled(),ranarr)
        assert_array_almost_equal(datapacked[:],packeddata,decimal=4)
        file.close()

if __name__ == '__main__':
    unittest.main()
