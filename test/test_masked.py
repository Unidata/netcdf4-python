import sys
import unittest
import os
import tempfile
import numpy as np
from numpy import ma
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.random.mtrand import uniform
import netCDF4
from numpy.ma import masked_all
import pathlib

# test automatic conversion of masked arrays, and
# packing/unpacking of short ints.

# create an n1dim by n2dim random ranarr.
FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME2 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME3 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
ndim = 10
ranarr = 100.*uniform(size=(ndim))
ranarr2 = 100.*uniform(size=(ndim))
# used for checking vector missing_values
arr3 = np.linspace(0,9,ndim)
mask = np.zeros(ndim,np.bool_); mask[-1]=True; mask[-2]=True
marr3 = np.ma.array(arr3, mask=mask, dtype=np.int32)
packeddata = 10.*uniform(size=(ndim))
missing_value = -9999.
missing_value2 = np.nan
missing_value3 = [8,9]
ranarr[::2] = missing_value
ranarr2[::2] = missing_value2
np.seterr(invalid='ignore') # silence warnings from ma.masked_values
maskedarr = ma.masked_values(ranarr,missing_value)
#maskedarr2 = ma.masked_values(ranarr2,missing_value2)
maskedarr2 = ma.masked_invalid(ranarr2)
scale_factor = (packeddata.max()-packeddata.min())/(2.*32766.)
add_offset = 0.5*(packeddata.max()+packeddata.min())
packeddata2 = np.around((packeddata-add_offset)/scale_factor).astype('i2')

class PrimitiveTypesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        self.file2 = FILE_NAME2
        self.file3 = FILE_NAME3
        file = netCDF4.Dataset(self.file,'w')
        file.createDimension('n', ndim)
        foo = file.createVariable('maskeddata', 'f8', ('n',))
        foo2 = file.createVariable('maskeddata2', 'f8', ('n',))
        foo3 = file.createVariable('maskeddata3', 'i4', ('n',))
        foo.missing_value = missing_value
        foo.set_auto_maskandscale(True)
        foo2.missing_value = missing_value2
        foo2.set_auto_maskandscale(True)
        foo3.missing_value = missing_value3
        foo3.set_auto_maskandscale(True)
        bar = file.createVariable('packeddata', 'i2', ('n',))
        bar.set_auto_maskandscale(True)
        bar.scale_factor = scale_factor
        bar.add_offset = add_offset
        foo[:] = maskedarr
        foo2[:] = maskedarr2
        foo3[:] = arr3
        bar[:] = packeddata
        # added to test fix to issue 46
        doh = file.createVariable('packeddata2','i2','n')
        doh.scale_factor = 0.1
        doh.add_offset = 0.
        doh[0] = 1.1
        # added to test fix to issue 381
        doh2 = file.createVariable('packeddata3','i2','n')
        doh2.add_offset = 1.
        doh2[0] = 1.
        # added test for issue 515
        file.createDimension('x',1)
        v = file.createVariable('v',np.float64,'x',fill_value=-9999)
        file.close()

        # issue #972: when auto_fill off byte arrays (u1,i1) should
        # not be masked, but other datatypes should.
        dataset = netCDF4.Dataset(self.file2, "w")
        dataset.set_fill_off()
        dim = dataset.createDimension("dim", 10)
        var1 = dataset.createVariable("var1", "f8", (dim.name,))
        var1[:] = masked_all((10,), "f8")
        var2 = dataset.createVariable("var2", "u1", (dim.name,))
        var2[:] = masked_all((10,), "u1")
        dataset.close()

        # issue #1152: if missing_value is a string that can't
        # be cast to the variable type, issue a warning instead
        # of raising an exception when auto-converted slice to a
        # masked array
        with netCDF4.Dataset(pathlib.Path(__file__).parent / "issue1152.nc") as dataset:
            data = dataset['v'][:]

        # issue #1271 (mask is ignored when assigning bool array to uint8 var)
        ds = netCDF4.Dataset(self.file3, "w")
        dim = ds.createDimension('time', 48)
        var = ds.createVariable('blaat', np.uint8, ('time',),
              zlib=True, complevel=4, shuffle=True, fletcher32=True,
              fill_value=240)
        mask = np.full((48,), False)
        for x in range(30):
            mask[x] = True
        mama = ma.array(np.full((48,), True), mask=mask)
        var[:] = mama
        ds.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
        os.remove(self.file2)

    def runTest(self):
        """testing auto-conversion of masked arrays and packed integers"""
        file = netCDF4.Dataset(self.file)
        datamasked = file.variables['maskeddata']
        datamasked2 = file.variables['maskeddata2']
        datamasked3 = file.variables['maskeddata3']
        datapacked = file.variables['packeddata']
        datapacked2 = file.variables['packeddata2']
        datapacked3 = file.variables['packeddata3']
        # check missing_value, scale_factor and add_offset attributes.
        assert datamasked.missing_value == missing_value
        assert datapacked.scale_factor == scale_factor
        assert datapacked.add_offset == add_offset
        # no auto-conversion.
        datamasked.set_auto_maskandscale(False)
        datamasked2.set_auto_maskandscale(False)
        datapacked.set_auto_maskandscale(False)
        assert_array_equal(datapacked[:],packeddata2)
        assert_array_equal(datamasked3[:],marr3)
        assert_array_almost_equal(datamasked[:],ranarr)
        assert_array_almost_equal(datamasked2[:],ranarr2)
        # auto-conversion
        datamasked.set_auto_maskandscale(True)
        datamasked2.set_auto_maskandscale(True)
        datapacked.set_auto_maskandscale(True)
        datapacked2.set_auto_maskandscale(False)
        assert_array_almost_equal(datamasked[:].filled(),ranarr)
        assert_array_almost_equal(datamasked2[:].filled(),ranarr2)
        assert_array_almost_equal(datapacked[:],packeddata,decimal=4)
        assert datapacked3[:].dtype == np.float64 
        # added to test fix to issue 46 (result before r865 was 10)
        assert_array_equal(datapacked2[0],11)
        # added test for issue 515
        assert file['v'][0] is np.ma.masked 
        file.close()
        # issue 766
        np.seterr(invalid='raise')
        f = netCDF4.Dataset(self.file, 'w')
        f.createDimension('dimension', 2)
        f.createVariable('variable', np.float32, dimensions=('dimension',))
        f['variable'][:] = np.nan
        data = f['variable'][:] # should not raise an error
        f.close()
        # issue #972
        dataset = netCDF4.Dataset(self.file2, "r")
        var1 = dataset.variables["var1"]
        var2 = dataset.variables["var2"]
        assert var1[:].mask.all()
        assert var2[:].mask.any() == False
        dataset.close()
        # issue #1271
        ds = netCDF4.Dataset(self.file3,"r")
        var = ds['blaat']
        assert np.count_nonzero(var[:].mask) == 30
        ds.close()

if __name__ == '__main__':
    unittest.main()
