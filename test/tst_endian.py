import netCDF4
import numpy as np
import unittest, os, tempfile
from numpy.testing import assert_array_equal, assert_array_almost_equal

data = np.arange(12,dtype='f4').reshape(3,4)
FILE_NAME = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME2 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
FILE_NAME3 = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

def create_file(file,format,data):
    import warnings
    dataset = netCDF4.Dataset(file,'w',format=format)
    dataset.createDimension('time', None)
    dataset.createDimension('space', 4)
    dims = ('time', 'space')
    little = data.astype('<f4')
    big = data.astype('>f4')
    warnings.simplefilter('ignore') # ignore UserWarnings generated below
    ll = dataset.createVariable('little-little', '<f4', dims)
    lb = dataset.createVariable('little-big', '<f4', dims)
    bl = dataset.createVariable('big-little', '>f4', dims)
    bb = dataset.createVariable('big-big', '>f4', dims)
    ll[:] = little
    lb[:] = big
    bl[:] = little
    bb[:] = big
    dataset.close()

def check_byteswap(file, data):
    # byteswapping is done internally to native endian format
    # when numpy array has non-native byte order.  The byteswap was
    # initially done in place, which caused the numpy array to
    # be modified in the calling program.  Pull request #555
    # changed the byteswap to a copy, and this test checks
    # to make sure the input numpy array is not modified.
    dataset = netCDF4.Dataset(file,'w')
    dataset.createDimension('time', None)
    dataset.createDimension('space', 4)
    dims = ('time', 'space')
    bl = dataset.createVariable('big-little', np.float32, dims, endian='big')
    data2 = data.copy()
    bl[:] = data
    dataset.close()
    f = netCDF4.Dataset(file)
    bl = f.variables['big-little'][:]
    # check data.
    assert_array_almost_equal(data, data2)
    assert_array_almost_equal(bl, data)
    f.close()


def check_data(file, data):
    f = netCDF4.Dataset(file)
    ll = f.variables['little-little'][:]
    lb = f.variables['little-big'][:]
    bb = f.variables['big-big'][:]
    bl = f.variables['big-little'][:]
    # check data.
    assert_array_almost_equal(ll, data)
    assert_array_almost_equal(lb, data)
    assert_array_almost_equal(bl, data)
    assert_array_almost_equal(bb, data)
    f.close()

def issue310(file):
    mval = 999.; fval = -999
    nc = netCDF4.Dataset(file, "w")
    nc.createDimension('obs', 10)
    if netCDF4.is_native_little:
        endian='big'
    elif netCDF4.is_native_big:
        endian='little'
    else:
        raise ValueError('cannot determine native endianness')
    var_big_endian = nc.createVariable(\
            'obs_big_endian', '>f8', ('obs', ),\
            endian=endian,fill_value=fval)
    # use default _FillValue
    var_big_endian2 = nc.createVariable(\
            'obs_big_endian2', '>f8', ('obs', ),\
            endian=endian)
    # NOTE: missing_value  be written in same byte order
    # as variable, or masked array won't be masked correctly
    # when data is read in.
    var_big_endian.missing_value = mval
    var_big_endian[0]=np.pi
    var_big_endian[1]=mval
    var_big_endian2.missing_value = mval
    var_big_endian2[0]=np.pi
    var_big_endian2[1]=mval
    var_native_endian = nc.createVariable(\
             'obs_native_endian', '<f8', ('obs', ),\
             endian='native',fill_value=fval)
    var_native_endian.missing_value = mval
    var_native_endian[0]=np.pi
    var_native_endian[1]=mval
    assert_array_almost_equal(var_native_endian[:].filled(),
                              var_big_endian[:].filled())
    assert_array_almost_equal(var_big_endian[:].filled(),
                              var_big_endian2[:].filled())
    nc.close()

def issue346(file):
    # create a big and a little endian variable
    xb = np.arange(10, dtype='>i4')
    xl = np.arange(xb.size, dtype='<i4')
    nc = netCDF4.Dataset(file, mode='w')
    nc.createDimension('x', size=xb.size)
    vb=nc.createVariable('xb', xb.dtype, ('x'),
                         endian='big')
    vl=nc.createVariable('xl', xl.dtype, ('x'),
                         endian='little')
    nc.variables['xb'][:] = xb
    nc.variables['xl'][:] = xl
    nc.close()
    nc = netCDF4.Dataset(file)
    datab = nc.variables['xb'][:]
    datal = nc.variables['xl'][:]
    assert_array_equal(datab,xb)
    assert_array_equal(datal,xl)
    nc.close()

class EndianTestCase(unittest.TestCase):

    def setUp(self):
        create_file(FILE_NAME,'NETCDF4_CLASSIC',data); self.file=FILE_NAME
        create_file(FILE_NAME2,'NETCDF3_CLASSIC',data); self.file2=FILE_NAME2
        self.file3 = FILE_NAME3

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
        os.remove(self.file2)
        os.remove(self.file3)

    def runTest(self):
        """testing endian conversion capability"""
        check_data(self.file, data)
        check_data(self.file2, data)
        check_byteswap(self.file3, data)
        issue310(self.file)
        issue346(self.file2)

if __name__ == '__main__':
    unittest.main()
