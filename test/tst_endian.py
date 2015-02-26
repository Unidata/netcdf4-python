import netCDF4
import numpy as np
import unittest, os, tempfile
from numpy.testing import assert_array_equal, assert_array_almost_equal

data = np.arange(12,dtype='f4').reshape(3,4)
FILE_NAME = tempfile.mktemp(".nc")
FILE_NAME2 = tempfile.mktemp(".nc")

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
    var_big_endian.missing_value = mval
    var_big_endian[0]=np.pi
    var_big_endian[1]=mval
    var_native_endian = nc.createVariable(\
             'obs_native_endian', '<f8', ('obs', ),\
             endian='native',fill_value=fval)
    var_native_endian.missing_value = mval
    var_native_endian[0]=np.pi
    var_native_endian[1]=mval
    assert_array_almost_equal(var_native_endian[:].filled(),
                              var_big_endian[:].filled())
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

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)
        os.remove(self.file2)

    def runTest(self):
        """testing endian conversion capability"""
        check_data(self.file, data)
        check_data(self.file2, data)
        issue310(self.file)
        issue346(self.file2)

if __name__ == '__main__':
    unittest.main()
