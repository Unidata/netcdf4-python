import unittest, os, tempfile
import netCDF4
from numpy.testing import assert_array_equal
import numpy as np

fill_val = np.array(9.9e31)

# test Variable.get_fill_value

class TestGetFillValue(unittest.TestCase):
    def setUp(self):
        self.testfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name
        f = netCDF4.Dataset(self.testfile, 'w')
        dim = f.createDimension('x',10)
        for dt in netCDF4.default_fillvals.keys():
            if not dt.startswith('c'):
                v = f.createVariable(dt+'_var',dt,dim)
        v = f.createVariable('float_var',np.float64,dim,fill_value=fill_val)
        # test fill_value='default' option (issue #1374)
        v2 = f.createVariable('float_var2',np.float64,dim,fill_value='default')
        f.close()
        
    def tearDown(self):
        os.remove(self.testfile)

    def runTest(self):
        f = netCDF4.Dataset(self.testfile, "r")
        # no _FillValue set, test that default fill value returned
        for dt in netCDF4.default_fillvals.keys():
            if not dt.startswith('c'):
                fillval = np.array(netCDF4.default_fillvals[dt])
                if dt == 'S1': fillval = fillval.astype(dt)
                v = f[dt+'_var']
                assert_array_equal(fillval, v.get_fill_value())
        # _FillValue attribute is set.
        v = f['float_var']
        assert_array_equal(fill_val, v.get_fill_value())
        v = f['float_var2']
        assert_array_equal(np.array(netCDF4.default_fillvals['f8']), v._FillValue)
        f.close()

if __name__ == '__main__':
    unittest.main()
